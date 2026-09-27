"""Recording state indicator and audio level visualizer."""

import math
from collections import deque

from rich.text import Text

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

WAVEFORM_HEIGHT = 6  # braille cell rows -> 24 dot rows
COMPACT_WAVEFORM_HEIGHT = 3  # narrow-terminal strip
COMPACT_PANEL_HEIGHT = 4  # waveform 3 + state line 1, no padding
HISTORY_MAXLEN = 512  # level samples kept for the scrolling oscilloscope
IDLE_LEVEL = 0.10
FALLBACK_WIDTH = 16

# Braille cell: 2 dot columns x 4 dot rows. Bit for (dot_col, dot_row).
_BRAILLE_BASE = 0x2800
_DOT_BITS = (
    (0x01, 0x02, 0x04, 0x40),  # left column, top to bottom
    (0x08, 0x10, 0x20, 0x80),  # right column, top to bottom
)

# Multi-stop gradients: amplitude picks the position, low -> high.
_GRADIENTS = {
    "idle": ((0x3A, 0x34, 0x2C), (0x6E, 0x65, 0x58), (0xA8, 0x9B, 0x86)),
    "recording": ((0x6B, 0x3E, 0x28), (0xD4, 0x78, 0x4A), (0xF0, 0xC8, 0xA0)),
    "stopping": ((0x5C, 0x4A, 0x32), (0xC4, 0xA3, 0x6A), (0xE6, 0xD3, 0xA8)),
    "transcribing": ((0x3E, 0x48, 0x3C), (0x7D, 0x9A, 0x84), (0xD5, 0xE2, 0xC8)),
}

_EDGE_DIM = 0.35  # brightness falloff from the center line to the edges
_SHIMMER = 0.10  # spatial drift along the gradient, per column


def _gradient_color(stops: tuple, t: float, brightness: float = 1.0) -> str:
    """Interpolate a multi-stop RGB gradient at t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    scaled = t * (len(stops) - 1)
    index = min(int(scaled), len(stops) - 2)
    frac = scaled - index
    (r1, g1, b1), (r2, g2, b2) = stops[index], stops[index + 1]
    r = int((r1 + (r2 - r1) * frac) * brightness)
    g = int((g1 + (g2 - g1) * frac) * brightness)
    b = int((b1 + (b2 - b1) * frac) * brightness)
    return f"#{r:02x}{g:02x}{b:02x}"


class StatusPanel(Widget):
    """Borderless side rail: braille oscilloscope, state line, model info."""

    DEFAULT_CSS = """
    StatusPanel {
        background: #24211c;
        color: #f3eee4;
        layout: vertical;
        align: center middle;
        padding: 1 2;
        min-width: 24;
    }

    StatusPanel > Static {
        width: 100%;
        height: auto;
    }

    #waveform {
        height: 6;
    }

    #state-line {
        margin: 1 0 0 0;
    }

    #model-line {
        dock: bottom;
    }
    """

    state: reactive[str] = reactive("idle")

    def __init__(self, model_label: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._model_label = model_label
        self._levels: deque[float] = deque(maxlen=HISTORY_MAXLEN)
        self._elapsed = 0.0
        self._sine_tick: int = 0
        self._transcribe_timer = None
        self._waveform_rows = WAVEFORM_HEIGHT

    def compose(self) -> ComposeResult:
        yield Static(id="waveform")
        yield Static(id="state-line")
        yield Static(id="model-line")

    def on_mount(self) -> None:
        self._refresh_display()

    def set_model_label(self, label: str) -> None:
        self._model_label = label
        self._refresh_display()

    def set_compact(self, compact: bool) -> None:
        """Shrink the oscilloscope and hide model info for narrow strips."""
        self._waveform_rows = COMPACT_WAVEFORM_HEIGHT if compact else WAVEFORM_HEIGHT
        try:
            self.query_one("#waveform", Static).styles.height = self._waveform_rows
            self.query_one("#model-line", Static).display = not compact
            self.query_one("#state-line", Static).styles.margin = 0 if compact else (1, 0, 0, 0)
            self.styles.padding = (0, 1) if compact else (1, 2)
        except Exception:
            pass
        self._refresh_display()

    def on_resize(self, event) -> None:
        self._refresh_display()

    def watch_state(self, value: str) -> None:
        # Stop transcribing animation when leaving that state.
        if self._transcribe_timer is not None:
            self._transcribe_timer.stop()
            self._transcribe_timer = None

        self._elapsed = 0.0
        match value:
            case "recording":
                self._levels.clear()
            case "transcribing":
                self._sine_tick = 0
                self._transcribe_timer = self.set_interval(
                    0.1, self._tick_transcribe_animation
                )
            case _:
                self._sine_tick = 0
        self._refresh_display()

    def push_level(self, level: float) -> None:
        """Push a new audio level sample and refresh the waveform."""
        self._levels.append(max(0.0, min(1.0, level)))
        self._update_waveform()

    def update_elapsed(self, seconds: float) -> None:
        """Update the timer shown next to the state label."""
        self._elapsed = seconds
        self._update_state_line()

    def _tick_transcribe_animation(self) -> None:
        """Periodic callback that animates a sine wave during transcription."""
        self._sine_tick += 1
        self._update_waveform()

    def _update_waveform(self) -> None:
        try:
            self.query_one("#waveform", Static).update(self._render_waveform())
        except Exception:
            pass

    def _update_state_line(self) -> None:
        try:
            self.query_one("#state-line", Static).update(self._render_state_line())
        except Exception:
            pass

    def _refresh_display(self) -> None:
        try:
            self.query_one("#waveform", Static).update(self._render_waveform())
            self.query_one("#state-line", Static).update(self._render_state_line())
            self.query_one("#model-line", Static).update(self._render_model_line())
        except Exception:
            pass

    def _waveform_width(self) -> int:
        """Current waveform width in character cells, tracking panel size."""
        try:
            width = self.query_one("#waveform", Static).content_size.width
        except Exception:
            width = 0
        return width if width > 0 else FALLBACK_WIDTH

    def _column_levels(self, dot_cols: int) -> list[float]:
        """One level (0..1) per braille dot column for the current state."""
        match self.state:
            case "recording":
                history = list(self._levels)[-dot_cols:]
                pad = dot_cols - len(history)
                return [IDLE_LEVEL * 0.3] * pad + history
            case "stopping":
                return self._sine_levels(dot_cols, amplitude=0.08, baseline=0.04)
            case "transcribing":
                return self._sine_levels(
                    dot_cols, amplitude=0.25, baseline=0.10, speed=0.15
                )
            case _:
                return [IDLE_LEVEL] * dot_cols

    def _sine_levels(
        self,
        dot_cols: int,
        amplitude: float,
        baseline: float,
        speed: float = 0.0,
    ) -> list[float]:
        t = self._sine_tick * speed
        return [
            baseline
            + amplitude * abs(math.sin((x / max(dot_cols, 1)) * 2 * math.pi + t))
            for x in range(dot_cols)
        ]

    def _render_waveform(self) -> Text:
        """Render a symmetric braille oscilloscope around the vertical center.

        Color is two-dimensional: amplitude picks the position along a
        multi-stop gradient (with a gentle spatial shimmer per column), and
        distance from the center line dims the cell for a glow falloff.
        """
        stops = _GRADIENTS.get(self.state, _GRADIENTS["idle"])

        width = self._waveform_width()
        rows = self._waveform_rows
        dot_cols = width * 2
        dot_rows = rows * 4
        center = dot_rows // 2
        levels = self._column_levels(dot_cols)

        # Per dot column: envelope half-height in dots (>=1 keeps a center line).
        max_half = center - 1
        half_heights = [max(1, round(level * max_half)) for level in levels]

        text = Text()
        for row in range(rows):
            if row > 0:
                text.append("\n")
            # Glow falloff: character rows near the center line stay bright.
            row_mid = row * 4 + 2
            edge = abs(row_mid - center) / max(center, 1)
            brightness = 1.0 - _EDGE_DIM * edge
            for col in range(width):
                bits = 0
                peak = 0.0
                for sub_col in range(2):
                    x = col * 2 + sub_col
                    half = half_heights[x]
                    peak = max(peak, levels[x])
                    top = center - half
                    bottom = center + half
                    for sub_row in range(4):
                        y = row * 4 + sub_row
                        if top <= y < bottom:
                            bits |= _DOT_BITS[sub_col][sub_row]
                if bits:
                    shimmer = _SHIMMER * math.sin(
                        (col / max(width, 1)) * 2 * math.pi + self._sine_tick * 0.1
                    )
                    t = 0.12 + 0.88 * peak + shimmer
                    color = _gradient_color(stops, t, brightness)
                    text.append(chr(_BRAILLE_BASE + bits), style=f"bold {color}")
                else:
                    text.append(" ")
        return text

    def _render_state_line(self) -> Text:
        text = Text(justify="center")
        match self.state:
            case "idle":
                text.append("ready", style="#7d9a84")
            case "recording":
                text.append("recording", style="bold #d4784a")
                mins = int(self._elapsed) // 60
                secs = self._elapsed - (mins * 60)
                text.append(f"  {mins:02d}:{secs:04.1f}", style="#f3eee4")
            case "stopping":
                text.append("closing mic", style="#c4a36a")
            case "transcribing":
                text.append("writing", style="#c4a36a")
        return text

    def _render_model_line(self) -> Text:
        text = Text(justify="center")
        if self._model_label:
            label = self._model_label
            if self.size.width < 28:
                label = "R2T2" if "r2t2" in label.lower() else "Qwen3-ASR"
            text.append(label, style="#9c9386")
            text.append("\n16 kHz", style="#9c9386")
        return text
