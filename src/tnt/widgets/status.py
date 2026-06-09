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
COMPACT_PANEL_HEIGHT = 7  # waveform 3 + state line 2 + padding 2
HISTORY_MAXLEN = 512  # level samples kept for the scrolling oscilloscope
IDLE_LEVEL = 0.10
FALLBACK_WIDTH = 16

# Braille cell: 2 dot columns x 4 dot rows. Bit for (dot_col, dot_row).
_BRAILLE_BASE = 0x2800
_DOT_BITS = (
    (0x01, 0x02, 0x04, 0x40),  # left column, top to bottom
    (0x08, 0x10, 0x20, 0x80),  # right column, top to bottom
)


class StatusPanel(Widget):
    """Borderless side rail: braille oscilloscope, state line, model info."""

    DEFAULT_CSS = """
    StatusPanel {
        background: #161618;
        color: #f8f4ff;
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

    def set_compact(self, compact: bool) -> None:
        """Shrink the oscilloscope and hide model info for narrow strips."""
        self._waveform_rows = COMPACT_WAVEFORM_HEIGHT if compact else WAVEFORM_HEIGHT
        try:
            self.query_one("#waveform", Static).styles.height = self._waveform_rows
            self.query_one("#model-line", Static).display = not compact
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
            baseline + amplitude * abs(math.sin((x / max(dot_cols, 1)) * 2 * math.pi + t))
            for x in range(dot_cols)
        ]

    def _render_waveform(self) -> Text:
        """Render a symmetric braille oscilloscope around the vertical center."""
        palettes = {
            "idle": ("#3f4f8f", "#5f6cff", "#5ad8ff", "#6ef3ff"),
            "recording": ("#7f5dff", "#ff4fd8", "#ff9f1c", "#ffe347"),
            "stopping": ("#8a6d3b", "#ffb347", "#ffd166", "#ffe347"),
            "transcribing": ("#6d7bff", "#ff71ce", "#ffb347", "#ffe347"),
        }
        palette = palettes.get(self.state, palettes["idle"])

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
                    color = palette[min(len(palette) - 1, int(peak * len(palette)))]
                    text.append(chr(_BRAILLE_BASE + bits), style=f"bold {color}")
                else:
                    text.append(" ")
        return text

    def _render_state_line(self) -> Text:
        text = Text(justify="center")
        match self.state:
            case "idle":
                text.append("■ READY", style="bold #7afcff")
            case "recording":
                text.append("● REC", style="bold #ff5ccf")
                mins = int(self._elapsed) // 60
                secs = self._elapsed - (mins * 60)
                text.append(f"  {mins:02d}:{secs:04.1f}", style="bold #7afcff")
            case "stopping":
                text.append("◌ STOPPING MIC", style="bold #ffd166")
            case "transcribing":
                text.append("◌ TRANSCRIBING", style="bold #ffd166")
        return text

    def _render_model_line(self) -> Text:
        text = Text(justify="center")
        if self._model_label:
            text.append(self._model_label, style="#6e6e76")
            text.append(" · ", style="#3f3f46")
            text.append("16kHz", style="#6e6e76")
        return text
