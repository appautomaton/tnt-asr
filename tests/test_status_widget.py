import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tnt.widgets.status import (  # noqa: E402
    FALLBACK_WIDTH,
    WAVEFORM_HEIGHT,
    StatusPanel,
)


def _render_lines(panel: StatusPanel) -> list[str]:
    return panel._render_waveform().plain.split("\n")


def test_waveform_renders_braille_grid_in_every_state() -> None:
    panel = StatusPanel(model_label="test-model")
    for state in ("idle", "recording", "stopping", "transcribing"):
        panel.set_reactive(StatusPanel.state, state)
        lines = _render_lines(panel)
        assert len(lines) == WAVEFORM_HEIGHT
        assert all(len(line) == FALLBACK_WIDTH for line in lines)
        for line in lines:
            assert all(ch == " " or 0x2800 <= ord(ch) <= 0x28FF for ch in line)


def test_waveform_recording_uses_pushed_levels() -> None:
    panel = StatusPanel()
    panel.set_reactive(StatusPanel.state, "recording")
    quiet = panel._render_waveform().plain

    for _ in range(FALLBACK_WIDTH * 2):
        panel.push_level(1.0)
    loud = panel._render_waveform().plain

    def dot_count(rendered: str) -> int:
        return sum(
            bin(ord(ch) - 0x2800).count("1")
            for ch in rendered
            if ch != " " and ord(ch) >= 0x2800
        )

    assert dot_count(loud) > dot_count(quiet)


def test_push_level_clamps_input() -> None:
    panel = StatusPanel()
    panel.push_level(5.0)
    panel.push_level(-3.0)
    assert max(panel._levels) <= 1.0
    assert min(panel._levels) >= 0.0


def test_compact_mode_shrinks_waveform() -> None:
    panel = StatusPanel()
    panel.set_compact(True)
    assert len(_render_lines(panel)) == 3
    panel.set_compact(False)
    assert len(_render_lines(panel)) == WAVEFORM_HEIGHT
