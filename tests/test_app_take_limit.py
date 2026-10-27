import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tnt.app import TntApp  # noqa: E402


class _LongTake:
    is_recording = True

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def elapsed(self) -> float:
        return self.seconds

    def get_level(self) -> float:
        return 0.0


def _run(mode: str, seconds: float, *, press: bool = True) -> dict[str, object]:
    calls: list[str] = []
    seen: dict[str, object] = {}

    async def scenario() -> None:
        app = TntApp()
        async with app.run_test(size=(100, 28)) as pilot:
            app.recorder = _LongTake(seconds)
            app.state = "recording"
            app._space_recording_mode = mode

            def stop() -> None:
                calls.append("stop")
                app._reset_space_recording_mode()
                app.state = "transcribing"

            app._stop_recording = stop
            app._cancel_transcription = lambda: calls.append("cancel")

            app._update_recording_info()
            seen["swallow"] = app._space_swallow
            if not press:
                return
            app.action_toggle_recording()  # key repeat right after the stop
            await pilot.pause(app._SPACE_HOLD_RELEASE_WINDOW_SECONDS + 0.2)
            seen["swallow_after_release"] = app._space_swallow
            app.action_toggle_recording()  # a fresh press after release

    asyncio.run(scenario())
    seen["calls"] = calls
    return seen


def test_take_limit_stops_and_swallows_a_held_space_until_release() -> None:
    seen = _run("hold", TntApp._MAX_TAKE_SECONDS)
    assert seen["swallow"] is True
    assert seen["swallow_after_release"] is False
    assert seen["calls"] == ["stop", "cancel"]


def test_take_limit_after_a_tap_swallows_nothing() -> None:
    seen = _run("ready", TntApp._MAX_TAKE_SECONDS + 1)
    assert seen["swallow"] is False
    assert seen["calls"] == ["stop", "cancel", "cancel"]


def test_short_take_keeps_recording() -> None:
    seen = _run("ready", TntApp._MAX_TAKE_SECONDS - 1, press=False)
    assert "stop" not in seen["calls"]
