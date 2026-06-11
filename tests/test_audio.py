import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tnt.audio import MicRecorder  # noqa: E402


def test_mic_recorder_stop_aborts_stream_without_stop(monkeypatch) -> None:
    calls: list[str] = []

    class FakeStream:
        def abort(self, ignore_errors=True):  # noqa: ANN001
            calls.append(f"abort:{ignore_errors}")

        def close(self, ignore_errors=True):  # noqa: ANN001
            calls.append(f"close:{ignore_errors}")

        def stop(self, ignore_errors=True):  # noqa: ANN001
            calls.append(f"stop:{ignore_errors}")

    monkeypatch.delenv("TNT_INPUT_DEVICE", raising=False)
    monkeypatch.setattr("tnt.audio.sd", SimpleNamespace())
    recorder = MicRecorder()
    recorder._recording = True
    recorder._stream = FakeStream()
    recorder._chunks = [np.zeros((16, 1), dtype=np.int16)]

    recorder.begin_stop()
    wav_bytes = recorder.stop()

    assert wav_bytes
    assert calls == ["abort:True", "close:True"]


def test_start_aborts_orphaned_stream_when_stop_races_ahead(monkeypatch) -> None:
    """A stop arriving while the stream opens must not leave the mic running."""
    monkeypatch.delenv("TNT_INPUT_DEVICE", raising=False)
    monkeypatch.setattr("tnt.audio.sd", SimpleNamespace())
    recorder = MicRecorder()
    calls: list[str] = []

    class FakeStream:
        def start(self) -> None:
            calls.append("start")
            # Simulate the race: a stop lands while the stream is opening.
            recorder.begin_stop()

        def abort(self, ignore_errors=True):  # noqa: ANN001
            calls.append("abort")

        def close(self, ignore_errors=True):  # noqa: ANN001
            calls.append("close")

    monkeypatch.setattr(
        "tnt.audio.sd", SimpleNamespace(InputStream=lambda **kwargs: FakeStream())
    )

    recorder.start()

    assert recorder._stream is None
    assert recorder.is_recording is False
    assert calls == ["start", "abort", "close"]


def test_audio_callback_ignores_late_frames_after_stop(monkeypatch) -> None:
    monkeypatch.delenv("TNT_INPUT_DEVICE", raising=False)
    monkeypatch.setattr("tnt.audio.sd", SimpleNamespace())
    recorder = MicRecorder()
    recorder._recording = False

    recorder._audio_callback(
        np.ones((8, 1), dtype=np.int16),
        frames=8,
        time_info=None,
        status=None,
    )

    assert recorder._chunks == []
