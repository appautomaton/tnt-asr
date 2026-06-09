import sys
from pathlib import Path

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
    recorder = MicRecorder()
    recorder._recording = True
    recorder._stream = FakeStream()
    recorder._chunks = [np.zeros((16, 1), dtype=np.int16)]

    recorder.begin_stop()
    wav_bytes = recorder.stop()

    assert wav_bytes
    assert calls == ["abort:True", "close:True"]


def test_audio_callback_ignores_late_frames_after_stop(monkeypatch) -> None:
    monkeypatch.delenv("TNT_INPUT_DEVICE", raising=False)
    recorder = MicRecorder()
    recorder._recording = False

    recorder._audio_callback(
        np.ones((8, 1), dtype=np.int16),
        frames=8,
        time_info=None,
        status=None,
    )

    assert recorder._chunks == []
