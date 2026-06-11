import io
import stat
import sys
import threading
import time
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tnt.audio import MicRecorder, create_recorder  # noqa: E402
from tnt.avf_audio import AVFRecorder, ensure_helper  # noqa: E402

# Stands in for the Swift helper: same handshake protocol, synthetic PCM.
FAKE_HELPER_OK = """#!/usr/bin/env python3
import signal, sys, time

if "--list" in sys.argv:
    print("Input device 0: Fake Mic (uid=fake)")
    sys.exit(0)

stop = False
def handle(sig, frame):
    global stop
    stop = True
signal.signal(signal.SIGTERM, handle)

sys.stderr.write("TNT_READY\\n")
sys.stderr.flush()
while not stop:
    sys.stdout.buffer.write(b"\\x00\\x40" * 160)  # loud-ish int16 samples
    sys.stdout.buffer.flush()
    time.sleep(0.01)
sys.exit(0)
"""

FAKE_HELPER_FAIL = """#!/usr/bin/env python3
import sys

if "--list" in sys.argv:
    print("Input device 0: Fake Mic (uid=fake)")
    sys.exit(0)

sys.stderr.write("TNT_ERROR: boom\\n")
sys.stderr.flush()
sys.exit(1)
"""

FAKE_HELPER_HANG = """#!/usr/bin/env python3
import sys, time

if "--list" in sys.argv:
    print("Input device 0: Fake Mic (uid=fake)")
    sys.exit(0)

time.sleep(30)  # never sends the handshake; SIGTERM kills us
"""


def write_helper(tmp_path: Path, source: str) -> Path:
    helper = tmp_path / "fake-mic-helper"
    helper.write_text(source)
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR)
    return helper


def test_avf_recorder_records_and_returns_wav(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TNT_INPUT_DEVICE", raising=False)
    recorder = AVFRecorder(helper_path=write_helper(tmp_path, FAKE_HELPER_OK))

    recorder.start()
    assert recorder.is_recording
    # Poll instead of a fixed sleep: the fake helper's first PCM chunk can
    # arrive late on slow CI runners.
    deadline = time.monotonic() + 5.0
    while recorder.get_level() == 0.0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert recorder.get_level() > 0.0
    assert recorder.elapsed() > 0.0
    wav_bytes = recorder.stop()

    assert not recorder.is_recording
    assert recorder._proc is None
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getnframes() > 0


def test_avf_recorder_start_failure_raises_with_hints(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TNT_INPUT_DEVICE", raising=False)
    recorder = AVFRecorder(helper_path=write_helper(tmp_path, FAKE_HELPER_FAIL))

    with pytest.raises(RuntimeError) as excinfo:
        recorder.start()

    assert "boom" in str(excinfo.value)
    assert "Fake Mic" in str(excinfo.value)
    assert not recorder.is_recording
    assert recorder._proc is None


def test_begin_stop_unblocks_a_hung_start(tmp_path, monkeypatch) -> None:
    """A stop racing a wedged helper must unblock start() and kill the process."""
    monkeypatch.delenv("TNT_INPUT_DEVICE", raising=False)
    recorder = AVFRecorder(helper_path=write_helper(tmp_path, FAKE_HELPER_HANG))

    errors: list[Exception] = []

    def run_start() -> None:
        try:
            recorder.start()
        except Exception as exc:
            errors.append(exc)

    starter = threading.Thread(target=run_start, daemon=True)
    starter.start()
    time.sleep(0.3)
    recorder.begin_stop()
    starter.join(timeout=3)

    assert not starter.is_alive()
    assert not recorder.is_recording
    assert len(errors) == 1


def test_stop_without_start_returns_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TNT_INPUT_DEVICE", raising=False)
    recorder = AVFRecorder(helper_path=write_helper(tmp_path, FAKE_HELPER_OK))
    assert recorder.stop() == b""


def test_create_recorder_rejects_unknown_backend(monkeypatch) -> None:
    monkeypatch.setenv("TNT_CAPTURE_BACKEND", "bogus")
    with pytest.raises(RuntimeError, match="Unknown TNT_CAPTURE_BACKEND"):
        create_recorder()


@pytest.mark.skipif(sys.platform != "darwin", reason="AVFoundation is macOS-only")
def test_portaudio_is_rejected_on_macos(monkeypatch) -> None:
    """PortAudio must be unselectable on macOS, even explicitly."""
    monkeypatch.setenv("TNT_CAPTURE_BACKEND", "portaudio")
    with pytest.raises(RuntimeError, match="not supported on macOS"):
        create_recorder()


def test_non_macos_uses_portaudio_and_rejects_avfoundation(monkeypatch) -> None:
    """Best-effort Linux coverage: simulate a non-darwin platform."""
    from types import SimpleNamespace

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("tnt.audio.sd", SimpleNamespace())
    monkeypatch.delenv("TNT_INPUT_DEVICE", raising=False)
    monkeypatch.delenv("TNT_CAPTURE_BACKEND", raising=False)

    assert isinstance(create_recorder(), MicRecorder)

    monkeypatch.setenv("TNT_CAPTURE_BACKEND", "avfoundation")
    with pytest.raises(RuntimeError, match="requires macOS"):
        create_recorder()


@pytest.mark.skipif(sys.platform != "darwin", reason="AVFoundation is macOS-only")
def test_create_recorder_never_falls_back_silently_on_macos(monkeypatch) -> None:
    """A broken AVFoundation setup must fail loudly, not revive PortAudio."""
    import tnt.avf_audio as avf_module

    def explode(*args, **kwargs):
        raise RuntimeError("swiftc not found; install the Xcode command line tools")

    monkeypatch.delenv("TNT_CAPTURE_BACKEND", raising=False)
    monkeypatch.setattr(avf_module, "AVFRecorder", explode)

    with pytest.raises(RuntimeError, match="swiftc not found"):
        create_recorder()


@pytest.mark.skipif(sys.platform != "darwin", reason="AVFoundation is macOS-only")
def test_real_helper_compiles_and_lists_devices() -> None:
    """The Swift source must compile and enumerate devices on macOS."""
    import subprocess

    binary = ensure_helper()
    result = subprocess.run(
        [str(binary), "--list"], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="AVFoundation is macOS-only")
def test_create_recorder_prefers_avfoundation_on_macos(monkeypatch) -> None:
    monkeypatch.delenv("TNT_CAPTURE_BACKEND", raising=False)
    monkeypatch.delenv("TNT_INPUT_DEVICE", raising=False)
    assert isinstance(create_recorder(), AVFRecorder)
