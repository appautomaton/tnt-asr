"""Recorder protocol, backend selection, and PortAudio capture (non-macOS)."""

import io
import math
import os
import sys
import threading
import time
import wave
from typing import Any, Protocol

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - env-dependent import failure
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR: Exception | None = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

class Recorder(Protocol):
    """Shared recorder interface used by the app state machine."""

    @property
    def is_recording(self) -> bool:
        """Whether capture is currently active."""

    def start(self) -> None:
        """Begin capture."""

    def stop(self) -> bytes:
        """Stop capture and return WAV bytes."""

    def begin_stop(self) -> None:
        """Abort capture immediately and hand off final cleanup."""

    def elapsed(self) -> float:
        """Elapsed seconds since capture start."""

    def get_level(self) -> float:
        """Level meter value in range 0.0..1.0."""


def encode_wav(audio_data: np.ndarray, sample_rate: int, channels: int) -> bytes:
    """Encode numpy int16 audio data as WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return buf.getvalue()


def int16_level(chunk: np.ndarray) -> float:
    """RMS amplitude of an int16 chunk normalized to 0.0-1.0."""
    samples = chunk.astype(np.float64)
    rms = np.sqrt(np.mean(samples**2))
    if rms <= 1.0:
        return 0.0
    db = 20.0 * math.log10(rms / 32767.0)
    return max(0.0, min(1.0, (db + 60.0) / 60.0))


def resolve_input_device(explicit: int | str | None = None) -> int | str | None:
    """Pick input device from argument or TNT_INPUT_DEVICE env var."""
    if explicit is not None:
        return explicit

    env_value = os.environ.get("TNT_INPUT_DEVICE", "").strip()
    if not env_value:
        return None

    if env_value.isdigit():
        return int(env_value)
    return env_value


def create_recorder(
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: str = "int16",
) -> Recorder:
    """Build the live microphone recorder.

    On macOS the AVFoundation helper-process backend is mandatory: capture
    runs in a child process the app can always kill, so a wedged audio stack
    can never trap the microphone in-process the way PortAudio can. There is
    deliberately no silent fallback — if the helper cannot be built, startup
    fails with instructions rather than reviving the PortAudio failure mode.
    Other platforms use PortAudio/sounddevice. TNT_CAPTURE_BACKEND=portaudio
    forces the old backend for debugging only.
    """
    backend = os.environ.get("TNT_CAPTURE_BACKEND", "auto").strip().lower() or "auto"
    if backend not in ("auto", "avfoundation", "portaudio"):
        raise RuntimeError(
            f"Unknown TNT_CAPTURE_BACKEND={backend!r}; "
            "use 'auto', 'avfoundation', or 'portaudio'."
        )

    if sys.platform == "darwin":
        if backend == "portaudio":
            raise RuntimeError(
                "PortAudio capture is not supported on macOS; "
                "TNT uses native AVFoundation here."
            )
        from tnt.avf_audio import AVFRecorder

        return AVFRecorder(sample_rate=sample_rate, channels=channels)

    if backend == "avfoundation":
        raise RuntimeError("TNT_CAPTURE_BACKEND=avfoundation requires macOS.")

    return MicRecorder(sample_rate=sample_rate, channels=channels, dtype=dtype)


class MicRecorder:
    """Records audio from the default microphone and returns WAV bytes."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
        device: int | str | None = None,
    ) -> None:
        if sd is None:
            detail = str(_SOUNDDEVICE_IMPORT_ERROR) if _SOUNDDEVICE_IMPORT_ERROR else ""
            raise RuntimeError(
                "Live microphone backend unavailable: sounddevice failed to load.\n"
                f"Reason: {detail}\n"
                "Install PortAudio and retry."
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device = self._resolve_device(device)

        self._stream: Any | None = None
        self._closing_stream: Any | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._recording = False
        self._start_time: float = 0.0
        self._current_level: float = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        """Open an InputStream and begin capturing audio."""
        if self._recording:
            return

        with self._lock:
            self._chunks = []
            self._current_level = 0.0

        self._start_time = time.monotonic()
        self._recording = True

        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.device,
                callback=self._audio_callback,
            )
            stream.start()
        except Exception as exc:
            self._recording = False
            self._stream = None
            raise RuntimeError(self._build_mic_error(str(exc))) from exc

        self._stream = stream
        # start() runs on a worker thread; if a stop raced ahead of us while
        # the stream was opening, shut the orphaned stream down immediately.
        if not self._recording:
            self._stream = None
            self._abort_stream(stream)
            self._close_stream(stream)

    def stop(self) -> bytes:
        """Stop recording, close the stream, return WAV bytes."""
        if self._recording:
            self.begin_stop()

        stream = self._closing_stream
        self._closing_stream = None
        if stream is not None:
            self._close_stream(stream)

        with self._lock:
            if not self._chunks:
                return b""
            audio_data = np.concatenate(self._chunks)
            self._chunks = []

        return self._encode_wav(audio_data)

    def begin_stop(self) -> None:
        """Abort capture immediately so the mic turns off promptly."""
        if not self._recording:
            return

        self._recording = False

        stream = self._stream
        self._stream = None
        if stream is not None:
            self._abort_stream(stream)
            self._closing_stream = stream

    def _abort_stream(self, stream: Any) -> None:
        """Ask PortAudio to stop immediately."""
        try:
            stream.abort(ignore_errors=True)
        except TypeError:
            stream.abort()
        except Exception:
            pass

    def _close_stream(self, stream: Any) -> None:
        """Close a previously aborted stream."""
        try:
            stream.close(ignore_errors=True)
        except TypeError:
            stream.close()
        except Exception:
            pass

    def elapsed(self) -> float:
        """Seconds since start() was called."""
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    def get_level(self) -> float:
        """Current RMS amplitude normalized to 0.0-1.0."""
        with self._lock:
            return self._current_level

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Called on the audio thread for each chunk."""
        del frames, time_info, status
        if not self._recording:
            return
        chunk = indata.copy()
        normalized = int16_level(chunk)

        with self._lock:
            self._chunks.append(chunk)
            self._current_level = normalized

    def _encode_wav(self, audio_data: np.ndarray) -> bytes:
        """Encode numpy int16 audio data as WAV bytes."""
        return encode_wav(audio_data, self.sample_rate, self.channels)

    def _resolve_device(self, explicit: int | str | None) -> int | str | None:
        """Pick input device from argument or TNT_INPUT_DEVICE env var."""
        return resolve_input_device(explicit)

    def _build_mic_error(self, base_error: str) -> str:
        """Create a user-facing mic error with actionable setup hints."""
        lines = [base_error]
        lines.append("Set TNT_INPUT_DEVICE to an input index or device name if needed.")
        lines.extend(self._list_input_hints(limit=5))
        return "\n".join(lines)

    def _list_input_hints(self, limit: int = 5) -> list[str]:
        """Return a short list of discoverable input devices."""
        if sd is None:
            return ["No audio device list available from PortAudio."]

        try:
            devices = sd.query_devices()
        except Exception:
            return ["No audio device list available from PortAudio."]

        hints: list[str] = []
        count = 0
        for idx, dev in enumerate(devices):
            max_in = int(dev.get("max_input_channels", 0))
            if max_in <= 0:
                continue
            name = str(dev.get("name", f"device-{idx}"))
            hints.append(f"Input device {idx}: {name} (max_in={max_in})")
            count += 1
            if count >= limit:
                break

        if not hints:
            return ["No input-capable devices reported by PortAudio."]
        return hints
