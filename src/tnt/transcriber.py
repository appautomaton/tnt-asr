"""In-process Qwen3-ASR transcription on the Apple GPU via mlx-speech."""

import asyncio
import importlib.util
import io
import os
import platform
import sys
import threading
import wave
from pathlib import Path

import numpy as np

from tnt.async_threads import start_daemon_thread

MODEL_LABEL = "qwen3-asr-1.7b-mlx"
DEFAULT_TIMEOUT_SECONDS = 60.0


class TranscriptionTimeoutError(asyncio.TimeoutError):
    """Raised when inference exceeds its timeout."""


def recommended_timeout(audio_seconds: float) -> float:
    """Return a conservative timeout for local GPU inference."""
    return max(DEFAULT_TIMEOUT_SECONDS, 15.0 + max(audio_seconds, 0.0) * 4.0)


class MlxQwenTranscriber:
    """In-process Qwen3-ASR inference on the GPU via mlx-speech.

    The model stays resident after the first load, so there is no subprocess
    lifecycle to manage. The trade-off: an in-flight generate() cannot be
    killed, so cancel/timeout abandon the result and a class-level lock keeps
    a stale generation from overlapping the next one.
    """

    REQUIRED_MODEL_FILES = (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "vocab.json",
        "merges.txt",
    )

    _model_lock = threading.Lock()
    _model_cache: dict[Path, object] = {}

    def __init__(self, model_dir: str | None = None) -> None:
        raw_dir = (
            model_dir
            or os.environ.get("TNT_MLX_MODEL", "").strip()
            or "bin/qwen3-asr-mlx"
        )
        self.model_dir = Path(raw_dir).resolve()
        self.language = os.environ.get("TNT_MLX_LANGUAGE", "").strip() or None
        if self.language and self.language.lower() == "auto":
            self.language = None
        self._abandoned = False
        self._validate()

    @property
    def model_label(self) -> str:
        return MODEL_LABEL

    def _validate(self) -> None:
        if sys.platform != "darwin" or platform.machine().lower() != "arm64":
            raise RuntimeError("TNT requires an Apple Silicon Mac for MLX inference.")
        if importlib.util.find_spec("mlx_speech") is None:
            raise RuntimeError("mlx-speech is not installed. Run: uv sync")
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"MLX model directory not found at {self.model_dir}\n"
                "Run ./bootstrap-mlx-asr.sh <path-to-Qwen3-ASR-1.7B-MLX-BF16>."
            )
        missing = [
            name for name in self.REQUIRED_MODEL_FILES if not (self.model_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"MLX model is incomplete at {self.model_dir}\n"
                f"Missing files: {', '.join(missing)}"
            )

    def warmup(self) -> None:
        """Load the model on a background thread so the first take is warm."""

        def _load() -> None:
            try:
                self._load_model_locked()
            except Exception:
                pass  # surfaced on first real transcription

        threading.Thread(target=_load, name="tnt-mlx-warmup", daemon=True).start()

    def _load_model_locked(self) -> object:
        with self._model_lock:
            model = self._model_cache.get(self.model_dir)
            if model is None:
                import mlx_speech

                model = mlx_speech.asr.load(str(self.model_dir))
                self._model_cache[self.model_dir] = model
            return model

    def _transcribe_sync(self, wav_bytes: bytes, timeout: float) -> str:
        del timeout  # enforced by the async wrapper; generate() is not killable
        if self._abandoned:
            raise asyncio.CancelledError()
        audio, sample_rate = _wav_bytes_to_float32(wav_bytes)
        model = self._load_model_locked()
        # Serialize generations: an abandoned (cancelled/timed-out) generate
        # keeps the GPU busy until it finishes; the next one must wait.
        with self._model_lock:
            if self._abandoned:
                raise asyncio.CancelledError()
            result = model.generate(
                audio, sample_rate=sample_rate, language=self.language
            )
        if self._abandoned:
            raise asyncio.CancelledError()
        return result.text.strip()

    async def transcribe_async(self, wav_bytes: bytes, timeout: float = 120) -> str:
        """Run transcription in a worker thread with cancellation support."""
        self._abandoned = False
        fut = start_daemon_thread(
            self._transcribe_sync,
            wav_bytes,
            timeout,
            name="tnt-mlx-transcribe",
        )
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.CancelledError:
            self._abandoned = True
            raise
        except asyncio.TimeoutError as exc:
            self._abandoned = True
            raise TranscriptionTimeoutError(
                f"mlx inference exceeded {timeout:.0f}s; result abandoned."
            ) from exc

    def abandon(self) -> None:
        """Mark any in-flight generation as abandoned; its result is dropped."""
        self._abandoned = True


def _wav_bytes_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode 16-bit PCM WAV bytes to a mono float32 waveform in [-1, 1]."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width}.")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate
