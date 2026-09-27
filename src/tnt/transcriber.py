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

MODEL_LABEL = "qwen3-asr-1.7b-int8-mlx"
R2T2_LABEL = "confucius4-r2t2-bf16"
DEFAULT_TIMEOUT_SECONDS = 60.0


def model_choices() -> list[tuple[str, Path]]:
    """Installed models the user can switch between: ``(label, dir)``.

    Qwen3-ASR (one-shot) comes first and is the default; R2T2 (live
    streaming) is offered when its checkpoint link exists, found the same
    way as Qwen3-ASR: ``TNT_R2T2_MODEL``, else ``bin/r2t2-mlx``, else
    ``~/.local/share/tnt/r2t2-mlx``.
    """

    choices = [(MODEL_LABEL, MlxQwenTranscriber._default_model_dir())]
    raw = os.environ.get("TNT_R2T2_MODEL", "").strip()
    if raw:
        r2t2 = Path(raw).expanduser()
    elif Path("bin/r2t2-mlx").exists():
        r2t2 = Path("bin/r2t2-mlx")
    else:
        r2t2 = Path.home() / ".local" / "share" / "tnt" / "r2t2-mlx"
    if r2t2.exists():
        choices.append((R2T2_LABEL, r2t2.resolve()))
    return choices


def label_for_model_dir(model_dir: Path) -> str:
    resolved = Path(model_dir).resolve()
    for label, path in model_choices():
        if path.resolve() == resolved:
            return label
    return resolved.name
_STREAM_CHUNK_MS = 160
_STREAM_LOOKAHEAD_MS = 160


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
        raw_dir = model_dir or os.environ.get("TNT_MLX_MODEL", "").strip()
        self.model_dir = (
            Path(raw_dir).resolve() if raw_dir else self._default_model_dir()
        )
        self.language = os.environ.get("TNT_MLX_LANGUAGE", "").strip() or None
        if self.language and self.language.lower() == "auto":
            self.language = None
        self._abandoned = False
        self._validate()

    @property
    def model_label(self) -> str:
        return label_for_model_dir(self.model_dir)

    @staticmethod
    def _default_model_dir() -> Path:
        """Repo-local checkpoint link when present, else the user-level one.

        The cwd-relative bin/ link only exists when running from a checkout;
        pip-installed runs from anywhere fall through to the per-user path
        that bootstrap-mlx-asr.sh also links.
        """
        repo_local = Path("bin/qwen3-asr-mlx")
        if repo_local.exists():
            return repo_local.resolve()
        return Path.home() / ".local" / "share" / "tnt" / "qwen3-asr-mlx"

    def _validate(self) -> None:
        if sys.platform != "darwin" or platform.machine().lower() != "arm64":
            raise RuntimeError("TNT requires an Apple Silicon Mac for MLX inference.")
        if importlib.util.find_spec("mlx_speech") is None:
            raise RuntimeError("mlx-speech is not installed. Run: uv sync")
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"MLX model not found at {self.model_dir}\n"
                "Fetch it with: ./bootstrap-mlx-asr.sh  (downloads the int8 "
                "Qwen3-ASR checkpoint from Hugging Face and links it). If you "
                "already have a checkpoint, pass its path to that script or set "
                "TNT_MLX_MODEL to it."
            )
        missing = [
            name
            for name in self.REQUIRED_MODEL_FILES
            if not (self.model_dir / name).exists()
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

    def activate(self) -> None:
        """Switch to this model: drop every other loaded model, then load
        this one. Runs on a daemon thread; the lock may be held by an
        abandoned generation, and the UI must never wait on it."""

        def _swap() -> None:
            with self._model_lock:
                for path in list(self._model_cache):
                    if path != self.model_dir:
                        del self._model_cache[path]
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except Exception:
                    pass
            try:
                self._load_model_locked()
            except Exception:
                pass  # surfaced on first real transcription

        threading.Thread(target=_swap, name="tnt-mlx-switch", daemon=True).start()

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
        import mlx.core as mx  # local: keep transcriber importable without MLX

        audio, sample_rate = _wav_bytes_to_float32(wav_bytes)
        model = self._load_model_locked()
        # Serialize generations: an abandoned (cancelled/timed-out) generate
        # keeps the GPU busy until it finishes; the next one must wait.
        with self._model_lock:
            if self._abandoned:
                raise asyncio.CancelledError()
            try:
                if hasattr(model, "stream_session"):
                    text = _stream_utterance(
                        model,
                        audio,
                        sample_rate=sample_rate,
                        language=self.language,
                    )
                else:
                    result = model.generate(
                        audio, sample_rate=sample_rate, language=self.language
                    )
                    text = result.text
            finally:
                # Release MLX's Metal buffer cache. MLX pools freed GPU buffers
                # (per size class) up to ~device memory and never returns them to
                # the OS on its own; each transcription's scratch — chiefly the
                # prefill logits [1, prompt_len, vocab] and the per-call KV cache,
                # both scaling with recording length — leaves ever-larger buffers
                # cached. Over a long session of varied/long recordings that
                # ratchets RSS into the tens of GB. The resident weights are live
                # ("active") memory, not cache, so this frees only reusable scratch
                # and leaves the model loaded. Runs inside the lock (no concurrent
                # generate) and in finally so a context-overflow error still frees.
                mx.clear_cache()
        if self._abandoned:
            raise asyncio.CancelledError()
        return _visible_transcript(text)

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


class LiveUtterance:
    """Feed mic PCM into one R2T2 session while the user is still talking.

    ``text`` is the append-only committed string. ``finish`` releases the
    held-back token. A model step can take a moment; it runs on the caller
    thread, never the UI thread.
    """

    def __init__(self, model: object, language: str | None) -> None:
        self.text = ""
        self._session = model.stream_session(  # type: ignore[attr-defined]
            language=language or "Chinese",
            chunk_ms=_STREAM_CHUNK_MS,
            lookahead_ms=_STREAM_LOOKAHEAD_MS,
        )

    def push_int16(self, pcm: np.ndarray) -> str:
        if pcm.size == 0:
            return self.text
        audio = np.asarray(pcm, dtype=np.float32).reshape(-1) / 32768.0
        update = self._session.feed(audio)
        self.text = _visible_transcript(update.committed)
        return self.text

    def finish(self) -> str:
        import mlx.core as mx

        try:
            self.text = _visible_transcript(self._session.finalize().text)
        finally:
            mx.clear_cache()
        return self.text.strip()


def pump_live(recorder, utterance: LiveUtterance, stop: threading.Event) -> str:
    """Pull new mic samples until ``stop``, then finalize.

    The last pull happens after ``stop`` is set, while the recorder still
    holds its buffer. The caller stops the device only after this returns.
    """

    offset = 0
    while not stop.is_set():
        pcm, offset = recorder.copy_new_pcm(offset)
        if pcm.size:
            utterance.push_int16(pcm)
        else:
            stop.wait(0.05)
    pcm, offset = recorder.copy_new_pcm(offset)
    if pcm.size:
        utterance.push_int16(pcm)
    return utterance.finish()


_LANGUAGE_NAMES = (
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
)


def _visible_transcript(text: str) -> str:
    """Drop a leaked ``language English`` prefix glued onto the words."""

    cleaned = text.strip()
    if not cleaned.lower().startswith("language"):
        return cleaned
    rest = cleaned[len("language") :].lstrip()
    for name in sorted(_LANGUAGE_NAMES, key=len, reverse=True):
        if rest.lower().startswith(name.lower()):
            return rest[len(name) :].lstrip()
    return cleaned


def _stream_utterance(
    model, audio: np.ndarray, *, sample_rate: int, language: str | None
) -> str:
    """Run one recorded take through the R2T2 chunk loop.

    The state machine splits the buffer. A model without ``stream_session``
    never reaches this function.
    """

    if sample_rate != 16000:
        result = model.generate(audio, sample_rate=sample_rate, language=language)
        return result.text
    session = model.stream_session(
        language=language,
        chunk_ms=_STREAM_CHUNK_MS,
        lookahead_ms=_STREAM_LOOKAHEAD_MS,
    )
    session.feed(np.asarray(audio, dtype=np.float32))
    return session.finalize().text


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
