import asyncio
import io
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tnt.transcriber import (  # noqa: E402
    LiveUtterance,
    MlxQwenTranscriber,
    TranscriptionTimeoutError,
    _visible_transcript,
    _wav_bytes_to_float32,
    pump_live,
    recommended_timeout,
)


def _make_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


def _make_transcriber(tmp_path: Path, monkeypatch) -> MlxQwenTranscriber:
    model_dir = tmp_path / "qwen3-asr-mlx"
    model_dir.mkdir()
    for name in MlxQwenTranscriber.REQUIRED_MODEL_FILES:
        (model_dir / name).write_text("")
    monkeypatch.setattr("tnt.transcriber.sys.platform", "darwin")
    monkeypatch.setattr("tnt.transcriber.platform.machine", lambda: "arm64")
    monkeypatch.setattr(
        "tnt.transcriber.importlib.util.find_spec", lambda name: object()
    )
    monkeypatch.delenv("TNT_MLX_LANGUAGE", raising=False)
    return MlxQwenTranscriber(model_dir=str(model_dir))


def test_visible_transcript_drops_glued_language_prefix() -> None:
    assert _visible_transcript("language EnglishSearch for the e") == "Search for the e"
    assert _visible_transcript("  language Chinese你好") == "你好"
    assert _visible_transcript("Search for the e") == "Search for the e"


def test_recommended_timeout_scales_with_audio_length() -> None:
    assert recommended_timeout(5) == 60.0
    assert recommended_timeout(60) == 255.0


def test_wav_bytes_to_float32_round_trip() -> None:
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    audio, sample_rate = _wav_bytes_to_float32(_make_wav_bytes(samples))
    assert sample_rate == 16000
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, samples / 32768.0, atol=1e-6)


def test_mlx_transcriber_rejects_incomplete_model_dir(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("tnt.transcriber.sys.platform", "darwin")
    monkeypatch.setattr("tnt.transcriber.platform.machine", lambda: "arm64")
    monkeypatch.setattr(
        "tnt.transcriber.importlib.util.find_spec", lambda name: object()
    )
    model_dir = tmp_path / "incomplete"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        MlxQwenTranscriber(model_dir=str(model_dir))


def test_mlx_language_env_override(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "qwen3-asr-mlx-lang"
    model_dir.mkdir()
    for name in MlxQwenTranscriber.REQUIRED_MODEL_FILES:
        (model_dir / name).write_text("")
    monkeypatch.setattr("tnt.transcriber.sys.platform", "darwin")
    monkeypatch.setattr("tnt.transcriber.platform.machine", lambda: "arm64")
    monkeypatch.setattr(
        "tnt.transcriber.importlib.util.find_spec", lambda name: object()
    )

    monkeypatch.setenv("TNT_MLX_LANGUAGE", "Chinese")
    transcriber = MlxQwenTranscriber(model_dir=str(model_dir))
    assert transcriber.language == "Chinese"

    monkeypatch.setenv("TNT_MLX_LANGUAGE", "auto")
    transcriber = MlxQwenTranscriber(model_dir=str(model_dir))
    assert transcriber.language is None


def test_mlx_transcribe_async_returns_text(tmp_path: Path, monkeypatch) -> None:
    transcriber = _make_transcriber(tmp_path, monkeypatch)

    class FakeModel:
        def generate(self, audio, *, sample_rate, language):  # noqa: ANN001
            assert sample_rate == 16000
            assert language is None
            return SimpleNamespace(text=" hello world ", language="English")

    MlxQwenTranscriber._model_cache[transcriber.model_dir] = FakeModel()
    try:
        wav = _make_wav_bytes(np.zeros(1600, dtype=np.int16))
        text = asyncio.run(transcriber.transcribe_async(wav, timeout=5))
    finally:
        MlxQwenTranscriber._model_cache.pop(transcriber.model_dir, None)

    assert text == "hello world"


def test_mlx_transcribe_async_timeout_abandons_result(
    tmp_path: Path, monkeypatch
) -> None:
    transcriber = _make_transcriber(tmp_path, monkeypatch)
    started = threading.Event()
    drained = threading.Event()
    # clear_cache runs in _transcribe_sync's finally; use it to know the
    # abandoned worker has fully drained so it cannot outlive this test.
    monkeypatch.setattr("mlx.core.clear_cache", drained.set)

    class SlowModel:
        def generate(self, audio, *, sample_rate, language):  # noqa: ANN001
            started.set()
            time.sleep(0.3)
            return SimpleNamespace(text="late", language="English")

    MlxQwenTranscriber._model_cache[transcriber.model_dir] = SlowModel()
    try:
        wav = _make_wav_bytes(np.zeros(1600, dtype=np.int16))
        with pytest.raises(TranscriptionTimeoutError):
            asyncio.run(transcriber.transcribe_async(wav, timeout=0.05))
        assert started.wait(timeout=2.0)
        # Drain the abandoned generation (it is not killable) before returning,
        # so its global clear_cache() cannot land inside a later test.
        assert drained.wait(timeout=2.0)
    finally:
        MlxQwenTranscriber._model_cache.pop(transcriber.model_dir, None)

    assert transcriber._abandoned is True


def test_mlx_abandon_cancels_pending_transcription(tmp_path: Path, monkeypatch) -> None:
    transcriber = _make_transcriber(tmp_path, monkeypatch)
    transcriber.abandon()
    wav = _make_wav_bytes(np.zeros(16, dtype=np.int16))
    with pytest.raises(asyncio.CancelledError):
        transcriber._transcribe_sync(wav, timeout=1)


def test_pump_live_feeds_new_samples_then_finalizes() -> None:
    chunks = [
        np.array([1, 2], dtype=np.int16),
        np.array([3, 4], dtype=np.int16),
    ]

    class Recorder:
        def __init__(self) -> None:
            self._seen = 0

        def copy_new_pcm(self, offset: int):
            if self._seen >= len(chunks):
                return np.zeros((0,), dtype=np.int16), offset
            pcm = chunks[self._seen]
            self._seen += 1
            return pcm, offset + 1

    class Session:
        def __init__(self) -> None:
            self.fed: list[int] = []

        def feed(self, audio):  # noqa: ANN001
            self.fed.append(int(audio.shape[0]))
            return SimpleNamespace(committed="你" * len(self.fed))

        def finalize(self):
            return SimpleNamespace(text="你你。", language="Chinese")

    class Model:
        def __init__(self) -> None:
            self.session = Session()

        def stream_session(self, *, language, chunk_ms, lookahead_ms):  # noqa: ANN001
            assert language == "Chinese"
            assert chunk_ms == 160
            return self.session

    model = Model()
    utterance = LiveUtterance(model, None)
    stop = threading.Event()

    def stop_soon() -> None:
        time.sleep(0.12)
        stop.set()

    threading.Thread(target=stop_soon, daemon=True).start()
    text = pump_live(Recorder(), utterance, stop)
    assert model.session.fed == [2, 2]
    assert text == "你你。"


def test_mlx_transcribe_uses_stream_session_when_present(
    tmp_path: Path, monkeypatch
) -> None:
    transcriber = _make_transcriber(tmp_path, monkeypatch)
    transcriber.language = "Chinese"
    events: list[str] = []
    monkeypatch.setattr("mlx.core.clear_cache", lambda: events.append("clear"))

    class FakeSession:
        def feed(self, audio):  # noqa: ANN001
            events.append(f"feed:{audio.shape[0]}")
            return SimpleNamespace(committed="你好")

        def finalize(self):
            events.append("finalize")
            return SimpleNamespace(text="你好。", language="Chinese")

    class StreamingModel:
        def stream_session(self, *, language, chunk_ms, lookahead_ms):  # noqa: ANN001
            assert language == "Chinese"
            assert chunk_ms == 160
            assert lookahead_ms == 160
            events.append("session")
            return FakeSession()

        def generate(self, *args, **kwargs):  # noqa: ANN001
            raise AssertionError("streaming models must not use generate()")

    MlxQwenTranscriber._model_cache[transcriber.model_dir] = StreamingModel()
    try:
        wav = _make_wav_bytes(np.zeros(3200, dtype=np.int16))
        text = transcriber._transcribe_sync(wav, timeout=5)
    finally:
        MlxQwenTranscriber._model_cache.pop(transcriber.model_dir, None)

    assert text == "你好。"
    assert events == ["session", "feed:3200", "finalize", "clear"]


def test_mlx_transcribe_clears_buffer_cache_after_generate(
    tmp_path: Path, monkeypatch
) -> None:
    """Each generation must release MLX's Metal buffer cache, after generate().

    Without this the cache pools freed GPU buffers up to ~device memory and RSS
    ratchets into the tens of GB over a long session.
    """
    transcriber = _make_transcriber(tmp_path, monkeypatch)
    events: list[str] = []
    monkeypatch.setattr("mlx.core.clear_cache", lambda: events.append("clear"))

    class FakeModel:
        def generate(self, audio, *, sample_rate, language):  # noqa: ANN001
            events.append("generate")
            return SimpleNamespace(text=" hi ", language="English")

    MlxQwenTranscriber._model_cache[transcriber.model_dir] = FakeModel()
    try:
        wav = _make_wav_bytes(np.zeros(1600, dtype=np.int16))
        text = transcriber._transcribe_sync(wav, timeout=5)
    finally:
        MlxQwenTranscriber._model_cache.pop(transcriber.model_dir, None)

    assert text == "hi"
    assert events == ["generate", "clear"]


def test_mlx_transcribe_clears_buffer_cache_when_generate_raises(
    tmp_path: Path, monkeypatch
) -> None:
    """The cache must still be released when generate() fails (e.g. context overflow)."""
    transcriber = _make_transcriber(tmp_path, monkeypatch)
    events: list[str] = []
    monkeypatch.setattr("mlx.core.clear_cache", lambda: events.append("clear"))

    class ExplodingModel:
        def generate(self, audio, *, sample_rate, language):  # noqa: ANN001
            events.append("generate")
            raise RuntimeError("context overflow")

    MlxQwenTranscriber._model_cache[transcriber.model_dir] = ExplodingModel()
    try:
        wav = _make_wav_bytes(np.zeros(1600, dtype=np.int16))
        with pytest.raises(RuntimeError, match="context overflow"):
            transcriber._transcribe_sync(wav, timeout=5)
    finally:
        MlxQwenTranscriber._model_cache.pop(transcriber.model_dir, None)

    assert events == ["generate", "clear"]
