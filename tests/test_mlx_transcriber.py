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
    MlxQwenTranscriber,
    TranscriptionTimeoutError,
    _wav_bytes_to_float32,
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


def test_recommended_timeout_scales_with_audio_length() -> None:
    assert recommended_timeout(5) == 60.0
    assert recommended_timeout(60) == 255.0


def test_wav_bytes_to_float32_round_trip() -> None:
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    audio, sample_rate = _wav_bytes_to_float32(_make_wav_bytes(samples))
    assert sample_rate == 16000
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, samples / 32768.0, atol=1e-6)


def test_mlx_transcriber_rejects_incomplete_model_dir(tmp_path: Path, monkeypatch) -> None:
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


def test_mlx_transcribe_async_timeout_abandons_result(tmp_path: Path, monkeypatch) -> None:
    transcriber = _make_transcriber(tmp_path, monkeypatch)
    started = threading.Event()

    class SlowModel:
        def generate(self, audio, *, sample_rate, language):  # noqa: ANN001
            started.set()
            time.sleep(1.0)
            return SimpleNamespace(text="late", language="English")

    MlxQwenTranscriber._model_cache[transcriber.model_dir] = SlowModel()
    try:
        wav = _make_wav_bytes(np.zeros(1600, dtype=np.int16))
        with pytest.raises(TranscriptionTimeoutError):
            asyncio.run(transcriber.transcribe_async(wav, timeout=0.05))
    finally:
        MlxQwenTranscriber._model_cache.pop(transcriber.model_dir, None)

    assert started.wait(timeout=2.0)
    assert transcriber._abandoned is True


def test_mlx_abandon_cancels_pending_transcription(tmp_path: Path, monkeypatch) -> None:
    transcriber = _make_transcriber(tmp_path, monkeypatch)
    transcriber.abandon()
    wav = _make_wav_bytes(np.zeros(16, dtype=np.int16))
    with pytest.raises(asyncio.CancelledError):
        transcriber._transcribe_sync(wav, timeout=1)
