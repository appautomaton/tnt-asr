"""Memory-leak probe for the resident MLX Qwen3-ASR path.

Runs many transcriptions through the exact TNT code path (MlxQwenTranscriber)
on synthetic audio of varying lengths, and after each iteration records:

  rss        - process resident set size (what the OS / Activity Monitor shows)
  active     - mx.get_active_memory()  (live MLX arrays; a true leak grows this)
  cache      - mx.get_cache_memory()   (freed buffers MLX keeps for reuse)
  peak       - mx.get_peak_memory()

Two phases:
  A. baseline  - current TNT behaviour (no cache management)
  B. clear     - call mx.clear_cache() after each transcription

If `active` is flat while `cache`/`rss` climb in phase A and B flattens rss,
the leak is the unbounded MLX buffer cache, fixable from TNT.
"""

from __future__ import annotations

import gc
import io
import os
import resource
import sys
import wave

import numpy as np

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tnt.transcriber import MlxQwenTranscriber  # noqa: E402


def rss_gb() -> float:
    # ru_maxrss is bytes on macOS.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024 / 1024


def synth_wav(seconds: float, sr: int = 16000, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    tone = 0.2 * np.sin(2 * np.pi * 180.0 * t) + 0.1 * np.sin(2 * np.pi * 320.0 * t)
    sig = tone + 0.02 * rng.standard_normal(n)
    pcm = np.clip(sig * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def line(tag: str, i: int) -> None:
    print(
        f"{tag} {i:3d} | rss {rss_gb():6.2f}G | "
        f"active {mx.get_active_memory()/1e9:6.2f}G | "
        f"cache {mx.get_cache_memory()/1e9:6.2f}G | "
        f"peak {mx.get_peak_memory()/1e9:6.2f}G",
        flush=True,
    )


def main() -> None:
    iters = int(os.environ.get("PROBE_ITERS", "24"))
    # Vary durations so each call has a different working-set / KV-cache size,
    # which is what makes a high-RAM machine's buffer cache ratchet upward.
    durations = [2.0, 5.0, 9.0, 14.0, 3.0, 7.0]

    tr = MlxQwenTranscriber()
    tr._load_model_locked()  # warm load, like the app's warmup()
    mx.reset_peak_memory()
    print(f"# loaded model | rss {rss_gb():.2f}G | active {mx.get_active_memory()/1e9:.2f}G")

    print("\n## PHASE A: baseline (no cache management, current TNT behaviour)")
    for i in range(iters):
        wav = synth_wav(durations[i % len(durations)], seed=i)
        tr._transcribe_sync(wav, timeout=120.0)
        if i % 2 == 0 or i == iters - 1:
            line("A", i)

    print("\n## PHASE B: mx.clear_cache() after each transcription")
    mx.reset_peak_memory()
    for i in range(iters):
        wav = synth_wav(durations[i % len(durations)], seed=100 + i)
        tr._transcribe_sync(wav, timeout=120.0)
        mx.clear_cache()
        gc.collect()
        if i % 2 == 0 or i == iters - 1:
            line("B", i)


if __name__ == "__main__":
    main()
