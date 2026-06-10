# Agent Guidelines for TNT

## Mission

Keep TNT functional, stable, and usable first. Prioritize runtime reliability and clear user-visible errors over polish work.

## Project summary

TNT is a terminal voice-to-text TUI:
- tap `Space` to start recording
- tap `Space` again to stop and transcribe
- hold `Space` to record until release
- `Space` during transcription cancels it
- ASR: Qwen3-ASR-1.7B (BF16) in-process on the Apple GPU via `mlx-speech`
- the model loads once (background warmup at startup) and stays resident

## Platform

- macOS arm64 (Apple Silicon) only.
- capture backend: `live` (`sounddevice` + PortAudio)
- env overrides:
  - `TNT_MLX_MODEL=<path-to-converted-MLX-checkpoint>` (default `bin/qwen3-asr-mlx`)
  - `TNT_MLX_LANGUAGE=Chinese | English | auto` (default auto; use `Chinese` for mixed zh/en speech — auto may translate Chinese segments to English)
  - `TNT_INPUT_DEVICE=<index-or-name>`
- unsupported: Linux, Android / Termux / proot. The old CPU backends
  (Moonshine C++, qwen_asr C) were removed on 2026-06-09; do not reintroduce
  them or attempt to debug their history.

## Non-negotiables

- No network calls at runtime.
- No PyTorch, transformers, or CUDA. MLX (Apple GPU) is the only inference path.
- Use `uv` only (`uv sync`, `uv run`, `uv add`).
- Keep runtime dependencies minimal (`textual`, `sounddevice`, `numpy`, `mlx-speech` + stdlib).
- `mlx-speech` is our own package, installed from PyPI (`>=0.4.1`). Its source
  lives at `/Users/ac/dev/ai/genai/mlx-voice`; keep the projects decoupled —
  do not reintroduce a `[tool.uv.sources]` path override outside of temporary
  local debugging.
- Keep blocking work off the UI path (use async/worker patterns).
- The UI thread must NEVER call into PortAudio (recorder start/stop/abort).
  PortAudio can wedge inside C where Python cannot interrupt it; all audio
  calls run on daemon threads with timeouts (1s stop, 3s start), and timed-out
  recorders are flagged stopped, abandoned, and rebuilt.
- main() must terminate via os._exit(): sounddevice's atexit hook calls
  Pa_Terminate(), which deadlocks on a wedged stream and leaves a zombie
  python holding the microphone. Do not "clean up" this exit path.
- In-process MLX inference cannot be killed mid-generate: cancel/timeout must
  abandon the result, and generations are serialized behind a lock.

## Source layout

\`\`\`text
src/tnt/
  app.py             # TUI state machine and keybindings
  audio.py           # live microphone capture
  async_threads.py   # daemon-thread helpers for blocking work
  transcriber.py     # in-process MLX Qwen3-ASR transcription
  widgets/
    transcript.py
    status.py
bin/
  qwen3-asr-mlx      # Symlink to converted MLX checkpoint (gitignored)
\`\`\`

## Bootstrap and artifacts

- `./bootstrap-mlx-asr.sh /path/to/Qwen3-ASR-1.7B-MLX-BF16`
  - Symlinks a converted MLX checkpoint to `bin/qwen3-asr-mlx`
  - BF16 is currently the only supported weight format (mlx-speech defers quantization)
  - Convert upstream weights with mlx-voice's `scripts/convert/qwen3_asr.py`

## Audio contract

- Required format for inference: 16 kHz, mono, 16-bit PCM WAV.
- App state flow:
  - `idle -> recording -> stopping -> transcribing -> idle`

## Validation commands

\`\`\`bash
uv sync
uv run ruff check src/ tests/
uv run python -m pytest tests/ -q
uv run tnt
\`\`\`
