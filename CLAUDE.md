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
- capture backends (selected in `create_recorder()`):
  - macOS: native AVFoundation via an isolated helper process, MANDATORY
    (`mic_helper.swift`, compiled on demand with `swiftc`, cached in
    `~/Library/Caches/tnt/`). Capture runs in a child process so a wedged
    audio stack can always be SIGKILLed and the OS releases the mic.
    There is no silent fallback: if the helper cannot be built, startup
    fails and tells the user to run `xcode-select --install`. Do not
    reintroduce automatic PortAudio fallback on macOS.
  - non-macOS: `sounddevice` + PortAudio in-process. PortAudio is not
    installed, imported, or selectable on macOS — `TNT_CAPTURE_BACKEND=portaudio`
    is rejected there.
- env overrides:
  - `TNT_MLX_MODEL=<path-to-converted-MLX-checkpoint>` (default `bin/qwen3-asr-mlx`
    when it exists, else `~/.local/share/tnt/qwen3-asr-mlx`; bootstrap links both)
  - `TNT_MLX_LANGUAGE=Chinese | English | auto` (default auto; use `Chinese` for mixed zh/en speech — auto may translate Chinese segments to English)
  - `TNT_INPUT_DEVICE=<index-or-name>`
  - `TNT_CAPTURE_BACKEND=auto | avfoundation | portaudio` (default auto)
- unsupported: Linux, Android / Termux / proot. The old CPU backends
  (Moonshine C++, qwen_asr C) were removed on 2026-06-09; do not reintroduce
  them or attempt to debug their history.

## Non-negotiables

- No network calls at runtime.
- No PyTorch, transformers, or CUDA. MLX (Apple GPU) is the only inference path.
- Use `uv` only (`uv sync`, `uv run`, `uv add`).
- Keep runtime dependencies minimal (`textual`, `numpy`, `mlx-speech` + stdlib;
  `sounddevice` on non-macOS only).
- `mlx-speech` is our own package, installed from PyPI (`>=0.4.1`). Its source
  lives at `/Users/ac/dev/ai/genai/mlx-voice`; keep the projects decoupled —
  do not reintroduce a `[tool.uv.sources]` path override outside of temporary
  local debugging.
- Keep blocking work off the UI path (use async/worker patterns).
- The UI thread must NEVER call the recorder directly (start/stop/abort).
  PortAudio can wedge inside C where Python cannot interrupt it; all audio
  calls run on daemon threads with timeouts (1s stop, 3s start), and timed-out
  recorders are flagged stopped, abandoned, and rebuilt. The AVFoundation
  backend keeps the same invariant even though it is pipe/signal based: the
  helper process is the kill boundary that guarantees mic release.
- main() must terminate via os._exit(): sounddevice's atexit hook calls
  Pa_Terminate(), which deadlocks on a wedged stream and leaves a zombie
  python holding the microphone. Do not "clean up" this exit path.
- In-process MLX inference cannot be killed mid-generate: cancel/timeout must
  abandon the result, and generations are serialized behind a lock.

## Source layout

\`\`\`text
src/tnt/
  app.py             # TUI state machine and keybindings
  audio.py           # recorder protocol, backend selection, PortAudio (non-macOS)
  avf_audio.py       # AVFoundation capture via the helper process (macOS)
  mic_helper.swift   # AVFoundation helper source, compiled on demand
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
  - Published checkpoint: `appautomaton/qwen3-asr-1.7b-bf16-mlx` on Hugging Face
  - Or convert upstream weights with mlx-voice's `scripts/convert/qwen3_asr.py`

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
