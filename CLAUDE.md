# TNT

Terminal voice-to-text for Apple Silicon. Keep it working first: runtime
reliability and clear user-visible errors come before polish.

## How it works

- Tap `Space` to start and stop a take, or hold it to record until release;
  `Space` during transcription cancels. `m` switches models. A take stops on
  its own at `_MAX_TAKE_SECONDS` (3 minutes); a still-held `Space` is ignored
  until release so it cannot cancel that transcription.
- ASR runs in-process on the Apple GPU through `mlx-speech`. Confucius4-R2T2
  (default) transcribes live; Qwen3-ASR-1.7B transcribes after stop. The model
  loads once and stays resident.
- States: `idle -> recording -> stopping -> transcribing -> idle`. Inference
  input is 16 kHz mono 16-bit PCM.

## Non-negotiables

- macOS arm64 only. Linux, Android, and the removed CPU backends are out of
  scope; do not revive them.
- No network at runtime. No PyTorch, transformers, or CUDA: MLX is the only
  inference path.
- `uv` only. Runtime dependencies: `textual`, `numpy`, `mlx-speech` (from PyPI,
  `>=0.5.5`, never a local path override), and the stdlib; `sounddevice` off
  macOS only.
- macOS capture runs in the AVFoundation helper process (`mic_helper.swift`),
  the kill boundary that guarantees mic release. If it cannot be built, startup
  fails with instructions; there is no PortAudio fallback on macOS.
- The UI thread never calls the recorder. Audio calls run on daemon threads with
  timeouts (1 s stop, 3 s start); a timed-out recorder is abandoned and rebuilt.
- `main()` exits through `os._exit()`: sounddevice's atexit `Pa_Terminate()`
  can deadlock and keep the mic held.
- MLX inference cannot be killed mid-generate. Cancel and timeout abandon the
  result, and generations serialize behind a lock.

## Where things live

- `src/tnt/transcriber.py`: model discovery and order, the R2T2 window
  (`_STREAM_*`), `DEFAULT_CONTEXT` and its `~/.config/tnt/context.txt`
  override, language handling.
- `src/tnt/app.py`: TUI state machine and keys. `audio.py`, `avf_audio.py`:
  recorder backends. `async_threads.py`: daemon-thread helpers.
- `bootstrap-mlx-asr.sh [r2t2|qwen3] [repo-id | path]`: downloads or links
  checkpoints into `bin/` and `~/.local/share/tnt/`.
- User-facing settings (environment variables, context file) are documented in
  `README.md`.

## Validate

```bash
uv run ruff check src/ tests/
uv run python -m pytest tests/ -q
uv run tnt   # live check with a real microphone
```
