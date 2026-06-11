# TNT 🧨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/documentation/apple-silicon)

Terminal voice-to-text. Tap <kbd>Space</kbd>, speak, tap <kbd>Space</kbd> — your words land in the transcript and on the clipboard.

Qwen3-ASR-1.7B runs in-process on the Apple GPU via [mlx-speech](https://github.com/appautomaton/mlx-speech): the model loads once, stays resident, and transcribes a short take in about a second. Fully local — no cloud, no runtime network calls. The microphone is captured natively through AVFoundation by a small Swift helper process, so a misbehaving audio stack can never trap the mic: TNT just kills the helper and macOS releases it.

> [!NOTE]
> Using Termux on Android? Use the preserved
> `legacy/android-termux-qwen0.6b` branch instead of `master`.
> It is a legacy proot setup and may need device-specific fixes; validate it
> locally and adapt it with your own tools or agentic AI workflow.
>
> ```bash
> git fetch origin
> git switch --track origin/legacy/android-termux-qwen0.6b
> ```

## Features

- **In-process GPU inference** — pure MLX, no PyTorch
- **Resident model** — loads once in the background at startup; every take is warm
- **Native mic capture** — AVFoundation via an isolated Swift helper process; the mic can always be reclaimed
- **English, Chinese, and mixed speech** — language auto-detected, or forced via env var
- **Live braille oscilloscope** — real audio levels while you record
- **Clipboard-first** — new transcriptions auto-copy; click any past entry to copy it again
- **Responsive TUI** — side-rail layout on wide terminals, stacked on narrow ones

## Setup

> [!IMPORTANT]
> Requires an Apple Silicon Mac (M1 or later), Python 3.13+,
> [uv](https://docs.astral.sh/uv/), and the Xcode command line tools
> (`xcode-select --install`) — the mic capture helper is compiled from Swift
> on first launch and cached.

```bash
git clone https://github.com/appautomaton/tnt-asr.git
cd tnt-asr
uv sync
./bootstrap-mlx-asr.sh /path/to/qwen3-asr-1.7b-bf16-mlx
uv run tnt
```

Or install from PyPI ([`automaton-tnt`](https://pypi.org/project/automaton-tnt/)):

```bash
uv tool install automaton-tnt
TNT_MLX_MODEL=/path/to/qwen3-asr-1.7b-bf16-mlx tnt
```

(Instead of exporting `TNT_MLX_MODEL`, you can symlink the checkpoint at
`~/.local/share/tnt/qwen3-asr-mlx`.)

### Model checkpoint

TNT expects a converted Qwen3-ASR-1.7B MLX checkpoint (BF16). A ready-to-use
one is published at
[appautomaton/qwen3-asr-1.7b-bf16-mlx](https://huggingface.co/appautomaton/qwen3-asr-1.7b-bf16-mlx)
(~4.7 GB) — download it however you prefer, then point the bootstrap script at
it:

```bash
./bootstrap-mlx-asr.sh /path/to/qwen3-asr-1.7b-bf16-mlx
```

This symlinks the checkpoint to `bin/qwen3-asr-mlx` and validates that the
required files are present. Alternatively, convert the upstream
[Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) weights
yourself with mlx-speech's `scripts/convert/qwen3_asr.py`.

## Configuration

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `TNT_MLX_MODEL` | `bin/qwen3-asr-mlx`, else `~/.local/share/tnt/qwen3-asr-mlx` | Path to the converted MLX checkpoint |
| `TNT_MLX_LANGUAGE` | `auto` | `Chinese`, `English`, or `auto`. Use `Chinese` to keep mixed Chinese/English speech from being translated to English |
| `TNT_INPUT_DEVICE` | system default | Microphone, by index or name |
| `TNT_CAPTURE_BACKEND` | `auto` | macOS always uses native AVFoundation (needs the Xcode command line tools: `xcode-select --install`); other platforms use PortAudio. `portaudio` is rejected on macOS |

## Keybindings

| Key | Action |
|-----|--------|
| <kbd>Space</kbd> | Start / stop recording, or hold to record until release; cancels during transcription |
| <kbd>c</kbd> | Copy the last transcript entry |
| mouse click | Copy the clicked transcript entry |
| <kbd>x</kbd> | Clear the transcript |
| <kbd>q</kbd> | Quit |

## Project structure

```text
src/tnt/
├── app.py             # Textual TUI, state machine, keybindings
├── audio.py           # Recorder protocol, backend selection, PortAudio (non-macOS)
├── avf_audio.py       # Native AVFoundation capture via helper process (macOS)
├── mic_helper.swift   # AVFoundation helper source, compiled on demand
├── async_threads.py   # Daemon-thread helpers for blocking work
├── transcriber.py     # In-process MLX Qwen3-ASR transcription
└── widgets/
    ├── transcript.py  # Scrollable transcript log
    └── status.py      # Braille oscilloscope + state rail
bin/
└── qwen3-asr-mlx      # Symlink to converted MLX checkpoint (gitignored)
```

> [!TIP]
> The inference path expects 16 kHz mono PCM WAV; the recorder produces exactly
> that. Cancelling a transcription abandons its result — the in-process
> generation cannot be killed mid-flight and quietly finishes in the background.

## Acknowledgements

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — the underlying speech model by the Qwen team ([converted MLX checkpoint](https://huggingface.co/appautomaton/qwen3-asr-1.7b-bf16-mlx))
- [mlx-speech](https://github.com/appautomaton/mlx-speech) — MLX-native speech runtime for Apple Silicon
- [MLX](https://github.com/ml-explore/mlx) — Apple's array framework for Apple Silicon
- [Textual](https://github.com/Textualize/textual) — the TUI framework

## License

MIT. See [`LICENSE`](LICENSE).
