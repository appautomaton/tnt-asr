# TNT 🧨

[![Website](https://img.shields.io/badge/website-appautomaton.com-ff4fd8?logo=github&logoColor=white)](https://appautomaton.com/tnt-asr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-automaton--tnt-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/automaton-tnt/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/documentation/apple-silicon)

Terminal voice-to-text for Apple Silicon. Tap <kbd>Space</kbd>, speak, tap <kbd>Space</kbd>. The text lands in the transcript and on the clipboard.

Models run locally on the Apple GPU through [mlx-speech](https://github.com/appautomaton/mlx-speech). No cloud, no PyTorch.

- **Confucius4-R2T2** (default): transcribes live while you speak.
- **Qwen3-ASR-1.7B** (optional): transcribes each take after you stop.

Press <kbd>m</kbd> to switch models. Only the active model stays in memory.

> [!NOTE]
> Using Termux on Android? Use the legacy `legacy/android-termux-qwen0.6b` branch.

## Setup

Requires an Apple Silicon Mac, Python 3.13+, [uv](https://docs.astral.sh/uv/), and the Xcode command line tools (`xcode-select --install`) for the Swift mic helper.

```bash
git clone https://github.com/appautomaton/tnt-asr.git
cd tnt-asr
uv sync
./bootstrap-mlx-asr.sh   # download and link the R2T2 checkpoint
uv run tnt
```

Or from PyPI: `uv tool install automaton-tnt`.

### Model checkpoints

TNT reads models from local paths only.

| Model | Checkpoint | Location |
|-------|------------|----------|
| Confucius4-R2T2 | [confucius4-r2t2-bf16-mlx](https://huggingface.co/appautomaton/confucius4-r2t2-bf16-mlx) | `TNT_R2T2_MODEL`, else `bin/r2t2-mlx`, else `~/.local/share/tnt/r2t2-mlx` |
| Qwen3-ASR | [qwen3-asr-1.7b-int8-mlx](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx) | `bin/qwen3-asr-mlx`, else `~/.local/share/tnt/qwen3-asr-mlx` |

TNT starts with the first installed model in this order. `./bootstrap-mlx-asr.sh` downloads R2T2; `./bootstrap-mlx-asr.sh qwen3` downloads Qwen3-ASR. Add a local path or a Hugging Face repo id after the model name to link or fetch a specific checkpoint.

## Configuration

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `TNT_MLX_MODEL` | first installed model | Startup checkpoint, of either model |
| `TNT_R2T2_MODEL` | see above | Confucius4-R2T2 checkpoint |
| `TNT_MLX_LANGUAGE` | `auto` | `Chinese`, `English`, or `auto`. R2T2 streams `auto` as `Chinese`, which handles mixed speech and keeps closing punctuation. For Qwen3-ASR, `Chinese` keeps mixed speech from being translated to English |
| `TNT_INPUT_DEVICE` | system default | Microphone, by index or name |

### Recognition context

TNT tells the model which names to expect. The default covers Anthropic, Claude, Claude Code, Opus, and Sonnet. `~/.config/tnt/context.txt` replaces the default entirely, so list every term you want, after a domain line: `Software engineering dictation. It may mention Anthropic, Claude, Claude Code, MLX, and Qwen.` A bare word list or a domain line alone barely helps. An empty file turns context off. TNT reads the file for each take.

## Keys

| Key | Action |
|-----|--------|
| <kbd>Space</kbd> | Start / stop recording (or hold to record); cancels a running transcription. A take stops and transcribes on its own after 3 minutes |
| <kbd>m</kbd> | Switch model |
| <kbd>c</kbd> | Copy the last entry |
| click | Copy the clicked entry |
| <kbd>x</kbd> | Clear the transcript |
| <kbd>q</kbd> | Quit |

## Related

- [mlx-speech](https://github.com/appautomaton/mlx-speech): the MLX speech runtime behind TNT
- [huggingface.co/appautomaton](https://huggingface.co/appautomaton): our MLX checkpoints

## License

MIT. See [`LICENSE`](LICENSE).
