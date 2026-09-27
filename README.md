# TNT 🧨

[![Website](https://img.shields.io/badge/website-appautomaton.com-ff4fd8?logo=github&logoColor=white)](https://appautomaton.com/tnt-asr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-automaton--tnt-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/automaton-tnt/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/documentation/apple-silicon)

Terminal voice-to-text for Apple Silicon. Tap <kbd>Space</kbd>, speak, tap <kbd>Space</kbd>. The text lands in the transcript and on the clipboard.

Models run locally on the Apple GPU through [mlx-speech](https://github.com/appautomaton/mlx-speech). No cloud, no PyTorch.

- **Qwen3-ASR-1.7B** (default): transcribes each take after you stop.
- **Confucius4-R2T2** (optional): transcribes live while you speak.

Press <kbd>m</kbd> to switch models. Only the active model stays in memory.

> [!NOTE]
> Using Termux on Android? Use the legacy `legacy/android-termux-qwen0.6b` branch.

## Setup

Requires an Apple Silicon Mac, Python 3.13+, [uv](https://docs.astral.sh/uv/), and the Xcode command line tools (`xcode-select --install`) for the Swift mic helper.

```bash
git clone https://github.com/appautomaton/tnt-asr.git
cd tnt-asr
uv sync
./bootstrap-mlx-asr.sh   # link or download the Qwen3-ASR checkpoint
uv run tnt
```

Or from PyPI: `uv tool install automaton-tnt`.

### Model checkpoints

TNT reads models from local paths only.

| Model | Checkpoint | Location |
|-------|------------|----------|
| Qwen3-ASR | [qwen3-asr-1.7b-int8-mlx](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx) | `TNT_MLX_MODEL`, else `bin/qwen3-asr-mlx`, else `~/.local/share/tnt/qwen3-asr-mlx` |
| Confucius4-R2T2 | [confucius4-r2t2-bf16-mlx](https://huggingface.co/appautomaton/confucius4-r2t2-bf16-mlx) | `TNT_R2T2_MODEL`, else `bin/r2t2-mlx`, else `~/.local/share/tnt/r2t2-mlx` |

`./bootstrap-mlx-asr.sh /path/to/checkpoint` links a Qwen3-ASR checkpoint you already have. With no argument it downloads the int8 build. For R2T2, symlink your checkpoint to one of the locations above.

## Configuration

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `TNT_MLX_MODEL` | see above | Qwen3-ASR checkpoint |
| `TNT_R2T2_MODEL` | see above | Confucius4-R2T2 checkpoint |
| `TNT_MLX_LANGUAGE` | `auto` | `Chinese`, `English`, or `auto`. `Chinese` keeps mixed Chinese/English speech from being translated to English |
| `TNT_INPUT_DEVICE` | system default | Microphone, by index or name |

## Keys

| Key | Action |
|-----|--------|
| <kbd>Space</kbd> | Start / stop recording (or hold to record); cancels a running transcription |
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
