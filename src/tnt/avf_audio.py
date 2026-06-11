"""Native AVFoundation mic capture for macOS via an isolated helper process.

The helper (mic_helper.swift, compiled on demand) captures audio with
AVAudioEngine and streams 16 kHz mono s16le PCM over stdout. Keeping capture
in a child process is the whole point: if the audio stack ever wedges, the
app kills the process and macOS releases the microphone at the process
boundary — a guarantee in-process PortAudio could never provide.

Helper protocol:
- stderr "TNT_READY" once capture runs, "TNT_ERROR: <msg>" on startup failure
- SIGTERM (or the parent dying, detected via stdin EOF) stops capture
"""

import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from tnt.audio import encode_wav, int16_level, resolve_input_device

_SWIFT_SOURCE = Path(__file__).resolve().parent / "mic_helper.swift"
_CACHE_DIR = Path.home() / "Library" / "Caches" / "tnt"
_COMPILE_TIMEOUT_SECONDS = 120
# How long stop() waits for the helper to flush and exit after SIGTERM
# before escalating to SIGKILL. Grace + kill-wait + reader join must stay
# under the app-level 1s stop timeout.
_TERM_GRACE_SECONDS = 0.5


def helper_binary_path() -> Path:
    """Cache path for the compiled helper, keyed by source hash."""
    digest = hashlib.sha256(_SWIFT_SOURCE.read_bytes()).hexdigest()[:12]
    return _CACHE_DIR / f"tnt-mic-{digest}"


def ensure_helper() -> Path:
    """Return the compiled helper binary, building it with swiftc if needed."""
    binary = helper_binary_path()
    if binary.exists():
        return binary

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    staging = binary.with_name(f"{binary.name}.build-{os.getpid()}")
    try:
        result = subprocess.run(
            ["xcrun", "swiftc", "-O", "-o", str(staging), str(_SWIFT_SOURCE)],
            capture_output=True,
            text=True,
            timeout=_COMPILE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "AVFoundation capture unavailable: xcrun/swiftc not found. "
            "Install the Xcode command line tools (xcode-select --install)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("AVFoundation helper compile timed out.") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-800:]
        raise RuntimeError(f"AVFoundation helper failed to compile:\n{detail}")

    os.replace(staging, binary)
    return binary


class AVFRecorder:
    """Records from the microphone through the AVFoundation helper process.

    Implements the same Recorder protocol as MicRecorder. Every operation here
    is killable from Python (it is all pipes and signals), so the app's
    daemon-thread timeouts can never strand a live microphone.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
        helper_path: Path | str | None = None,
    ) -> None:
        if channels != 1:
            raise ValueError("AVFRecorder only captures mono audio")

        self.sample_rate = sample_rate
        self.channels = channels
        self.device = resolve_input_device(device)
        self.helper_path = Path(helper_path) if helper_path else ensure_helper()

        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._recording = False
        self._start_time: float = 0.0
        self._current_level: float = 0.0
        self._last_helper_error: str = ""

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        """Spawn the helper and wait for its capture handshake."""
        if self._recording:
            return

        with self._lock:
            self._chunks = []
            self._current_level = 0.0
        self._last_helper_error = ""

        self._start_time = time.monotonic()
        self._recording = True

        cmd = [str(self.helper_path), "--rate", str(self.sample_rate)]
        if self.device is not None:
            cmd += ["--device", str(self.device)]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            self._recording = False
            raise RuntimeError(self._build_mic_error(str(exc))) from exc

        # Publish the process before the (blocking) handshake read so a
        # concurrent begin_stop() can SIGTERM it and unblock us via EOF.
        self._proc = proc

        if not self._wait_for_ready(proc):
            error = self._last_helper_error or "mic helper exited before capture started"
            self._recording = False
            self._proc = None
            self._terminate(proc, immediate=True)
            raise RuntimeError(self._build_mic_error(error))

        self._reader = threading.Thread(
            target=self._drain_stdout,
            args=(proc,),
            name="tnt-avf-reader",
            daemon=True,
        )
        self._reader.start()
        threading.Thread(
            target=self._drain_stderr,
            args=(proc,),
            name="tnt-avf-stderr",
            daemon=True,
        ).start()

        # start() runs on a worker thread; if a stop raced ahead of us while
        # the helper was launching, shut it down immediately.
        if not self._recording:
            self._proc = None
            self._terminate(proc)

    def stop(self) -> bytes:
        """Stop the helper, drain captured audio, return WAV bytes."""
        if self._recording:
            self.begin_stop()

        proc = self._proc
        self._proc = None
        if proc is not None:
            self._terminate(proc)

        reader = self._reader
        self._reader = None
        if reader is not None:
            reader.join(timeout=0.2)

        with self._lock:
            if not self._chunks:
                return b""
            audio_data = np.concatenate(self._chunks)
            self._chunks = []

        return encode_wav(audio_data, self.sample_rate, self.channels)

    def begin_stop(self) -> None:
        """Signal the helper to stop so the mic turns off promptly."""
        if not self._recording:
            return

        self._recording = False
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    def elapsed(self) -> float:
        """Seconds since start() was called."""
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    def get_level(self) -> float:
        """Current RMS amplitude normalized to 0.0-1.0."""
        with self._lock:
            return self._current_level

    def _wait_for_ready(self, proc: subprocess.Popen) -> bool:
        """Block until the helper reports TNT_READY (True) or fails (False)."""
        stream = proc.stderr
        if stream is None:
            return False
        while True:
            line = stream.readline()
            if not line:
                return False
            text = line.decode("utf-8", errors="replace").strip()
            if text == "TNT_READY":
                return True
            if text.startswith("TNT_ERROR:"):
                self._last_helper_error = text.removeprefix("TNT_ERROR:").strip()
                return False

    def _terminate(self, proc: subprocess.Popen, immediate: bool = False) -> None:
        """SIGTERM the helper, escalating to SIGKILL if it does not exit.

        Process death is the release guarantee: even a fully wedged audio
        stack loses the microphone when macOS reaps the helper.
        """
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=0.05 if immediate else _TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=0.2)
            except Exception:
                pass
        except Exception:
            pass
        for stream in (proc.stdin,):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    def _drain_stdout(self, proc: subprocess.Popen) -> None:
        """Reader thread: accumulate PCM chunks until the helper exits."""
        stream = proc.stdout
        if stream is None:
            return
        pending = b""
        while True:
            try:
                data = stream.read(4096)
            except Exception:
                break
            if not data:
                break
            data = pending + data
            usable = len(data) - (len(data) % 2)
            pending = data[usable:]
            if not usable:
                continue
            chunk = np.frombuffer(data[:usable], dtype=np.int16).copy()
            normalized = int16_level(chunk)
            with self._lock:
                self._chunks.append(chunk)
                self._current_level = normalized
        try:
            stream.close()
        except Exception:
            pass

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Keep the helper's stderr pipe from filling; remember the last line."""
        stream = proc.stderr
        if stream is None:
            return
        for raw in stream:
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                self._last_helper_error = text
        try:
            stream.close()
        except Exception:
            pass

    def _build_mic_error(self, base_error: str) -> str:
        """Create a user-facing mic error with actionable setup hints."""
        lines = [base_error]
        lines.append("Set TNT_INPUT_DEVICE to an input index or device name if needed.")
        lines.extend(self._list_input_hints())
        return "\n".join(lines)

    def _list_input_hints(self, limit: int = 5) -> list[str]:
        """Ask the helper to enumerate input devices for error messages."""
        try:
            result = subprocess.run(
                [str(self.helper_path), "--list"],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return ["No audio device list available from AVFoundation."]
        hints = [line for line in result.stdout.splitlines() if line.strip()]
        if not hints:
            return ["No input-capable devices reported by AVFoundation."]
        return hints[:limit]
