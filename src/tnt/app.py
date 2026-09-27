"""Textual TUI app for voice-to-text transcription."""

import asyncio
import os
import signal
import subprocess
import sys
import threading
import traceback

from rich.table import Table
from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Static

from tnt.async_threads import start_daemon_thread
from tnt.audio import Recorder, create_recorder
from tnt.transcriber import (
    MODEL_LABEL,
    LiveUtterance,
    MlxQwenTranscriber,
    model_choices,
    pump_live,
    recommended_timeout,
)
from tnt.widgets.status import COMPACT_PANEL_HEIGHT, StatusPanel
from tnt.widgets.transcript import TranscriptEntry, TranscriptView


class HeaderBar(Static):
    """Slim header: brand left, state indicator right."""

    DEFAULT_CSS = """
    HeaderBar {
        dock: top;
        height: 1;
        background: #1c1915;
        color: #f3eee4;
        padding: 0 2;
    }
    """

    state: reactive[str] = reactive("idle")
    model: reactive[str] = reactive("")

    def render(self) -> Table:
        left = Text()
        left.append("TNT", style="bold #f3eee4")
        if self.model:
            short = "R2T2" if "r2t2" in self.model.lower() else "Qwen3-ASR"
            left.append(f"  {short}", style="#7d9a84")

        right = Text()
        match self.state:
            case "idle":
                right.append("ready", style="#7d9a84")
            case "recording":
                right.append("recording", style="bold #d4784a")
            case "stopping":
                right.append("closing mic", style="#c4a36a")
            case "transcribing":
                right.append("writing", style="#c4a36a")

        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            expand=True,
            padding=0,
        )
        table.add_column(justify="left", ratio=1, no_wrap=True)
        table.add_column(justify="right", no_wrap=True)
        table.add_row(left, right)
        return table


class HintBar(Static):
    """Bottom bar showing keybindings with state-dependent labels."""

    DEFAULT_CSS = """
    HintBar {
        dock: bottom;
        height: auto;
        min-height: 1;
        max-height: 2;
        background: #1c1915;
        color: #f3eee4;
        padding: 0 1;
    }
    """

    state: reactive[str] = reactive("idle")

    _KEY_STYLE = "#f3eee4 on #3a342c"
    _LABEL_STYLE = "#9c9386"

    def render(self) -> Text:
        match self.state:
            case "recording":
                action, action_color = "stop", "#d4784a"
            case "stopping":
                action, action_color = "wait", "#c4a36a"
            case "transcribing":
                action, action_color = "cancel", "#c4a36a"
            case _:
                action, action_color = "record", "#7d9a84"
        width = self.size.width
        keys = [("c", "copy"), ("x", "clear"), ("m", "model"), ("q", "quit")]
        text = Text()
        text.append(" space ", style="bold #1c1915 on #d4784a")
        text.append(f" {action}", style=f"bold {action_color}")
        if width >= 60:
            # One line: every key with its label.
            for key, label in keys:
                text.append("  ")
                text.append(f" {key} ", style=self._KEY_STYLE)
                text.append(f" {label}", style=self._LABEL_STYLE)
            return text
        # Narrow: keys and labels on a second line, never unlabeled.
        text.append("\n")
        boxed = width >= 42
        for index, (key, label) in enumerate(keys):
            if index:
                text.append(" " if boxed else "  ")
            text.append(f" {key} " if boxed else key, style=self._KEY_STYLE)
            text.append(f" {label}", style=self._LABEL_STYLE)
        return text


class TntApp(App):
    """Voice-to-text TUI powered by in-process MLX inference."""

    _SPACE_PENDING_STOP_SECONDS = 0.18
    _SPACE_HOLD_RELEASE_WINDOW_SECONDS = 0.30
    _NARROW_BREAKPOINT = 72
    # Normal PortAudio calls finish in milliseconds; a wedged backend gets no
    # patience — abandon it and rebuild. Start keeps a little headroom because
    # Bluetooth devices legitimately take 1-2s to wake from sleep.
    _RECORDER_STOP_TIMEOUT_SECONDS = 1.0
    _RECORDER_START_TIMEOUT_SECONDS = 3.0

    CSS = """
    Screen {
        layout: vertical;
        background: #1b1916;
        color: #f3eee4;
    }

    #main-layout {
        height: 1fr;
        margin: 1 1 0 1;
    }

    #main-layout TranscriptView {
        width: 1fr;
    }

    #main-layout StatusPanel {
        width: 30;
        margin: 0 0 0 1;
    }
    """

    BINDINGS = [
        Binding("space", "toggle_recording", "Record", show=False),
        Binding("c", "copy_last", "Copy last", show=False),
        Binding("x", "clear_transcript", "Clear", show=False),
        Binding("m", "switch_model", "Switch model", show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    state: reactive[str] = reactive("idle")

    def __init__(self) -> None:
        super().__init__()
        self.recorder: Recorder
        self.recorder = create_recorder()
        self._transcriber: MlxQwenTranscriber | None = None
        self._recording_timer = None
        self._recording_session_id = 0
        self._transcribe_worker = None
        self._space_recording_mode = "ready"
        self._space_mode_generation = 0
        self._live_stop: threading.Event | None = None
        self._live_thread: threading.Thread | None = None
        self._live_text = ""
        self._live_final = ""
        self._live_error = ""

    def _init_transcriber_label(self) -> str:
        try:
            return self._init_transcriber().model_label
        except Exception:
            return MODEL_LABEL

    def action_switch_model(self) -> None:
        """m: cycle between installed models while idle; the old one is freed."""
        if self.state != "idle":
            self.notify("Finish the current take before switching models.")
            return
        choices = model_choices()
        if len(choices) < 2:
            self.notify("Only one model is installed.")
            return
        current = self._transcriber.model_dir.resolve() if self._transcriber else None
        paths = [path.resolve() for _, path in choices]
        index = paths.index(current) if current in paths else -1
        label, path = choices[(index + 1) % len(choices)]
        try:
            transcriber = MlxQwenTranscriber(model_dir=str(path))
        except Exception as exc:
            self.notify(f"Cannot switch to {label}: {exc}", severity="error")
            return
        self._transcriber = transcriber
        transcriber.activate()
        self.query_one(StatusPanel).set_model_label(label)
        self.query_one(HeaderBar).model = label
        mode = "live streaming" if "r2t2" in label.lower() else "transcribe after stop"
        self.notify(f"Model: {label} ({mode})")

    def _init_transcriber(self) -> MlxQwenTranscriber:
        """Lazily initialize the MLX transcriber."""
        if self._transcriber is None:
            self._transcriber = MlxQwenTranscriber()
        return self._transcriber

    def compose(self) -> ComposeResult:
        yield HeaderBar()
        with Horizontal(id="main-layout"):
            yield TranscriptView()
            yield StatusPanel(model_label=self._init_transcriber_label())
        yield HintBar()

    def on_resize(self, event) -> None:
        self._apply_responsive_layout(event.size.width)

    def _apply_responsive_layout(self, width: int) -> None:
        """Stack panels vertically when the terminal is narrow."""
        try:
            layout = self.query_one("#main-layout")
            transcript = self.query_one(TranscriptView)
            status = self.query_one(StatusPanel)
        except Exception:
            return
        if width < self._NARROW_BREAKPOINT:
            layout.styles.layout = "vertical"
            layout.styles.margin = 0
            transcript.styles.padding = (0, 1)
            transcript.styles.width = "100%"
            transcript.styles.height = "1fr"
            status.styles.width = "100%"
            status.styles.height = COMPACT_PANEL_HEIGHT
            status.styles.margin = 0
            status.set_compact(True)
        else:
            layout.styles.layout = "horizontal"
            layout.styles.margin = (1, 1, 0, 1)
            transcript.styles.padding = (1, 2)
            transcript.styles.width = "1fr"
            transcript.styles.height = "100%"
            status.styles.width = 30
            status.styles.height = "100%"
            status.styles.margin = (0, 0, 0, 1)
            status.set_compact(False)

    def on_mount(self) -> None:
        self._apply_responsive_layout(self.size.width)
        self.query_one(HeaderBar).model = self._init_transcriber_label()
        self._install_signal_handlers()
        # Validate the model directory now so config errors surface at startup,
        # and start loading the MLX model so take one is warm.
        try:
            self._init_transcriber().warmup()
        except Exception as exc:
            self.notify(f"ASR backend error: {exc}", severity="error")

    def _install_signal_handlers(self) -> None:
        """Exit cleanly on SIGINT/SIGTERM/SIGHUP.

        Must go through loop.add_signal_handler: a plain signal.signal handler
        interrupts the event loop's kevent wait, posts the exit message, and
        then the loop goes right back to sleep without processing it. The
        asyncio variant wakes the loop via its wakeup fd.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                loop.add_signal_handler(sig, self._handle_termination_signal)
            except (ValueError, NotImplementedError, RuntimeError):
                pass

    def _handle_termination_signal(self) -> None:
        """Runs on the event loop when an external termination signal lands."""
        self._abort_inflight_work()
        self.exit()

    def watch_state(self, value: str) -> None:
        try:
            self.query_one(HeaderBar).state = value
            self.query_one(StatusPanel).state = value
            self.query_one(HintBar).state = value
        except Exception:
            pass

    def _update_recording_info(self) -> None:
        """Periodic callback during recording to update timer and level."""
        if not self.recorder.is_recording:
            return
        panel = self.query_one(StatusPanel)
        panel.update_elapsed(self.recorder.elapsed())
        panel.push_level(self.recorder.get_level())
        utterance = getattr(self, "_utterance", None)
        if utterance is not None and utterance.text:
            self._live_text = utterance.text
            self.query_one(TranscriptView).show_live(utterance.text)

    def action_toggle_recording(self) -> None:
        """Space key: tap toggles, while a held key records until release."""
        match self.state:
            case "idle":
                self._start_recording()
            case "recording":
                self._handle_recording_space()
            case "stopping":
                return
            case "transcribing":
                self._cancel_transcription()

    def _handle_recording_space(self) -> None:
        """Interpret Space during recording as a tap stop or held-key repeat."""
        match self._space_recording_mode:
            case "hold":
                self._arm_space_hold_release_timer()
            case "pending_stop":
                self._space_recording_mode = "hold"
                self._arm_space_hold_release_timer()
            case _:
                self._space_recording_mode = "pending_stop"
                self._space_mode_generation += 1
                generation = self._space_mode_generation
                self.set_timer(
                    self._SPACE_PENDING_STOP_SECONDS,
                    lambda: self._resolve_pending_space_stop(generation),
                )

    def _arm_space_hold_release_timer(self) -> None:
        """Refresh the inferred release timer while key-repeat is still arriving."""
        self._space_mode_generation += 1
        generation = self._space_mode_generation
        self.set_timer(
            self._SPACE_HOLD_RELEASE_WINDOW_SECONDS,
            lambda: self._finish_space_hold(generation),
        )

    def _resolve_pending_space_stop(self, generation: int) -> None:
        """Commit a stop when a follow-up repeat does not arrive."""
        if generation != self._space_mode_generation:
            return

        if self.state == "recording" and self._space_recording_mode == "pending_stop":
            self._space_recording_mode = "ready"
            self._stop_recording()

    def _finish_space_hold(self, generation: int) -> None:
        """Stop recording once key-repeat stops, which approximates key release."""
        if generation != self._space_mode_generation:
            return

        if self.state == "recording" and self._space_recording_mode == "hold":
            self._space_recording_mode = "ready"
            self._stop_recording()

    def _reset_space_recording_mode(self) -> None:
        """Clear inferred Space press state and invalidate pending timers."""
        self._space_recording_mode = "ready"
        self._space_mode_generation += 1

    def action_quit(self) -> None:
        """Quit the app, abandoning any in-flight transcription first."""
        self._abort_inflight_work()
        self.exit()

    def _cancel_transcription(self) -> None:
        """Cancel a running transcription; its result is abandoned."""
        self._reset_space_recording_mode()
        if self._live_stop is not None:
            self._live_stop.set()
        if self._transcriber is not None:
            self._transcriber.abandon()
        if self._transcribe_worker is not None:
            self._transcribe_worker.cancel()

    def _abort_inflight_work(self) -> None:
        """Best-effort shutdown for recorder + transcription worker.

        Must never block: PortAudio calls can wedge, and this runs from
        signal handlers and quit paths where a frozen UI thread would make
        the app unkillable.
        """
        self._reset_space_recording_mode()
        if self._recording_timer is not None:
            self._recording_timer.stop()
            self._recording_timer = None
        recorder = self.recorder
        try:
            if recorder.is_recording:
                threading.Thread(
                    target=recorder.begin_stop,
                    name="tnt-recorder-abort",
                    daemon=True,
                ).start()
        except Exception:
            pass
        if self._live_stop is not None:
            self._live_stop.set()
        if self._transcriber is not None:
            self._transcriber.abandon()
        if self._transcribe_worker is not None:
            self._transcribe_worker.cancel()

    def _recreate_recorder(self) -> None:
        """Abandon a wedged recorder and build a fresh one."""
        try:
            self.recorder = create_recorder()
        except Exception as exc:
            self.notify(f"Recorder reset failed: {exc}", severity="error")

    def _start_recording(self) -> None:
        """Begin mic capture on a worker thread; the UI never touches PortAudio."""
        self._reset_space_recording_mode()
        self._recording_session_id += 1
        self.state = "recording"
        self.run_worker(self._start_capture(self._recording_session_id))

    async def _start_capture(self, session_id: int) -> None:
        """Async worker: open the input stream without blocking the UI."""
        try:
            await asyncio.wait_for(
                asyncio.shield(
                    start_daemon_thread(self.recorder.start, name="tnt-recorder-start")
                ),
                self._RECORDER_START_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            # The abandoned start() may still complete later; flag it stopped
            # so its post-start check closes the stream instead of leaving an
            # ownerless mic capture running.
            stuck = self.recorder
            self._recreate_recorder()
            threading.Thread(
                target=stuck.begin_stop, name="tnt-recorder-abandon", daemon=True
            ).start()
            if session_id == self._recording_session_id and self.state == "recording":
                self.state = "idle"
                self.notify(
                    "Mic start timed out; audio backend reset. Try again.",
                    severity="error",
                )
            return
        except Exception as e:
            if session_id == self._recording_session_id and self.state == "recording":
                self.state = "idle"
                self.notify(f"Mic error: {e}", severity="error")
            return

        if session_id != self._recording_session_id or self.state != "recording":
            # Stopped before the mic finished opening; clean up off-thread.
            threading.Thread(
                target=self.recorder.stop, name="tnt-recorder-cleanup", daemon=True
            ).start()
            return

        self._recording_timer = self.set_interval(0.1, self._update_recording_info)
        self._start_live(session_id)

    def _start_live(self, session_id: int) -> None:
        """Start feeding mic audio into the streaming model off the UI thread."""
        try:
            transcriber = self._init_transcriber()
            model = transcriber._load_model_locked()
        except Exception as exc:
            self.notify(f"ASR load failed: {exc}", severity="error")
            return
        if not hasattr(model, "stream_session"):
            return
        utterance = LiveUtterance(model, transcriber.language)
        stop = threading.Event()
        self._live_stop = stop
        self._utterance = utterance
        self._live_text = ""
        self._live_final = ""
        self._live_error = ""

        def run() -> None:
            try:
                self._live_final = pump_live(self.recorder, utterance, stop)
            except Exception as exc:
                self._live_error = str(exc)
            self._live_text = utterance.text

        self._live_thread = threading.Thread(
            target=run, name="tnt-live-asr", daemon=True
        )
        self._live_thread.start()
        del session_id

    async def _finish_live(self, session_id: int, duration: float, tv) -> None:
        """Stop the mic, then wait for the live decode to consume every sample.

        No timeout drops captured audio: the worker is awaited until it
        finishes, and Space (cancel) is the only way out early.
        """
        boundary_uncertain = False
        try:
            await asyncio.wait_for(
                asyncio.shield(
                    start_daemon_thread(
                        self.recorder.begin_stop, name="tnt-recorder-stop"
                    )
                ),
                self._RECORDER_STOP_TIMEOUT_SECONDS,
            )
        except Exception:
            boundary_uncertain = True
        await asyncio.sleep(0.2)
        if self._live_stop is not None:
            self._live_stop.set()
        thread = self._live_thread
        self._live_thread = None
        self._live_stop = None
        if thread is not None:
            if session_id == self._recording_session_id:
                self.state = "transcribing"
            try:
                await asyncio.shield(
                    start_daemon_thread(thread.join, name="tnt-live-join")
                )
            except asyncio.CancelledError:
                # Space cancelled: the worker's result is abandoned.
                tv.clear_live()
                self._live_text = ""
                if session_id == self._recording_session_id:
                    self.state = "idle"
                raise
        try:
            await asyncio.wait_for(
                asyncio.shield(
                    start_daemon_thread(self.recorder.stop, name="tnt-recorder-clear")
                ),
                self._RECORDER_STOP_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self._recreate_recorder()
        text = (self._live_final or self._live_text).strip()
        error = self._live_error
        self._live_text = ""
        tv.clear_live()
        if session_id != self._recording_session_id:
            return
        if error and not text:
            self.notify(f"ASR error: {error}", severity="error")
        elif text:
            tv.append(text, duration=duration)
            await self._copy_transcript(text)
            if error:
                self.notify(f"ASR error; transcript may be partial: {error}", severity="error")
            elif boundary_uncertain:
                self.notify(
                    "Mic stop timed out; the end of the recording may be missing.",
                    severity="warning",
                )
        else:
            self.notify("No audio captured.", severity="warning")
        self.state = "idle"

    def _stop_recording(self) -> None:
        """Hand mic shutdown and transcription to a worker; never block the UI."""
        self._reset_space_recording_mode()
        if self._recording_timer is not None:
            self._recording_timer.stop()
            self._recording_timer = None
        duration = self.recorder.elapsed()

        self.state = "stopping"
        session_id = self._recording_session_id
        if self._live_thread is None:
            self.query_one(TranscriptView).show_placeholder()
        self._transcribe_worker = self.run_worker(
            self._stop_and_transcribe(session_id, duration)
        )

    async def _stop_and_transcribe(self, session_id: int, duration: float) -> None:
        """Async worker: stop capture and transcribe without blocking the UI."""
        tv = self.query_one(TranscriptView)
        if self._live_thread is not None:
            await self._finish_live(session_id, duration, tv)
            return
        try:
            # recorder.stop() aborts the stream and drains captured audio.
            wav_bytes = await asyncio.wait_for(
                asyncio.shield(
                    start_daemon_thread(
                        self.recorder.stop,
                        name="tnt-recorder-stop",
                    )
                ),
                self._RECORDER_STOP_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            tv.remove_placeholder()
            self._recreate_recorder()
            self.notify(
                "Mic stop timed out (audio backend stuck); recorder reset.",
                severity="error",
            )
            if session_id == self._recording_session_id:
                self.state = "idle"
            return
        except Exception as e:
            tv.remove_placeholder()
            self.notify(f"Stop error: {e}", severity="error")
            if session_id == self._recording_session_id:
                self.state = "idle"
            return

        if not wav_bytes:
            tv.remove_placeholder()
            self.notify("No audio captured.", severity="warning")
            if session_id == self._recording_session_id:
                self.state = "idle"
            return

        try:
            if session_id == self._recording_session_id:
                self.state = "transcribing"
            transcriber = self._init_transcriber()
            timeout = recommended_timeout(duration)
            text = await transcriber.transcribe_async(wav_bytes, timeout=timeout)
            tv.remove_placeholder()
            if text:
                tv.append(text, duration=duration)
                await self._copy_transcript(text)
            else:
                self.notify("No speech detected.", severity="warning")
        except asyncio.TimeoutError as e:
            tv.remove_placeholder()
            detail = str(e).strip()
            if detail:
                self.notify(f"Transcription timed out: {detail}", severity="error")
            else:
                self.notify("Transcription timed out.", severity="error")
        except asyncio.CancelledError:
            if self._transcriber is not None:
                self._transcriber.abandon()
            tv.remove_placeholder()
            self.notify("Transcription cancelled.", severity="warning")
        except FileNotFoundError as e:
            tv.remove_placeholder()
            self.notify(str(e), severity="error")
        except RuntimeError as e:
            tv.remove_placeholder()
            self.notify(f"Transcription failed: {e}", severity="error")
        except Exception as e:
            tv.remove_placeholder()
            self.notify(f"Error: {e}", severity="error")
        finally:
            self._transcribe_worker = None
            if session_id == self._recording_session_id:
                self.state = "idle"

    async def _copy_transcript(self, text: str) -> None:
        """Copy a finished transcript. Same path for live and one-shot results."""
        try:
            label = await asyncio.wait_for(
                asyncio.shield(
                    start_daemon_thread(
                        self._try_clipboard_copy,
                        text,
                        name="tnt-clipboard-copy",
                    )
                ),
                timeout=5,
            )
        except asyncio.TimeoutError:
            return
        if label:
            self.notify(f"Copied to clipboard ({label}).")

    def action_copy_last(self) -> None:
        """Copy the last transcript entry to clipboard."""
        text = self.query_one(TranscriptView).get_last()
        if not text:
            self.notify("Nothing to copy.", severity="warning")
            return
        label = self._try_clipboard_copy(text)
        if label:
            self.notify(f"Copied to clipboard ({label}).")
        else:
            self.notify(
                "Clipboard not available; text stored in buffer.", severity="warning"
            )

    def on_transcript_entry_selected(self, message: TranscriptEntry.Selected) -> None:
        """Clicking a transcript entry copies it to the clipboard."""
        label = self._try_clipboard_copy(message.text)
        if label:
            self.notify(f"Copied #{message.seq} to clipboard ({label}).")
        else:
            self.notify("Clipboard not available.", severity="warning")

    def _try_clipboard_copy(self, text: str) -> str | None:
        """Try to copy text to system clipboard.

        Returns the backend label on success, or None on failure.
        Does NOT call self.notify() — callers handle notification so
        this method is safe to run in a worker thread via asyncio.to_thread.
        """
        commands: list[tuple[list[str], bool, str]] = [
            (["pbcopy"], True, "pbcopy"),
            (["wl-copy"], True, "wl-copy"),
            (["xclip", "-selection", "clipboard"], True, "xclip"),
        ]
        for cmd, use_stdin, label in commands:
            try:
                input_bytes = text.encode("utf-8") if use_stdin else None
                proc = subprocess.run(
                    cmd,
                    input=input_bytes,
                    capture_output=True,
                    timeout=2,
                )
                if proc.returncode == 0:
                    return label
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None

    def action_clear_transcript(self) -> None:
        """Clear all transcript entries."""
        self.query_one(TranscriptView).clear()
        self.notify("Transcript cleared.")


def main() -> None:
    # SIGINT/SIGTERM/SIGHUP are handled inside the app via the asyncio loop
    # (see TntApp._install_signal_handlers).
    try:
        app = TntApp()
    except RuntimeError as exc:
        # Capture backend setup failed (on macOS this is mandatory native
        # AVFoundation — typically missing Xcode command line tools).
        print(f"tnt: {exc}", file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)

    exit_code = 0
    try:
        app.run()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        app._abort_inflight_work()
        # Exit without running interpreter shutdown. sounddevice registers an
        # atexit hook that calls Pa_Terminate(), and a wedged PortAudio stream
        # deadlocks it — leaving a zombie python that keeps the microphone
        # captured and ignores Ctrl-C. Textual has already restored the
        # terminal at this point; the OS reclaims everything else.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
