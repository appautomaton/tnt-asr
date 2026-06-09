"""Textual TUI app for voice-to-text transcription."""

import asyncio
import signal
import subprocess
import threading

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
    MlxQwenTranscriber,
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
        background: #100025;
        color: #f8f4ff;
        padding: 0 2;
    }
    """

    state: reactive[str] = reactive("idle")

    def render(self) -> Table:
        left = Text()
        left.append("● ", style="bold #39ff14")
        left.append("TNT", style="bold #ff4fd8")
        if self.size.width >= 56:
            left.append("  voice → text", style="#6f5fa8")

        right = Text()
        match self.state:
            case "idle":
                right.append("▮▮ IDLE", style="bold #7afcff")
            case "recording":
                right.append("● REC", style="bold #ff5ccf")
            case "stopping":
                right.append("◌ MIC", style="bold #ffd166")
            case "transcribing":
                right.append("◌ ...", style="bold #ffd166")

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
        height: 1;
        background: #140a2e;
        color: #f8f4ff;
        padding: 0 1;
    }
    """

    state: reactive[str] = reactive("idle")

    _KEY_STYLE = "bold #9c8fd9 on #221a40"
    _LABEL_STYLE = "#6f5fa8"

    def render(self) -> Text:
        match self.state:
            case "recording":
                action, action_color = "stop", "#ff8ad8"
            case "stopping":
                action, action_color = "wait", "#ffd166"
            case "transcribing":
                action, action_color = "cancel", "#ffd166"
            case _:
                action, action_color = "record", "#9bff7a"
        compact = self.size.width < 64
        text = Text()
        text.append(" Space ", style="bold #090014 on #39ff14")
        text.append(f" {action}  ", style=f"bold {action_color}")
        if compact:
            keys = [("c", "copy"), ("x", "clear"), ("q", "quit")]
        else:
            keys = [
                ("c", "copy last"),
                ("click", "copy entry"),
                ("x", "clear"),
                ("q", "quit"),
            ]
        for key, label in keys:
            text.append(f" {key} ", style=self._KEY_STYLE)
            text.append(f" {label}  ", style=self._LABEL_STYLE)
        return text


class TntApp(App):
    """Voice-to-text TUI powered by in-process MLX inference."""

    _SPACE_PENDING_STOP_SECONDS = 0.18
    _SPACE_HOLD_RELEASE_WINDOW_SECONDS = 0.30
    _NARROW_BREAKPOINT = 72

    CSS = """
    Screen {
        layout: vertical;
        background: #090014;
        color: #f8f4ff;
    }

    #main-layout {
        height: 1fr;
        margin: 1 1 0 1;
    }

    #main-layout TranscriptView {
        width: 3fr;
    }

    #main-layout StatusPanel {
        width: 1fr;
        margin: 0 0 0 1;
    }
    """

    BINDINGS = [
        Binding("space", "toggle_recording", "Record", show=False),
        Binding("c", "copy_last", "Copy last", show=False),
        Binding("x", "clear_transcript", "Clear", show=False),
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

    def _init_transcriber(self) -> MlxQwenTranscriber:
        """Lazily initialize the MLX transcriber."""
        if self._transcriber is None:
            self._transcriber = MlxQwenTranscriber()
        return self._transcriber

    def compose(self) -> ComposeResult:
        yield HeaderBar()
        with Horizontal(id="main-layout"):
            yield TranscriptView()
            yield StatusPanel(model_label=MODEL_LABEL)
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
            transcript.styles.width = "100%"
            transcript.styles.height = "1fr"
            status.styles.width = "100%"
            status.styles.height = COMPACT_PANEL_HEIGHT
            status.styles.margin = (1, 0, 0, 0)
            status.set_compact(True)
        else:
            layout.styles.layout = "horizontal"
            transcript.styles.width = "3fr"
            transcript.styles.height = "100%"
            status.styles.width = "1fr"
            status.styles.height = "100%"
            status.styles.margin = (0, 0, 0, 1)
            status.set_compact(False)

    def on_mount(self) -> None:
        self._apply_responsive_layout(self.size.width)
        # Validate the model directory now so config errors surface at startup,
        # and start loading the MLX model so take one is warm.
        try:
            self._init_transcriber().warmup()
        except Exception as exc:
            self.notify(f"ASR backend error: {exc}", severity="error")

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
                10,
            )
        except asyncio.TimeoutError:
            self._recreate_recorder()
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

    def _stop_recording(self) -> None:
        """Hand mic shutdown and transcription to a worker; never block the UI."""
        self._reset_space_recording_mode()
        if self._recording_timer is not None:
            self._recording_timer.stop()
            self._recording_timer = None
        duration = self.recorder.elapsed()

        self.state = "stopping"
        session_id = self._recording_session_id
        self.query_one(TranscriptView).show_placeholder()
        self._transcribe_worker = self.run_worker(
            self._stop_and_transcribe(session_id, duration)
        )

    async def _stop_and_transcribe(self, session_id: int, duration: float) -> None:
        """Async worker: stop capture and transcribe without blocking the UI."""
        tv = self.query_one(TranscriptView)
        try:
            # recorder.stop() aborts the stream and drains captured audio.
            wav_bytes = await asyncio.wait_for(
                asyncio.shield(
                    start_daemon_thread(
                        self.recorder.stop,
                        name="tnt-recorder-stop",
                    )
                ),
                10,
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
                    if label:
                        self.notify(f"Copied to clipboard ({label}).")
                except asyncio.TimeoutError:
                    pass
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
            self.notify("Clipboard not available; text stored in buffer.", severity="warning")

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
    app = TntApp()

    # On SIGINT (Ctrl-C), abandon in-flight work so worker threads unblock
    # and the recorder releases the input stream before exit.
    _orig_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(sig: int, frame: object) -> None:
        del sig, frame
        app._abort_inflight_work()
        app.exit()

    signal.signal(signal.SIGINT, _handle_sigint)

    # SIGTERM (`kill <pid>`) and SIGHUP (terminal closed) have no default
    # Python handler that runs our cleanup; handle them the same way.
    _term_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        _term_signals.append(signal.SIGHUP)
    _orig_term = {s: signal.getsignal(s) for s in _term_signals}

    def _handle_term(sig: int, frame: object) -> None:
        del sig, frame
        app._abort_inflight_work()
        app.exit()

    for _sig in _term_signals:
        signal.signal(_sig, _handle_term)

    try:
        app.run()
    finally:
        signal.signal(signal.SIGINT, _orig_sigint)
        for _sig, _handler in _orig_term.items():
            signal.signal(_sig, _handler)
        app._abort_inflight_work()


if __name__ == "__main__":
    main()
