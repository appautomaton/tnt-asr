"""Scrollable transcript log widget."""

from datetime import UTC, datetime

from rich.text import Text

from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Static


class TranscriptEntry(Static):
    """A single transcript entry; click to copy it to the clipboard."""

    DEFAULT_CSS = """
    TranscriptEntry {
        padding: 0 1;
        margin: 0 0 1 0;
        color: #ece6fb;
    }

    TranscriptEntry:hover {
        background: #221a40;
    }
    """

    class Selected(Message):
        """Posted when the user clicks an entry."""

        def __init__(self, text: str, seq: int) -> None:
            super().__init__()
            self.text = text
            self.seq = seq

    def __init__(self, content: Text, raw_text: str, seq: int, **kwargs) -> None:
        super().__init__(content, **kwargs)
        self.raw_text = raw_text
        self.seq = seq

    def on_click(self) -> None:
        self.post_message(self.Selected(self.raw_text, self.seq))


class TranscriptPlaceholder(Static):
    """Placeholder shown during transcription."""

    DEFAULT_CSS = """
    TranscriptPlaceholder {
        padding: 0 1;
        margin: 0 0 1 0;
        color: #ff71ce;
    }
    """


class TranscriptView(VerticalScroll):
    """Scrollable container of transcript entries."""

    DEFAULT_CSS = """
    TranscriptView {
        background: #0d041f;
        color: #ece6fb;
        padding: 1 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries: list[str] = []

    def append(self, text: str, duration: float = 0.0) -> None:
        """Add a new entry and scroll to bottom."""
        self.remove_placeholder()
        seq = len(self._entries) + 1
        self._entries.append(text)

        content = Text()
        content.append_text(self._build_meta(seq, duration))
        content.append(f"\n{text}", style="#ece6fb")

        self.mount(TranscriptEntry(content, raw_text=text, seq=seq))
        self.scroll_end(animate=False)

    @staticmethod
    def _build_meta(seq: int, duration: float) -> Text:
        """Build the muted metadata line for an entry."""
        utc_time = datetime.now(UTC).strftime("%H:%M:%S")
        meta = Text()
        meta.append(f"#{seq}", style="bold #42f5ff")
        meta.append(f" · {duration:.1f}s · {utc_time} UTC", style="#6f5fa8")
        return meta

    def show_placeholder(self) -> None:
        """Show a transcription-in-progress cursor."""
        self.remove_placeholder()
        self.mount(TranscriptPlaceholder("[#ff71ce]▊[/]", id="transcript-placeholder"))
        self.scroll_end(animate=False)

    def remove_placeholder(self) -> None:
        """Remove the transcription placeholder if present."""
        try:
            self.query_one("#transcript-placeholder").remove()
        except Exception:
            pass

    def get_last(self) -> str:
        """Return the last transcript entry, or empty string."""
        return self._entries[-1] if self._entries else ""

    def clear(self) -> None:
        """Remove all transcript entries."""
        self._entries.clear()
        self.query(TranscriptEntry).remove()
        self.remove_placeholder()
