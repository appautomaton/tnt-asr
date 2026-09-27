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
        color: #f3eee4;
    }

    TranscriptEntry:hover {
        background: #2c2822;
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
        color: #d4784a;
    }
    """


class TranscriptView(VerticalScroll):
    """Scrollable container of transcript entries."""

    DEFAULT_CSS = """
    TranscriptView {
        background: #161410;
        color: #f3eee4;
        padding: 1 2;
        scrollbar-size-vertical: 1;
        scrollbar-background: #161410;
        scrollbar-color: #3a342c;
        scrollbar-color-hover: #5a5046;
        scrollbar-color-active: #7d9a84;
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
        content.append(f"\n{text}", style="#f3eee4")

        self.mount(TranscriptEntry(content, raw_text=text, seq=seq))
        self.scroll_end(animate=False)

    @staticmethod
    def _build_meta(seq: int, duration: float) -> Text:
        """Build the muted metadata line for an entry."""
        utc_time = datetime.now(UTC).strftime("%H:%M:%S")
        meta = Text()
        meta.append(f"#{seq}", style="#d4784a")
        meta.append(f"  {duration:.1f}s  {utc_time}", style="#9c9386")
        return meta

    def show_live(self, text: str) -> None:
        """Show the committed text of the take that is still being recorded."""
        self.remove_placeholder()
        body = text if text else "…"
        try:
            node = self.query_one("#transcript-live", Static)
        except Exception:
            node = Static("", id="transcript-live")
            self.mount(node)
        node.update(Text(body, style="#f3eee4"))
        self.scroll_end(animate=False)

    def clear_live(self) -> None:
        """Remove the in-progress line."""
        try:
            self.query_one("#transcript-live").remove()
        except Exception:
            pass

    def show_placeholder(self) -> None:
        """Show a transcription-in-progress cursor."""
        self.remove_placeholder()
        self.mount(TranscriptPlaceholder("▊", id="transcript-placeholder"))
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
        self.clear_live()
        self.remove_placeholder()
