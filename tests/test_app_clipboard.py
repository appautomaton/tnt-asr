import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tnt.app import TntApp  # noqa: E402
from tnt.widgets.transcript import TranscriptEntry, TranscriptView  # noqa: E402


def test_clicking_entry_copies_it_to_clipboard(monkeypatch) -> None:
    copied: list[str] = []

    async def scenario() -> None:
        app = TntApp()
        monkeypatch.setattr(
            app, "_try_clipboard_copy", lambda text: copied.append(text) or "test"
        )
        async with app.run_test(size=(100, 28)) as pilot:
            await pilot.pause(0.1)
            tv = app.query_one(TranscriptView)
            tv.append("first entry", duration=1.0)
            tv.append("second entry", duration=2.0)
            await pilot.pause(0.1)
            entries = list(app.query(TranscriptEntry))
            assert len(entries) == 2
            await pilot.click(entries[0])
            await pilot.pause(0.1)

    asyncio.run(scenario())
    assert copied == ["first entry"]
