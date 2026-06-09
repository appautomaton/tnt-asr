import asyncio
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tnt.async_threads import start_daemon_thread


def test_start_daemon_thread_returns_result() -> None:
    async def run() -> int:
        future = start_daemon_thread(lambda value: value + 1, 2, name="test-worker")
        return await future

    assert asyncio.run(run()) == 3


def test_start_daemon_thread_survives_cancelled_wait() -> None:
    async def run() -> None:
        future = start_daemon_thread(time.sleep, 0.05, name="sleep-worker")

        async def waiter() -> None:
            await asyncio.shield(future)

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.1)
        assert future.done() is True

    asyncio.run(run())
