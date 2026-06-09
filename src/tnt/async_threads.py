"""Helpers for running blocking work on daemon threads."""

import asyncio
import threading
from typing import Callable, TypeVar

T = TypeVar("T")


def start_daemon_thread(
    func: Callable[..., T],
    *args: object,
    name: str | None = None,
) -> asyncio.Future[T]:
    """Run a blocking callable on a daemon thread and return an asyncio Future."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def _set_result(result: T) -> None:
        if not future.done():
            future.set_result(result)

    def _set_exception(exc: BaseException) -> None:
        if not future.done():
            future.set_exception(exc)

    def runner() -> None:
        try:
            result = func(*args)
        except BaseException as exc:
            loop.call_soon_threadsafe(_set_exception, exc)
        else:
            loop.call_soon_threadsafe(_set_result, result)

    thread_name = name or f"tnt-{getattr(func, '__name__', 'worker')}"
    thread = threading.Thread(target=runner, name=thread_name, daemon=True)
    thread.start()
    return future
