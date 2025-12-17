"""Lightweight retry helpers.

Used for network/GCS/BigQuery operations to improve resiliency.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Type, TypeVar, Awaitable

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry policy parameters."""

    max_attempts: int = 5
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 10.0
    jitter_fraction: float = 0.2


def retry_call(
    func: Callable[[], T],
    *,
    policy: RetryPolicy = RetryPolicy(),
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> T:
    """Call `func` with retries and exponential backoff."""
    attempt = 0
    last_exc: Optional[BaseException] = None
    while attempt < policy.max_attempts:
        attempt += 1
        try:
            return func()
        except retry_on as exc:  # noqa: PERF203
            last_exc = exc
            if attempt >= policy.max_attempts:
                break

            delay = min(policy.base_delay_seconds * (2 ** (attempt - 1)), policy.max_delay_seconds)
            jitter = delay * policy.jitter_fraction * (2 * random.random() - 1)
            sleep_for = max(0.0, delay + jitter)
            if on_retry is not None:
                on_retry(attempt, exc)
            time.sleep(sleep_for)
    if last_exc:
        raise last_exc
    raise RuntimeError("Retry loop finished without exception (impossible)")


async def retry_async_call(
    func: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy = RetryPolicy(),
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> T:
    """Call async `func` with retries and exponential backoff."""
    attempt = 0
    last_exc: Optional[BaseException] = None
    while attempt < policy.max_attempts:
        attempt += 1
        try:
            return await func()
        except retry_on as exc:
            last_exc = exc
            if attempt >= policy.max_attempts:
                break

            delay = min(policy.base_delay_seconds * (2 ** (attempt - 1)), policy.max_delay_seconds)
            jitter = delay * policy.jitter_fraction * (2 * random.random() - 1)
            sleep_for = max(0.0, delay + jitter)
            if on_retry is not None:
                on_retry(attempt, exc)
            await asyncio.sleep(sleep_for)
    if last_exc:
        raise last_exc
    raise RuntimeError("Retry loop finished without exception (impossible)")


