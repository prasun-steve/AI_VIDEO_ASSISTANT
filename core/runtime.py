"""Shared runtime helpers for reliable calls to external AI services."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def invoke_with_retry(operation: Callable[[], T], service: str, attempts: int = 4) -> T:
    """Retry temporary API/network failures with bounded exponential backoff.

    Authentication and invalid-request errors are raised immediately: retrying them
    only makes a long meeting appear to hang.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:  # Providers expose different exception classes.
            message = str(error).lower()
            permanent = any(token in message for token in (
                "api key", "authentication", "unauthorized", "forbidden",
                "invalid api", "invalid_request", "not found",
            ))
            if permanent or attempt == attempts - 1:
                raise RuntimeError(
                    f"{service} failed after {attempt + 1} attempt(s): {error}"
                ) from error

            last_error = error
            delay = min(20, 2 ** attempt) + random.uniform(0, 0.5)
            print(f"{service} temporarily failed ({error}). Retrying in {delay:.1f}s…")
            time.sleep(delay)

    raise RuntimeError(f"{service} failed: {last_error}")
