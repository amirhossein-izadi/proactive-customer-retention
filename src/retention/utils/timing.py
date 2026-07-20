"""Timing helper for pipeline stages."""

from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed(stage: str):
    start = time.perf_counter()
    yield
    print(f"[{stage}] completed in {time.perf_counter() - start:.1f}s")
