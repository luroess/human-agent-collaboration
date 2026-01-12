from __future__ import annotations

import time
from typing import Callable, Tuple


class Timer:
    def __enter__(self) -> "Timer":
        self._start = time.time()
        self.elapsed_ms = 0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_ms = int((time.time() - self._start) * 1000)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def safe_call(func: Callable[[], Tuple[str, int, int]]) -> Tuple[str, int, int]:
    try:
        return func()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Model call failed: {exc}") from exc
