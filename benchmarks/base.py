from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional, List

from agents.types import TaskInstance


class Benchmark(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def instances(self) -> Iterable[TaskInstance]:
        raise NotImplementedError

    def corpus(self) -> Optional[List[str]]:
        return None
