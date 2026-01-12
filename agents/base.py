from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from .types import AgentResult, TaskInstance


class Agent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, instance: TaskInstance) -> AgentResult:
        raise NotImplementedError

    def config(self) -> Dict[str, str]:
        return {"name": self.name}
