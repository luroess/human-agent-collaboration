from __future__ import annotations

from typing import Optional

from .base import Agent
from .model import HFModel, GenerationConfig
from .types import AgentResult, TaskInstance


class LongContextAgent(Agent):
    def __init__(self, model: HFModel, config: Optional[GenerationConfig] = None):
        super().__init__(name="long_context")
        self.model = model
        self.config = config

    def run(self, instance: TaskInstance) -> AgentResult:
        result = self.model.generate(instance.input, self.config)
        return AgentResult(
            text=result["text"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=result["latency_ms"],
            metadata={"agent": self.name},
        )
