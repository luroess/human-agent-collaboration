from __future__ import annotations

from typing import Optional

from .base import Agent
from .model import HFModel, GenerationConfig
from .types import AgentResult, TaskInstance


class SequencedMultiAgent(Agent):
    def __init__(
        self,
        worker_a: HFModel,
        worker_b: HFModel,
        coordinator: HFModel,
        config: Optional[GenerationConfig] = None,
    ):
        super().__init__(name="sequenced")
        self.worker_a = worker_a
        self.worker_b = worker_b
        self.coordinator = coordinator
        self.config = config

    def run(self, instance: TaskInstance) -> AgentResult:
        text = instance.input
        midpoint = max(1, len(text) // 2)
        part_a = text[:midpoint]
        part_b = text[midpoint:]

        result_a = self.worker_a.generate(f"Process part A:\n{part_a}", self.config)
        result_b = self.worker_b.generate(f"Process part B:\n{part_b}", self.config)

        merge_prompt = (
            "Combine the following two analyses into a single answer.\n\n"
            f"Analysis A:\n{result_a['text']}\n\n"
            f"Analysis B:\n{result_b['text']}\n\n"
            "Final answer:"
        )
        result = self.coordinator.generate(merge_prompt, self.config)

        return AgentResult(
            text=result["text"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=result["latency_ms"],
            metadata={
                "agent": self.name,
                "worker_a_tokens": result_a["tokens_in"],
                "worker_b_tokens": result_b["tokens_in"],
            },
        )
