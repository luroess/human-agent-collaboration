from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .base import Agent
from .model import HFModel, GenerationConfig
from .types import AgentResult, TaskInstance


@dataclass
class SummaryState:
    decisions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = [
            "Decisions:\n- " + "\n- ".join(self.decisions) if self.decisions else "Decisions: (none)",
            "Constraints:\n- " + "\n- ".join(self.constraints) if self.constraints else "Constraints: (none)",
            "Open questions:\n- " + "\n- ".join(self.open_questions) if self.open_questions else "Open questions: (none)",
        ]
        return "\n\n".join(parts)


class SummarizationAgent(Agent):
    def __init__(self, model: HFModel, config: Optional[GenerationConfig] = None):
        super().__init__(name="summarization")
        self.model = model
        self.config = config
        self.state = SummaryState()

    def _update_state(self, summary_text: str) -> None:
        # Placeholder: in the first iteration, store the raw summary in decisions.
        # Later steps will replace this with a structured update prompt.
        self.state.decisions = [summary_text]

    def run(self, instance: TaskInstance) -> AgentResult:
        summary = self.state.render()
        prompt = (
            "You are an assistant that uses a running structured summary.\n"
            f"Summary so far:\n{summary}\n\n"
            f"Current input:\n{instance.input}\n\n"
            "Respond and update the summary in your answer."
        )
        result = self.model.generate(prompt, self.config)
        self._update_state(result["text"])
        return AgentResult(
            text=result["text"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=result["latency_ms"],
            metadata={"agent": self.name},
        )
