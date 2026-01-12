from __future__ import annotations

import random
from typing import Iterable, List

from agents.types import TaskInstance
from .base import Benchmark


class SyntheticConstraintBenchmark(Benchmark):
    def __init__(self, seed: int = 42, num_instances: int = 20):
        super().__init__(name="synthetic_constraints")
        self.seed = seed
        self.num_instances = num_instances

    def _build_instance(self, idx: int) -> TaskInstance:
        rng = random.Random(self.seed + idx)
        constraints = [
            f"Use exactly {rng.randint(2, 5)} bullet points.",
            f"Mention the keyword '{rng.choice(['memory', 'context', 'retrieval', 'summary'])}'.",
            f"Avoid the word '{rng.choice(['maybe', 'possibly', 'perhaps'])}'.",
        ]
        prompt = (
            "You are collaborating on a short response."
            " Follow the constraints below and answer the question.\n\n"
            "Constraints:\n- "
            + "\n- ".join(constraints)
            + "\n\nQuestion: Summarize the core idea of long-context collaboration."
        )
        return TaskInstance(
            id=f"synthetic-{idx}",
            task_type="sequential_consistency",
            input=prompt,
            reference=None,
            evidence=constraints,
        )

    def instances(self) -> Iterable[TaskInstance]:
        return [self._build_instance(i) for i in range(self.num_instances)]
