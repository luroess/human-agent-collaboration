from __future__ import annotations

import random
from typing import Iterable, List

from agents.types import TaskInstance
from .base import Benchmark


class SyntheticLongSummaryBenchmark(Benchmark):
    def __init__(self, seed: int = 42, num_instances: int = 20, docs_per_instance: int = 8):
        super().__init__(name="synthetic_long_summary")
        self.seed = seed
        self.num_instances = num_instances
        self.docs_per_instance = docs_per_instance

    def _build_docs(self, rng: random.Random) -> List[str]:
        topics = [
            "context windows",
            "retrieval",
            "summarization",
            "multi-agent sequencing",
            "memory",
        ]
        docs = []
        for idx in range(self.docs_per_instance):
            topic = rng.choice(topics)
            docs.append(f"Section {idx}: This section focuses on {topic} and provides detail.")
        return docs

    def _build_instance(self, idx: int) -> TaskInstance:
        rng = random.Random(self.seed + idx)
        docs = self._build_docs(rng)
        key_points = [
            f"Finding A{idx}: retrieval improves recall.",
            f"Finding B{idx}: summarization reduces token cost.",
            f"Finding C{idx}: sequencing supports long context.",
        ]
        insert_pos = rng.sample(range(self.docs_per_instance), k=3)
        for pos, point in zip(insert_pos, key_points):
            docs[pos] = docs[pos] + " " + point

        prompt = (
            "Summarize the following multi-section document in 2-3 sentences.\n\n"
            + "\n".join(docs)
        )

        reference = " ".join(key_points)

        return TaskInstance(
            id=f"longsum-{idx}",
            task_type="summarization",
            input=prompt,
            reference=reference,
            evidence=key_points,
        )

    def instances(self) -> Iterable[TaskInstance]:
        return [self._build_instance(i) for i in range(self.num_instances)]
