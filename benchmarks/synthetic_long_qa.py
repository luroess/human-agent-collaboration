from __future__ import annotations

import random
from typing import Iterable, List

from agents.types import TaskInstance
from .base import Benchmark


class SyntheticLongQABenchmark(Benchmark):
    def __init__(self, seed: int = 42, num_instances: int = 20, docs_per_instance: int = 12):
        super().__init__(name="synthetic_long_qa")
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
            "evaluation",
        ]
        docs = []
        for idx in range(self.docs_per_instance):
            topic = rng.choice(topics)
            docs.append(f"Document {idx}: This section discusses {topic} and related details.")
        return docs

    def _build_instance(self, idx: int) -> TaskInstance:
        rng = random.Random(self.seed + idx)
        docs = self._build_docs(rng)
        target_doc = rng.randrange(self.docs_per_instance)
        key_fact = f"Document {target_doc}: The target keyword is ALPHA-{idx}."
        docs[target_doc] = key_fact

        prompt = (
            "You are given multiple documents. Answer the question using the documents.\n\n"
            + "\n".join(docs)
            + "\n\nQuestion: What is the target keyword?"
        )

        return TaskInstance(
            id=f"longqa-{idx}",
            task_type="long_context_qa",
            input=prompt,
            reference=f"ALPHA-{idx}",
            evidence=[f"ALPHA-{idx}"],
        )

    def instances(self) -> Iterable[TaskInstance]:
        return [self._build_instance(i) for i in range(self.num_instances)]
