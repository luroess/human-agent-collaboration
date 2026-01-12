from __future__ import annotations

import random
from typing import Iterable, List

from agents.types import TaskInstance
from .base import Benchmark


class SyntheticRetrievalBenchmark(Benchmark):
    def __init__(self, seed: int = 42, num_instances: int = 20, corpus_size: int = 200):
        super().__init__(name="synthetic_retrieval")
        self.seed = seed
        self.num_instances = num_instances
        self.corpus_size = corpus_size
        self._rng = random.Random(seed)
        self._corpus = self._build_corpus()

    def _build_corpus(self) -> List[str]:
        topics = [
            "context windows",
            "retrieval augmented generation",
            "summarization",
            "multi-agent sequencing",
            "memory systems",
        ]
        corpus = []
        for i in range(self.corpus_size):
            topic = self._rng.choice(topics)
            corpus.append(f"Document {i}: This passage discusses {topic} and related ideas.")
        return corpus

    def corpus(self) -> List[str]:
        return self._corpus

    def _build_instance(self, idx: int) -> TaskInstance:
        rng = random.Random(self.seed + idx)
        target_doc = rng.randrange(self.corpus_size)
        keyword = rng.choice(["latency", "accuracy", "consistency", "trust"])
        fact = f"Document {target_doc} states that {keyword} is a key metric."
        self._corpus[target_doc] = fact
        prompt = (
            "Use the provided corpus to answer the question.\n"
            f"Question: Which document mentions {keyword} as a key metric?"
        )
        return TaskInstance(
            id=f"retrieval-{idx}",
            task_type="retrieval",
            input=prompt,
            reference=str(target_doc),
            evidence=[str(target_doc), keyword],
            metadata={"target_doc": target_doc, "keyword": keyword},
        )

    def instances(self) -> Iterable[TaskInstance]:
        return [self._build_instance(i) for i in range(self.num_instances)]
