from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from datasets import load_dataset

from agents.types import TaskInstance
from .base import Benchmark


COMMON_INPUT_FIELDS = ["input", "question", "prompt", "context", "document", "article", "transcript"]
COMMON_REFERENCE_FIELDS = ["reference", "answer", "summary", "output", "targets"]
COMMON_EVIDENCE_FIELDS = ["evidence", "supporting_facts", "facts"]


def _pick_field(sample: dict, candidates: list[str]) -> Optional[str]:
    for field in candidates:
        if field in sample:
            return field
    return None


@dataclass
class HFDatasetConfig:
    name: str
    subset: Optional[str] = None
    split: str = "validation"
    input_field: Optional[str] = None
    reference_field: Optional[str] = None
    evidence_field: Optional[str] = None
    task_type: str = "unknown"
    limit: Optional[int] = None
    preprocess: Optional[Callable[[dict], dict]] = None
    cache_dir: Optional[str] = None


class HFDatasetBenchmark(Benchmark):
    def __init__(self, config: HFDatasetConfig):
        super().__init__(name=config.name)
        self.config = config

    def instances(self) -> Iterable[TaskInstance]:
        try:
            print(
                "Loading dataset:",
                self.config.name,
                f"subset={self.config.subset}",
                f"split={self.config.split}",
            )
            dataset = load_dataset(
                self.config.name,
                self.config.subset,
                split=self.config.split,
                cache_dir=self.config.cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load dataset. Ensure it is cached locally or provide a local path."
            ) from exc

        if self.config.limit is not None:
            dataset = dataset.select(range(self.config.limit))

        sample = dataset[0]
        input_field = self.config.input_field or _pick_field(sample, COMMON_INPUT_FIELDS)
        reference_field = self.config.reference_field or _pick_field(sample, COMMON_REFERENCE_FIELDS)
        evidence_field = self.config.evidence_field or _pick_field(sample, COMMON_EVIDENCE_FIELDS)

        if input_field is None:
            raise ValueError(f"No input field found for dataset {self.config.name}")

        for row in dataset:
            if self.config.preprocess is not None:
                row = self.config.preprocess(row)
            reference = row.get(reference_field) if reference_field else None
            if isinstance(reference, list) and reference:
                reference = reference[0]
            evidence = row.get(evidence_field) if evidence_field else None
            yield TaskInstance(
                id=str(row.get("id", row.get("qid", row.get("example_id", "unknown")))),
                task_type=self.config.task_type,
                input=row[input_field],
                reference=reference,
                evidence=evidence,
                metadata={"dataset": self.config.name},
            )
