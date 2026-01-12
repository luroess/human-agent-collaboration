from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from agents.types import TaskInstance
from .base import Benchmark


@dataclass
class JSONLDatasetConfig:
    path: str
    task_type: str
    input_field: str = "input"
    reference_field: Optional[str] = "reference"
    evidence_field: Optional[str] = "evidence"
    limit: Optional[int] = None


class JSONLDatasetBenchmark(Benchmark):
    def __init__(self, name: str, config: JSONLDatasetConfig):
        super().__init__(name=name)
        self.config = config

    def instances(self) -> Iterable[TaskInstance]:
        path = Path(self.config.path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL dataset not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if self.config.limit is not None and idx >= self.config.limit:
                    break
                row = json.loads(line)
                yield TaskInstance(
                    id=str(row.get("id", idx)),
                    task_type=self.config.task_type,
                    input=row[self.config.input_field],
                    reference=row.get(self.config.reference_field) if self.config.reference_field else None,
                    evidence=row.get(self.config.evidence_field) if self.config.evidence_field else None,
                    metadata={"dataset": self.config.path},
                )
