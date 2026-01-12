from __future__ import annotations

from typing import Optional

from .hf_dataset import HFDatasetBenchmark, HFDatasetConfig
from .synthetic import SyntheticConstraintBenchmark
from .synthetic_retrieval import SyntheticRetrievalBenchmark
from .synthetic_long_qa import SyntheticLongQABenchmark
from .synthetic_long_summary import SyntheticLongSummaryBenchmark


def get_benchmark(name: str, limit: Optional[int] = None, cache_dir: Optional[str] = None):
    name = name.lower()
    if name == "synthetic":
        return SyntheticConstraintBenchmark(num_instances=limit or 20)

    if name == "synthetic_retrieval":
        return SyntheticRetrievalBenchmark(num_instances=limit or 20)

    if name == "synthetic_long_qa":
        return SyntheticLongQABenchmark(num_instances=limit or 20)

    if name == "synthetic_long_summary":
        return SyntheticLongSummaryBenchmark(num_instances=limit or 20)

    if name == "longbench":
        raise ValueError("LongBench requires a dataset script and is not supported by datasets>=4.")

    if name == "leval":
        raise ValueError("L-Eval requires a dataset script and is not supported by datasets>=4.")

    if name == "govreport":
        raise ValueError("GovReport requires a dataset script and is not supported by datasets>=4.")

    if name == "qmsum":
        raise ValueError("QMSum requires a dataset script and is not supported by datasets>=4.")

    raise ValueError(f"Unknown benchmark: {name}")
