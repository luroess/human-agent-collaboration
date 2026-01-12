from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass
class BenchmarkConfig:
    name: str
    limit: Optional[int] = None


@dataclass
class ModelConfig:
    model_id: str
    load_in_4bit: bool = True


@dataclass
class RunConfig:
    output: str = "runs/output.jsonl"
    seed: int = 42
    cache_dir: str = "hf_cache"


@dataclass
class EvalConfig:
    output: str = "runs/metrics.json"


@dataclass
class VizConfig:
    output_dir: str = "viz"


@dataclass
class AppConfig:
    model: ModelConfig
    run: RunConfig
    benchmarks: List[BenchmarkConfig]
    eval: EvalConfig
    viz: VizConfig


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    model = ModelConfig(**data.get("model", {}))
    run = RunConfig(**data.get("run", {}))
    eval_cfg = EvalConfig(**data.get("eval", {}))
    viz = VizConfig(**data.get("viz", {}))

    bench_entries = data.get("benchmarks", [])
    benchmarks = [BenchmarkConfig(**entry) for entry in bench_entries]
    if not benchmarks:
        benchmarks = [BenchmarkConfig(name="synthetic", limit=10)]

    return AppConfig(
        model=model,
        run=run,
        benchmarks=benchmarks,
        eval=eval_cfg,
        viz=viz,
    )
