from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents import (
    HFModel,
    LongContextAgent,
    RAGAgent,
    RAGConfig,
    SummarizationAgent,
    SequencedMultiAgent,
)
from benchmarks import get_benchmark
from config import load_config
from eval.evaluate_runs import evaluate_runs
from viz.plot_metrics import plot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full benchmark pipeline")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--rag-cpu",
        action="store_true",
        help="Force RAG embeddings/indexing to run on CPU",
    )
    return parser.parse_args()


def load_corpus(instances) -> list[str]:
    return [inst.input for inst in instances]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_path = Path(cfg.run.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = HFModel(cfg.model.model_id, load_in_4bit=cfg.model.load_in_4bit)

    with output_path.open("w", encoding="utf-8") as f:
        for bench_cfg in cfg.benchmarks:
            benchmark = get_benchmark(
                bench_cfg.name,
                limit=bench_cfg.limit,
                cache_dir=cfg.run.cache_dir,
            )
            instances = list(benchmark.instances())
            corpus = benchmark.corpus() or load_corpus(instances)
            rag_config = RAGConfig(cache_dir=cfg.run.cache_dir, use_gpu=not args.rag_cpu)
            agents = [
                LongContextAgent(model),
                RAGAgent(model, corpus=corpus, rag_config=rag_config),
                SummarizationAgent(model),
                SequencedMultiAgent(model, model, model),
            ]
            for instance in instances:
                for agent in agents:
                    result = agent.run(instance)
                    record = {
                        "benchmark": benchmark.name,
                        "agent": agent.name,
                        "instance_id": instance.id,
                        "task_type": instance.task_type,
                        "output": result.text,
                        "reference": instance.reference,
                        "evidence": instance.evidence,
                        "tokens_in": result.tokens_in,
                        "tokens_out": result.tokens_out,
                        "latency_ms": result.latency_ms,
                        "metadata": result.metadata,
                    }
                    f.write(json.dumps(record) + "\n")

    metrics_path = Path(cfg.eval.output)
    evaluate_runs(output_path, metrics_path)

    output_dir = Path(cfg.viz.output_dir)
    plot_metrics(metrics_path, output_dir)


if __name__ == "__main__":
    main()
