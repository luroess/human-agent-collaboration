from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from agents import (
    HFModel,
    LongContextAgent,
    RAGAgent,
    RAGConfig,
    SummarizationAgent,
    SequencedMultiAgent,
)
from benchmarks import get_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent benchmarks")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B", help="HF model id")
    parser.add_argument("--output", default="runs/output.jsonl")
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["synthetic"],
        help="Benchmark names (synthetic, synthetic_long_qa, synthetic_long_summary, synthetic_retrieval)",
    )
    parser.add_argument("--cache-dir", default="hf_cache", help="HF datasets cache dir")
    parser.add_argument(
        "--rag-cpu",
        action="store_true",
        help="Force RAG embeddings/indexing to run on CPU",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip instances already present in the output file",
    )
    return parser.parse_args()


def load_corpus(instances) -> List[str]:
    return [inst.input for inst in instances]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                key = (record.get("benchmark"), record.get("agent"), record.get("instance_id"))
                seen.add(key)

    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        model = HFModel(args.model_id, load_in_4bit=True)
        for bench_name in args.benchmarks:
            try:
                benchmark = get_benchmark(bench_name, limit=args.instances, cache_dir=args.cache_dir)
            except ValueError as exc:
                print(f"Skipping benchmark '{bench_name}': {exc}")
                continue
            instances = list(benchmark.instances())
            corpus = benchmark.corpus() or load_corpus(instances)
            print(f"Running benchmark: {benchmark.name} ({len(instances)} instances)")
            rag_config = RAGConfig(cache_dir=args.cache_dir, use_gpu=not args.rag_cpu)
            agents = [
                LongContextAgent(model),
                RAGAgent(model, corpus=corpus, rag_config=rag_config),
                SummarizationAgent(model),
                SequencedMultiAgent(model, model, model),
            ]
            for instance in instances:
                for agent in agents:
                    key = (benchmark.name, agent.name, instance.id)
                    if args.resume and key in seen:
                        continue
                    print(f"  Agent: {agent.name} | Instance: {instance.id}")
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


if __name__ == "__main__":
    main()
