from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval.metrics import constraint_adherence, evidence_coverage, rouge_l, semantic_similarity, token_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate run outputs")
    parser.add_argument("--input", default="runs/output.jsonl")
    parser.add_argument("--output", default="runs/metrics.json")
    return parser.parse_args()


def evaluate_runs(input_path: Path, output_path: Path) -> dict:
    aggregates = defaultdict(list)
    aggregates_by_benchmark = defaultdict(list)

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            task_type = record.get("task_type")
            benchmark = record.get("benchmark", "unknown")
            output = record["output"]
            reference = record.get("reference")
            evidence = record.get("evidence") or []

            if task_type == "sequential_consistency":
                adherence = constraint_adherence(output, evidence)
                aggregates[(record["agent"], "constraint_adherence")].append(adherence)
                aggregates_by_benchmark[(benchmark, record["agent"], "constraint_adherence")].append(
                    adherence
                )

            if reference:
                f1 = token_f1(output, reference)
                aggregates[(record["agent"], "token_f1")].append(f1)
                aggregates_by_benchmark[(benchmark, record["agent"], "token_f1")].append(f1)

            if task_type == "summarization" and reference:
                rouge = rouge_l(output, reference)
                sim = semantic_similarity(output, reference)
                aggregates[(record["agent"], "rouge_l")].append(rouge)
                aggregates[(record["agent"], "semantic_similarity")].append(sim)
                aggregates_by_benchmark[(benchmark, record["agent"], "rouge_l")].append(rouge)
                aggregates_by_benchmark[(benchmark, record["agent"], "semantic_similarity")].append(
                    sim
                )

            if evidence:
                coverage = evidence_coverage(output, evidence)
                aggregates[(record["agent"], "evidence_coverage")].append(coverage)
                aggregates_by_benchmark[(benchmark, record["agent"], "evidence_coverage")].append(
                    coverage
                )

    overall = {}
    for (agent, metric), vals in aggregates.items():
        overall.setdefault(agent, {})[metric] = sum(vals) / len(vals) if vals else 0.0

    by_benchmark = {}
    for (benchmark, agent, metric), vals in aggregates_by_benchmark.items():
        by_benchmark.setdefault(benchmark, {}).setdefault(agent, {})[metric] = (
            sum(vals) / len(vals) if vals else 0.0
        )

    aggregated = {"overall": overall, "by_benchmark": by_benchmark}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    evaluate_runs(input_path, output_path)


if __name__ == "__main__":
    main()
