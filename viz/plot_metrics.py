from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

METRIC_LABELS = {
    "constraint_adherence": "Constraint Adherence",
    "evidence_coverage": "Evidence Coverage",
    "token_f1": "Token F1",
    "rouge_l": "ROUGE-L (LCS)",
    "semantic_similarity": "Semantic Similarity",
}

AGENT_ORDER = ["long_context", "rag", "summarization", "sequenced"]
AGENT_COLORS = {
    "long_context": "#2E86AB",
    "rag": "#F18F01",
    "summarization": "#6D3B47",
    "sequenced": "#1B998B",
}
TASK_PLOTS = {
    "synthetic_long_qa": ("token_f1", "Long QA (Token F1)"),
    "synthetic_long_summary": ("rouge_l", "Long Summary (ROUGE-L)"),
    "synthetic_retrieval": ("evidence_coverage", "Retrieval (Evidence Coverage)"),
    "synthetic_constraints": ("constraint_adherence", "Constraints (Adherence)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics")
    parser.add_argument("--input", default="runs/metrics.json")
    parser.add_argument("--output-dir", default="viz")
    return parser.parse_args()


def _order_agents(values: dict) -> list:
    ordered = [agent for agent in AGENT_ORDER if agent in values]
    ordered.extend([agent for agent in values.keys() if agent not in ordered])
    return ordered


def _plot_metric(values: dict, metric: str, title: str, output_path: Path) -> None:
    agents = _order_agents(values)
    vals = [values[a] for a in agents]
    colors = [AGENT_COLORS.get(a, "#2E86AB") for a in agents]

    plt.figure(figsize=(6, 4))
    plt.bar(agents, vals, color=colors)
    plt.ylabel(METRIC_LABELS.get(metric, metric.replace("_", " ")))
    plt.xlabel("Agent")
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metrics(input_path: Path, output_dir: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "overall" in data:
        by_benchmark = data.get("by_benchmark", {})
    else:
        by_benchmark = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    for benchmark, (metric, title) in TASK_PLOTS.items():
        bench_scores = by_benchmark.get(benchmark)
        if not bench_scores:
            continue
        values = {}
        for agent, scores in bench_scores.items():
            if metric in scores:
                values[agent] = scores[metric]
        if values:
            filename = f"{metric}__{benchmark}.png"
            _plot_metric(values, metric, title, output_dir / filename)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    plot_metrics(input_path, output_dir)


if __name__ == "__main__":
    main()
