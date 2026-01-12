# Paper Human-Agent Collaboration: Evaluation Harness

This repository contains an implementation plan and an evaluation harness for comparing four agent variants on long-context tasks.

## Quick start

1. Create a Python virtual environment.
2. Install dependencies from `pyproject.toml`.
3. Edit `config.toml` to choose benchmarks and limits.
4. Run the pipeline: `python pipeline.py --config config.toml`.

## Structure

- `agents/` agent definitions and orchestration
- `benchmarks/` dataset loaders and synthetic generators
- `eval/` metrics and evaluation scripts
- `runs/` raw outputs (JSONL)
- `viz/` plotting scripts and outputs
- `IMPLEMENTATION_PLAN.md` step-by-step implementation plan
- `config.toml` run configuration

## Notes

- Target hardware: NVIDIA 3060 Ti.
- Models will be configured for 7B/8B class with 4-bit quantization.
