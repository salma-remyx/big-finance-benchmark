"""Synthesize deep-research items from simple Big Finance QA via task evolution.

Loads simple `DatasetItem`s from a JSONL, runs each through the
Explorer -> Formalizer -> Challenger pipeline (`big_finance_harness.task_evolution`),
and writes an evolved JSONL whose rows are plain `DatasetItem`s — the exact schema
`scripts/run_eval_set.py` consumes, so the output drops into the eval pipeline with
no flag or schema changes:

    python scripts/evolve_tasks.py \\
      --input data/big_finance_subset.jsonl \\
      --output data/big_finance_evolved.jsonl \\
      --model openai:gpt-5.5 --limit 5

    python scripts/run_eval_set.py \\
      --dataset data/big_finance_evolved.jsonl --run-id evolved --kind dry_run

Adapted from "From Simple QA to Deep Research" (arXiv:2608.02163).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from big_finance_harness.models import make_client
from big_finance_harness.task_evolution import evolve_item
from big_finance_harness.types import DatasetItem


def _load_dataset(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(DatasetItem.model_validate_json(line))
    return items


async def _evolve_all(items: list[DatasetItem], model_id: str) -> list[DatasetItem]:
    client = make_client(model_id)
    evolved: list[DatasetItem] = []
    for item in items:
        result = await evolve_item(item=item, client=client)
        evolved.append(result.item)
        click.echo(
            f"[evolve] {item.id} -> {result.item.id} ({len(result.item.rubric)} rubric lines)"
        )
    return evolved


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--model", required=True, type=str, help="Evolution LLM in provider:snapshot form.")
@click.option("--limit", type=int, default=None, help="If set, evolve only the first N items.")
def main(
    input_path: Path,
    output_path: Path,
    model: str,
    limit: int | None,
) -> None:
    """Evolve simple QA items into deep-research items and write a pipeline-ready JSONL."""
    items = _load_dataset(input_path)
    if limit is not None:
        items = items[:limit]
    click.echo(f"[evolve] {len(items)} item(s) -> {model}")

    evolved = asyncio.run(_evolve_all(items, model))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for item in evolved:
            fh.write(item.model_dump_json() + "\n")
    click.echo(f"[evolve] wrote {len(evolved)} evolved item(s) to {output_path}")


if __name__ == "__main__":
    main()
