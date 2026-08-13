"""Score provenance grounding of finished agent runs (FinRank-style).

Reads a ``<label>.traces.jsonl`` written by ``scripts/run_eval_set.py`` together
with the dataset it was run against, and scores — per question — how well the
agent's retrieved evidence covers the gold supporting passages and whether it
confuses them with known hard negatives. This is the FinRank
(arXiv:2608.07400) evidence-grounding measurement applied to this harness's
own trace format; see ``big_finance_harness/evidence_grounding.py`` for the
metric definitions (clean-room port of the standard IR measurements — no
FinRank code or dataset is redistributed).

The shipped dataset does not yet populate ``DatasetItem.sources``, so gold
passages and hard negatives are supplied via an optional provenance sidecar:
a JSONL of ``{"id", "sources", "hard_negatives"}`` records. Items without a
sidecar entry are reported with ``n_gold=0`` (recall/precision are then 0.0
and excluded from the means).

Example::

    python scripts/eval_evidence_grounding.py \\
        --traces out/opus47.traces.jsonl \\
        --dataset data/big_finance_subset.jsonl \\
        --provenance data/finrank_provenance.jsonl \\
        --k 10
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from big_finance_harness.evidence_grounding import EvidenceGroundingScore, score_evidence_grounding
from big_finance_harness.types import DatasetItem, RunRecord


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_dataset(path: Path) -> dict[str, DatasetItem]:
    items: dict[str, DatasetItem] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = DatasetItem.model_validate_json(line)
            items[item.id] = item
    return items


def _load_traces(path: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            runs.append(RunRecord.model_validate_json(line))
    return runs


def _load_provenance(path: Path | None) -> dict[str, dict[str, list[str]]]:
    """Sidecar mapping question id -> {"sources": [...], "hard_negatives": [...]}."""
    provenance: dict[str, dict[str, list[str]]] = {}
    if path is None:
        return provenance
    for record in _read_jsonl(path):
        qid = record.get("id")
        if not qid:
            continue
        provenance[qid] = {
            "sources": list(record.get("sources") or []),
            "hard_negatives": list(record.get("hard_negatives") or []),
        }
    return provenance


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(scores: list[EvidenceGroundingScore]) -> dict[str, float]:
    """Mean metrics. Recall/precision are averaged only over scored questions
    that actually carry gold evidence; pairwise accuracy only over questions
    that carry hard negatives — matching how FinRank partitions its tasks."""
    with_gold = [s for s in scores if s.n_gold > 0]
    with_negatives = [s for s in scores if s.n_hard_negatives > 0]
    return {
        "n_scored": len(scores),
        "n_with_gold": len(with_gold),
        "n_with_hard_negatives": len(with_negatives),
        "mean_recall_at_k": _mean([s.recall_at_k for s in with_gold]),
        "mean_provenance_precision": _mean([s.provenance_precision for s in with_gold]),
        "mean_pairwise_accuracy": _mean([s.pairwise_accuracy for s in with_negatives]),
    }


@click.command()
@click.option(
    "--traces",
    "traces_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Run traces JSONL (<label>.traces.jsonl from run_eval_set.py).",
)
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Dataset JSONL the traces were run against.",
)
@click.option(
    "--provenance",
    "provenance_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional sidecar JSONL of {id, sources, hard_negatives} records.",
)
@click.option("--k", type=int, default=10, show_default=True, help="Recall cutoff.")
@click.option(
    "--match-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Token-overlap threshold for a retrieved passage to cover a gold/negative.",
)
@click.option(
    "--tools",
    default=None,
    help="Comma-separated retrieval tool names to score (default: all evidence tools).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write per-question scores as JSONL.",
)
def main(
    traces_path: Path,
    dataset_path: Path,
    provenance_path: Path | None,
    k: int,
    match_threshold: float,
    tools: str | None,
    out_path: Path | None,
) -> None:
    """Score evidence grounding for every trace against its dataset item."""
    items = _load_dataset(dataset_path)
    runs = _load_traces(traces_path)
    provenance = _load_provenance(provenance_path)
    tool_names = {t.strip() for t in tools.split(",")} if tools else None

    scores: list[EvidenceGroundingScore] = []
    skipped = 0
    for run in runs:
        item = items.get(run.question_id)
        if item is None:
            skipped += 1
            continue
        sidecar = provenance.get(run.question_id, {})
        if sidecar.get("sources"):
            # Overlay the sidecar's gold passages onto the item without mutating
            # the shared dataset object.
            item = item.model_copy(update={"sources": sidecar["sources"]})
        scores.append(
            score_evidence_grounding(
                run,
                item,
                hard_negatives=sidecar.get("hard_negatives"),
                k=k,
                match_threshold=match_threshold,
                tool_names=tool_names,
            )
        )

    summary = _aggregate(scores)
    click.echo(
        f"Scored {summary['n_scored']} runs "
        f"({summary['n_with_gold']} with gold evidence, "
        f"{summary['n_with_hard_negatives']} with hard negatives); "
        f"{skipped} skipped (no matching dataset item)."
    )
    click.echo(
        f"mean Recall@{k}           : {summary['mean_recall_at_k']:.3f}  "
        f"(over {summary['n_with_gold']})"
    )
    click.echo(
        f"mean provenance precision : {summary['mean_provenance_precision']:.3f}  "
        f"(over {summary['n_with_gold']})"
    )
    click.echo(
        f"mean pairwise accuracy    : {summary['mean_pairwise_accuracy']:.3f}  "
        f"(over {summary['n_with_hard_negatives']})"
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for score in scores:
                fh.write(json.dumps(score.as_dict(), ensure_ascii=False) + "\n")
        click.echo(f"wrote {len(scores)} per-question scores -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
