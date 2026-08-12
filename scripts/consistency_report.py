"""Cross-trial final-answer consistency report for a run directory.

Reads the same ``<label>.grades*.jsonl`` files as ``build_analysis_csv.py`` and
measures how often the repeated trials per (question, model) agree on their
final answer -- the multi-run consistency dimension from arXiv:2608.06202
("What Current AI Benchmarks Leave Unmeasured"). Single-run accuracy tables do
not surface this: a model can average 80% while still giving a different answer
on repeated runs of the same question.

Outputs (under ``<out-dir>``):
- ``consistency_by_question.csv`` -- one row per (qid, model, judge) group with
  >= 2 trials: agreement flags, distinct-answer count, the normalized answers.
- ``consistency_summary.csv`` -- one row per model_label plus an ``__overall__``
  row: the inconsistency rate (fraction of groups whose trials disagreed).

Usage:
    python scripts/consistency_report.py \\
        --run-dir runs/headline \\
        --out-dir runs/headline/analysis
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import click

from big_finance_harness.trial_consistency import (
    ConsistencySummary,
    GroupConsistency,
    compute_consistency,
    read_grade_records,
)


def _write_by_question(summary: ConsistencySummary, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "model_label",
                "judge",
                "n_trials",
                "text_consistent",
                "correctness_consistent",
                "distinct_answers",
                "normalized_answers",
            ],
        )
        w.writeheader()
        for g in summary.per_group:
            w.writerow(
                {
                    "question_id": g.question_id,
                    "model_label": g.model_label,
                    "judge": g.judge,
                    "n_trials": g.n_trials,
                    "text_consistent": g.text_consistent,
                    "correctness_consistent": g.correctness_consistent,
                    "distinct_answers": g.distinct_answers,
                    "normalized_answers": " | ".join(g.answers),
                }
            )


def _per_model(summary: ConsistencySummary) -> list[dict]:
    buckets: dict[str, list[GroupConsistency]] = defaultdict(list)
    for g in summary.per_group:
        buckets[g.model_label].append(g)
    rows: list[dict] = []
    for label, groups in sorted(buckets.items()):
        n = len(groups)
        n_inc = sum(1 for g in groups if not g.text_consistent)
        n_corr_inc = sum(1 for g in groups if not g.correctness_consistent)
        rows.append(
            {
                "model_label": label,
                "n_groups": n,
                "text_inconsistency_rate": round(n_inc / n, 4) if n else 0.0,
                "correctness_inconsistency_rate": round(n_corr_inc / n, 4) if n else 0.0,
            }
        )
    return rows


def _write_summary(summary: ConsistencySummary, out_path: Path) -> None:
    rows = _per_model(summary)
    rows.append(
        {
            "model_label": "__overall__",
            "n_groups": summary.n_groups,
            "text_inconsistency_rate": round(summary.text_inconsistency_rate, 4),
            "correctness_inconsistency_rate": round(summary.correctness_inconsistency_rate, 4),
        }
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model_label",
                "n_groups",
                "text_inconsistency_rate",
                "correctness_inconsistency_rate",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


@click.command()
@click.option("--run-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(path_type=Path))
def main(run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_grade_records(run_dir)
    click.echo(f"loaded {len(records):,} graded trials from {run_dir}")

    summary = compute_consistency(records)
    click.echo(f"  {summary.n_groups:,} (question, model, judge) groups with >=2 trials")

    by_q = out_dir / "consistency_by_question.csv"
    _write_by_question(summary, by_q)
    click.echo(f"wrote {by_q}")

    summ = out_dir / "consistency_summary.csv"
    _write_summary(summary, summ)
    click.echo(f"wrote {summ}")

    pct = summary.text_inconsistency_rate * 100
    click.echo(
        f"headline: {pct:.1f}% of groups gave inconsistent final answers across "
        f"repeated trials ({summary.n_text_inconsistent}/{summary.n_groups})"
    )


if __name__ == "__main__":
    main()
