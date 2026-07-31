"""Diagnose a model's weak capabilities from a run's rubric grades.

Sibling of ``build_analysis_csv.py``: it reads the same ``<label>.grades.*.jsonl``
files the analysis-CSV builder consumes, but instead of a flat per-grade table it
produces a CRAFT-style capability diagnosis — clusters the rubric criteria into a
hierarchical capability tree, scores the model at every node, and surfaces the
weak capabilities at the granularity where each failure is clearest. The method
and its adapted-port substitutions live in
``big_finance_harness.capability_diagnosis``.

Run after the eval + grade pipeline::

    python scripts/diagnose_capabilities.py \\
      --run-dir runs/headline \\
      --out-dir runs/headline/analysis
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click

from big_finance_harness.capability_diagnosis import (
    diagnose_weak_capabilities,
    format_report,
    probes_from_grades,
)
from big_finance_harness.types import GradedRun


def _load_graded_runs(run_dir: Path) -> list[GradedRun]:
    """Load every ``<label>.grades.*.jsonl`` row in the run dir as a GradedRun."""
    runs: list[GradedRun] = []
    for path in sorted(set(run_dir.glob("*.grades*.jsonl"))):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                runs.append(GradedRun.model_validate_json(line))
            except (ValueError, json.JSONDecodeError):
                # Skip malformed rows the same way build_analysis_csv does.
                continue
    return runs


@click.command()
@click.option("--run-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Where to write capability_diagnosis.json. Created if it doesn't exist.",
)
@click.option(
    "--weak-threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="Earned-points fraction below which a capability node counts as weak.",
)
@click.option(
    "--min-support",
    default=2,
    show_default=True,
    type=int,
    help="Minimum rubric criteria in a cluster to call it a capability.",
)
def main(run_dir: Path, out_dir: Path, weak_threshold: float, min_support: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"loading grades from {run_dir} ...")
    runs = _load_graded_runs(run_dir)
    click.echo(f"  {len(runs):,} graded runs")
    if not runs:
        raise click.ClickException(
            "no grade rows found; run scripts/run_eval_set.py and grade first"
        )

    probes = probes_from_grades(runs)
    click.echo(f"  {len(probes):,} rubric-criterion probes")

    diagnosis = diagnose_weak_capabilities(
        probes, weak_threshold=weak_threshold, min_support=min_support
    )
    click.echo(format_report(diagnosis))

    out_path = out_dir / "capability_diagnosis.json"
    out_path.write_text(json.dumps(asdict(diagnosis), indent=2), encoding="utf-8")
    click.echo(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
