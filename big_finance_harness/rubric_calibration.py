"""Bayesian rubric measurability filtering and compact bank assembly.

Which rubric lines can the judge panel actually measure? A line every judge
scores identically on every run carries no signal about the evaluated model —
but "no disagreement" from a small panel is weak evidence of "no variance".
Following CalibratedRubric (arXiv:2607.29252), each line's measurability is a
Beta-Bernoulli posterior over the probability that two judges agree on it:

    p_agree(line) ~ Beta(alpha + agreements, beta + disagreements)

with a uniform Beta(1, 1) prior. A line is *measurable* when the posterior
concentrates on intermediate agreement — the panel reliably distinguishes it —
and *non-measurable* when the posterior mass sits near certain-agreement,
i.e. the lower bound of the credible interval on p_disagree exceeds a floor.
Lines the panel cannot measure are filtered before aggregation.

The core is a lean 2PL IRT fit (joint gradient descent, no scipy): each line
gets a difficulty `b` from its observed satisfaction base rate; each model gets
an ability `theta`. Fit lines are then ranked by information gain — Fisher
information at the current ability estimate — and greedily selected until the
bank's cumulative information plateaus, yielding a compact bank that spans the
observed capability range instead of the full rubric list.

Judge redundancy: the posterior is only informative with >=3 judges; with 2
judges (the harness's default panel) a unanimous pair is far weaker evidence of
degeneracy, so `min_judges` defaults to 3 and 2-judge panels fall back to the
raw pairwise-agreement path with the caveat flagged on the report.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from pydantic import BaseModel

__all__ = [
    "LineMeasurability",
    "MeasurabilityReport",
    "fit_item_response",
    "load_grades_jsonl",
    "select_rubric_bank",
    "write_rubric_calibration_csv",
]

# Beta(1,1) prior: uniform — no line is assumed measurable or degenerate
# before the judges have spoken.
_PRIOR_ALPHA = 1.0
_PRIOR_BETA = 1.0


class LineMeasurability(BaseModel):
    """Measurability posterior for a single rubric line."""

    rubric_index: int  # 0-based index of the line within the question's rubric
    n_observations: int  # (run, judge) grades contributing pairs
    n_agree: int
    n_disagree: int
    mean_agreement: float  # Posterior mean of p_agree
    mean_disagreement: float  # Posterior mean of p_disagree
    ci_lo_disagreement: float  # Lower bound of the 90% credible interval
    measurable: bool


class MeasurabilityReport(BaseModel):
    """Per-question measurability report across the judge panel."""

    question_id: str
    lines: list[LineMeasurability]
    n_judges: int
    sufficient_judges: bool  # False for 2-judge panels (see module docstring)

    @property
    def measurable_indices(self) -> list[int]:
        return [ln.rubric_index for ln in self.lines if ln.measurable]


def _beta_cdf(x: float, a: float, b: int | float) -> float:
    """Regularized incomplete beta function via the continued-fraction
    Lentz algorithm (Numerical Recipes 6.4). No scipy dependency — the
    analysis scripts are deliberately stdlib-only."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(x, a, b) / a
    return 1.0 - front * _beta_cf(1.0 - x, b, a) / b


def _beta_cf(x: float, a: float, b: int | float, itmax: int = 200, eps: float = 3e-12) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _beta_ppf(q: float, a: float, b: int | float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Beta quantile by bisection — monotone CDF, ~40 iterations is plenty."""
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _bernoulli_pairs(pairs_by_run: dict[tuple, dict[str, bool]]) -> tuple[int, int]:
    """Count agreeing / disagreeing judge pairs within the same trace.

    `pairs_by_run` maps a run key — `(model, trial_idx)` — to that run's
    `{judge: earned}` grades. Only judges grading the *same* trace form an
    agreement pair: two judges scoring different runs may "agree" simply
    because the runs were equally easy. With k judges on one run there are
    k(k-1)/2 pairs; each contributes one Bernoulli draw of "these two judges
    agree on this line".
    """
    agree = 0
    disagree = 0
    for judges in pairs_by_run.values():
        vals = list(judges.values())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if vals[i] == vals[j]:
                    agree += 1
                else:
                    disagree += 1
    return agree, disagree


def measurability_report(
    *,
    question_id: str,
    grades: list[dict],
    disagree_floor: float = 0.10,
    min_judges: int = 3,
) -> MeasurabilityReport:
    """Build the Beta-Bernoulli measurability report for one question.

    `grades` is a list of GradedRun-shaped dicts (as read from a
    `{label}.grades.{judge}.jsonl` file) for a single question_id, each
    carrying `rubric_lines`, `judge`, `model`, and `trial_idx`.

    `disagree_floor`: a line is non-measurable when the judges agree on it
    so consistently that the posterior mean of p_disagreement falls below
    this floor — the panel cannot distinguish models through that line, and
    the line contributes noise (a free point or a guaranteed miss) rather
    than signal. With the uniform Beta(1,1) prior, a 3-judge panel over 3
    trials (9 agreeing pairs) puts the posterior mean at ~0.09, just under
    the default 0.10 — the filter fires exactly when a line shows no
    disagreement across a full headline-sized panel. The filter
    additionally requires `n_judges >= min_judges`: with a 2-judge panel
    (the harness default) unanimity is weak evidence of degeneracy, so such
    panels never filter and the report flags `sufficient_judges=False`.
    """
    n_judges = len({g.get("judge") for g in grades})
    per_line: dict[int, dict[tuple, dict[str, bool]]] = {}
    for g in grades:
        run_key = (g.get("model"), g.get("trial_idx", 0))
        for idx, line in enumerate(g.get("rubric_lines") or []):
            per_line.setdefault(idx, {}).setdefault(run_key, {})[g.get("judge")] = bool(
                line.get("earned")
            )

    lines: list[LineMeasurability] = []
    for idx in sorted(per_line):
        pairs_by_run = per_line[idx]
        n_obs = sum(len(j) for j in pairs_by_run.values())
        agree, disagree = _bernoulli_pairs(pairs_by_run)
        n_pairs = agree + disagree
        if n_pairs == 0:
            # Single judge on this line: no pair evidence, keep the line.
            lines.append(
                LineMeasurability(
                    rubric_index=idx,
                    n_observations=n_obs,
                    n_agree=0,
                    n_disagree=0,
                    mean_agreement=1.0,
                    mean_disagreement=0.0,
                    ci_lo_disagreement=0.0,
                    measurable=True,
                )
            )
            continue
        alpha = _PRIOR_ALPHA + agree
        beta = _PRIOR_BETA + disagree
        mean_agree = alpha / (alpha + beta)
        mean_disagree = beta / (alpha + beta)
        # 90% credible interval on p_disagreement; report the lower bound as
        # the optimistic (most-measurable) reading of the panel's noise.
        ci_lo = _beta_ppf(0.05, alpha, beta)
        degenerate = disagree == 0 and n_judges >= min_judges and mean_disagree < disagree_floor
        lines.append(
            LineMeasurability(
                rubric_index=idx,
                n_observations=n_obs,
                n_agree=agree,
                n_disagree=disagree,
                mean_agreement=round(mean_agree, 4),
                mean_disagreement=round(mean_disagree, 4),
                ci_lo_disagreement=round(ci_lo, 6),
                measurable=not degenerate,
            )
        )
    return MeasurabilityReport(
        question_id=question_id,
        lines=lines,
        n_judges=n_judges,
        sufficient_judges=n_judges >= min_judges,
    )


def fit_item_response(
    matrix: list[list[int | None]],
    *,
    n_iter: int = 200,
    lr: float = 0.1,
) -> tuple[list[float], list[float]]:
    """Fit a 2-parameter-logistic IRT model by joint gradient descent.

    `matrix` is (models x lines) with 1=satisfied, 0=not, None=ungraded.
    Returns `(thetas, difficulties)`: one ability per model, one difficulty
    per rubric line. Discrimination is fixed at 1.0 — with the benchmark's
    modest per-line sample sizes a free `a` overfits, and the paper's
    selection step only needs relative information, which the fixed-`a`
    Fisher information preserves.
    """
    n_models = len(matrix)
    n_lines = len(matrix[0]) if matrix else 0
    # Initialize difficulty from each line's base rate: a line most models
    # satisfy is easy (b<0); a line few satisfy is hard (b>0).
    difficulties: list[float] = []
    for j in range(n_lines):
        vals = [matrix[i][j] for i in range(n_models) if matrix[i][j] is not None]
        p = sum(vals) / len(vals) if vals else 0.5
        p = min(max(p, 0.05), 0.95)
        difficulties.append(math.log((1 - p) / p))
    thetas = [0.0] * n_models
    for _ in range(n_iter):
        grad_t = [0.0] * n_models
        grad_b = [0.0] * n_lines
        for i in range(n_models):
            for j in range(n_lines):
                y = matrix[i][j]
                if y is None:
                    continue
                z = thetas[i] - difficulties[j]
                p = 1.0 / (1.0 + math.exp(-z))
                err = p - y
                grad_t[i] += err
                grad_b[j] -= err
        for i in range(n_models):
            thetas[i] -= lr * grad_t[i]
        for j in range(n_lines):
            difficulties[j] -= lr * grad_b[j]
            difficulties[j] = max(-4.0, min(4.0, difficulties[j]))
    return thetas, difficulties


def select_rubric_bank(
    *,
    matrix: list[list[int | None]],
    n_lines: int,
    target_coverage: float = 0.95,
) -> list[int]:
    """Greedily select rubric-line indices that cover the ability range.

    Two filters, in the paper's order:

    1. *Variance filter* — a line every observed model passes (or every
       model fails) cannot separate systems and is dropped outright.
    2. *Submodular coverage* — at fixed discrimination, the paper's
       information-coverage objective reduces to iteratively picking the
       line that most increases total Fisher information over the observed
       abilities, where each line contributes `I(theta, b) = p(1-p)` with
       `p = sigmoid(theta - b)`. Total information is monotone and
       submodular in the selected set (diminishing returns once a line's
       difficulty neighborhood is already covered), so greedy selection is
       within (1 - 1/e) of the optimal bank. Selection stops once the bank
       reaches `target_coverage` of the information the full line set
       provides — the compactness that let CalibratedRubric reach target
       correlation with 49 of 131 rubrics. Lines duplicating an already-
       covered difficulty add near-zero information and are dropped.
    """
    thetas, difficulties = fit_item_response(matrix)
    if not thetas or n_lines == 0:
        return []
    # 1. Drop lines with no observed variance across models.
    informative: list[int] = []
    for j in range(n_lines):
        vals = [matrix[i][j] for i in range(len(matrix)) if matrix[i][j] is not None]
        if vals and not (all(vals) or not any(vals)):
            informative.append(j)
    if not informative:
        # Every line is degenerate — nothing to select on; keep the full set
        # rather than silently dropping everything.
        return list(range(n_lines))
    if len(informative) == 1:
        return informative
    # 2. Greedy submodular selection over the informative lines.
    full_info = sum(_info(t, difficulties[j]) for t in thetas for j in informative)
    threshold = target_coverage * full_info
    selected: list[int] = []
    remaining = set(informative)
    while remaining:
        covered = sum(_info(t, difficulties[j]) for t in thetas for j in selected)
        if covered >= threshold:
            break
        next_line = max(
            remaining,
            key=lambda j: sum(_info(t, difficulties[j]) for t in thetas),
        )
        selected.append(next_line)
        remaining.discard(next_line)
    return sorted(selected)


def _info(theta: float, b: float) -> float:
    p = 1.0 / (1.0 + math.exp(-(theta - b)))
    return p * (1 - p)


def load_grades_jsonl(paths: list[Path]) -> list[dict]:
    """Read GradedRun-shaped dicts from one or more grades JSONL files."""
    rows: list[dict] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_rubric_calibration_csv(
    *,
    grades: list[dict],
    out_path: Path,
    disagree_floor: float = 0.05,
    min_judges: int = 3,
) -> list[MeasurabilityReport]:
    """Emit `rubric_calibration.csv`: one row per (question, rubric line).

    Each row carries the Beta-Bernoulli agreement posterior, the measurable
    flag, and — when the question's grades span >=2 models — whether the
    line made it into the greedy compact bank.
    """
    by_q: dict[str, list[dict]] = {}
    for g in grades:
        by_q.setdefault(g["question_id"], []).append(g)

    reports = [
        measurability_report(
            question_id=qid, grades=group, disagree_floor=disagree_floor, min_judges=min_judges
        )
        for qid, group in sorted(by_q.items())
    ]

    # Bank selection needs a models x lines matrix per question; a question
    # graded by only one model has no range to cover, so its bank is the
    # full measurable set.
    bank_by_q: dict[str, set[int]] = {}
    for qid, group in by_q.items():
        models = sorted({g.get("model") for g in group})
        n_lines = len(group[0].get("rubric_lines") or [])
        if len(models) < 2 or n_lines == 0:
            report = next(r for r in reports if r.question_id == qid)
            bank_by_q[qid] = set(report.measurable_indices)
            continue
        # Model score = fraction of lines satisfied, per judge, averaged.
        matrix: list[list[int | None]] = []
        for m in models:
            per_line_hits: list[list[int]] = [[] for _ in range(n_lines)]
            for g in group:
                if g.get("model") != m:
                    continue
                for idx, line in enumerate(g.get("rubric_lines") or []):
                    per_line_hits[idx].append(1 if line.get("earned") else 0)
            row = [round(sum(h) / len(h)) if h else None for h in per_line_hits]
            matrix.append([v if v is None else int(v) for v in row])
        bank_by_q[qid] = set(select_rubric_bank(matrix=matrix, n_lines=n_lines))

    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "qid",
                "rubric_index",
                "n_grades",
                "n_agree",
                "n_disagree",
                "mean_agreement",
                "ci_lo_disagreement",
                "measurable",
                "in_bank",
                "n_judges",
                "sufficient_judges",
            ]
        )
        for r in reports:
            bank = bank_by_q.get(r.question_id, set())
            for ln in r.lines:
                w.writerow(
                    [
                        r.question_id,
                        ln.rubric_index,
                        ln.n_observations,
                        ln.n_agree,
                        ln.n_disagree,
                        ln.mean_agreement,
                        ln.ci_lo_disagreement,
                        ln.measurable,
                        ln.rubric_index in bank,
                        r.n_judges,
                        r.sufficient_judges,
                    ]
                )
    return reports
