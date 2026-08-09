"""Alternative Annotator Test for inter-judge agreement.

Implements the statistical procedure from "The Alternative Annotator Test for
LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with
LLMs" (arXiv:2501.10970). Given paired binary labels from two annotators over
the same items, it returns Cohen's kappa, a one-sided lower confidence bound,
and a REPLACE / RETAIN non-inferiority decision: the alternative annotator may
stand in for the reference only if agreement is high enough that the lower bound
clears a chosen kappa threshold.

This closes the grader's stated-but-unimplemented inter-judge-agreement gap
(`big_finance_harness/grader.py`): "callers should grade with at least two
non-evaluated judges and report Cohen's kappa; this module grades with one judge
per call." See ``inter_judge_agreement`` in the grader for the wiring.

The core test (Cohen's kappa + a one-sided lower confidence bound + the
non-inferiority decision) is a direct port of the paper's method. Scoped out as
auxiliary analysis that is not needed to reach the decision on a paired sample:
the paper's a-priori sample-size / power planning tables and its Monte-Carlo
simulation sweeps.
"""

from __future__ import annotations

import random
from typing import Sequence

from pydantic import BaseModel

Label = int  # Binary label in {0, 1}.


class AgreementResult(BaseModel):
    """Outcome of the alternative-annotator test for one pair of annotators."""

    reference_judge: str
    alternative_judge: str
    n: int  # Number of paired labels.
    observed_agreement: float  # Raw fraction of labels that agree.
    cohen_kappa: float  # Chance-corrected agreement.
    kappa_lower: float  # One-sided lower confidence bound on kappa.
    confidence: float  # e.g. 0.95 -> a 95% one-sided bound.
    kappa_min: float  # Non-inferiority threshold the decision used.
    decision: str  # "REPLACE" or "RETAIN".
    positive_agreement: float  # Agreement on items the reference labeled 1.
    negative_agreement: float  # Agreement on items the reference labeled 0.
    n_positive: int  # Items the reference labeled 1.
    n_negative: int  # Items the reference labeled 0.
    n_bootstrap: int  # Resamples used for the confidence bound.


def _normalize(labels: Sequence[Label]) -> list[int]:
    out = [int(x) for x in labels]
    if any(v not in (0, 1) for v in out):
        raise ValueError("labels must be binary (0/1 or bool)")
    return out


def observed_agreement(reference: Sequence[Label], alternative: Sequence[Label]) -> float:
    """Raw (uncorrected) fraction of paired labels that agree."""
    a = _normalize(reference)
    b = _normalize(alternative)
    if len(a) != len(b):
        raise ValueError("label vectors must be the same length")
    if not a:
        raise ValueError("need at least one paired label")
    agree = sum(1 for x, y in zip(a, b) if x == y)
    return agree / len(a)


def cohen_kappa(reference: Sequence[Label], alternative: Sequence[Label]) -> float:
    """Cohen's kappa for two paired binary label vectors.

    Returns 1.0 for the degenerate perfect-agreement-on-a-constant-label case
    (where the chance-correction denominator would be 0).
    """
    a = _normalize(reference)
    b = _normalize(alternative)
    if len(a) != len(b):
        raise ValueError("label vectors must be the same length")
    n = len(a)
    if n == 0:
        raise ValueError("need at least one paired label")

    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    denom = 1 - p_e
    if denom == 0:
        # Both marginals constant: perfect agreement iff they agree everywhere.
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / denom


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile; ``q`` in [0, 100]."""
    if not sorted_values:
        raise ValueError("empty distribution")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (q / 100) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def alternative_annotator_test(
    reference: Sequence[Label],
    alternative: Sequence[Label],
    *,
    reference_judge: str = "reference",
    alternative_judge: str = "alternative",
    kappa_min: float = 0.8,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> AgreementResult:
    """Run the alternative-annotator non-inferiority test.

    The alternative annotator may REPLACE the reference iff the one-sided lower
    confidence bound on Cohen's kappa is >= ``kappa_min``. ``confidence`` is the
    coverage of the one-sided bound (e.g. 0.95). The lower bound comes from a
    seeded paired bootstrap, so the result is deterministic for fixed inputs.

    Raises ValueError if the vectors disagree in length, are empty, or the
    parameters are out of range.
    """
    a = _normalize(reference)
    b = _normalize(alternative)
    if len(a) != len(b):
        raise ValueError("label vectors must be the same length")
    if not a:
        raise ValueError("need at least one paired label")
    if not 0.0 <= confidence < 1.0:
        raise ValueError("confidence must be in [0, 1)")
    if not -1.0 <= kappa_min <= 1.0:
        raise ValueError("kappa_min must be in [-1, 1]")

    n = len(a)
    kappa = cohen_kappa(a, b)
    p_o = observed_agreement(a, b)

    # Paired bootstrap for the kappa distribution; resample (a_i, b_i) together
    # so chance-correction stays meaningful. Skipped for n == 1 (no resampling
    # freedom) -- the point estimate is the only honest bound.
    effective_bootstrap = n_bootstrap if n > 1 else 0
    rng = random.Random(seed)
    pairs = list(zip(a, b))
    idxs = list(range(n))
    samples: list[float] = []
    for _ in range(effective_bootstrap):
        resampled = [pairs[rng.choice(idxs)] for _ in range(n)]
        samples.append(cohen_kappa([p[0] for p in resampled], [p[1] for p in resampled]))
    if samples:
        samples.sort()
        kappa_lower = _percentile(samples, (1.0 - confidence) * 100)
    else:
        kappa_lower = kappa

    n_pos = sum(a)
    n_neg = n - n_pos
    pos_agree = sum(1 for x, y in zip(a, b) if x == 1 and y == 1) / n_pos if n_pos else 1.0
    neg_agree = sum(1 for x, y in zip(a, b) if x == 0 and y == 0) / n_neg if n_neg else 1.0

    return AgreementResult(
        reference_judge=reference_judge,
        alternative_judge=alternative_judge,
        n=n,
        observed_agreement=p_o,
        cohen_kappa=kappa,
        kappa_lower=kappa_lower,
        confidence=confidence,
        kappa_min=kappa_min,
        decision="REPLACE" if kappa_lower >= kappa_min else "RETAIN",
        positive_agreement=pos_agree,
        negative_agreement=neg_agree,
        n_positive=n_pos,
        n_negative=n_neg,
        n_bootstrap=effective_bootstrap,
    )
