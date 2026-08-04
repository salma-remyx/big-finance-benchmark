"""Compound (judge-time-compute) aggregation for the rubric grader.

Standard LLM-as-a-judge grading makes a single structured call per question, which
leaves the verdict exposed to any one call's error. This module realizes the core
idea of *Verdict: A Library for Scaling Judge-Time Compute* (arXiv:2502.18018) at the
narrow point where it helps this harness: rather than porting the paper's full
verification / debate / unit-composition library, it aggregates N independent
categorical (yes/no) judge verdicts -- final-answer correctness and each rubric line
-- by strict majority vote. That is the paper's ``MajorityVoteUnit`` over
``CategoricalJudgeUnit``s: extra inference-time (judge-time) compute traded for a more
reliable verdict.

The aggregator is deliberately decoupled from the judge call itself: ``grade()`` in
``big_finance_harness.grader`` runs the structured judge N times and feeds the parsed
outputs here. The returned dict has the same shape as a single judge response, so the
grader's downstream translation into a ``GradedRun`` is unchanged.
"""

from __future__ import annotations

from typing import Any

# A parsed structured judge response: final-answer correctness plus a list of
# per-rubric-line categorical verdicts.
JudgeVerdict = dict[str, Any]


def _strict_majority(votes: list[bool]) -> bool:
    """True only when strictly more than half of the cast votes are True.

    On a tie (e.g. an even sample count split evenly) this returns False -- a
    conservative "do not award the point" outcome, matching how the grader treats a
    missing or ambiguous rubric line.
    """
    if not votes:
        return False
    return sum(1 for v in votes if v) * 2 > len(votes)


def aggregate_judge_samples(samples: list[JudgeVerdict]) -> JudgeVerdict:
    """Reduce N independent judge verdicts to one canonical verdict by majority vote.

    Each sample is a structured judge response with ``final_answer_correct`` and a
    ``rubric`` list of ``{index, satisfied, explanation}``. For final-answer
    correctness and for each rubric index we take a strict majority across the
    samples that reported that index; a rubric line no sample reports is omitted, so
    the grader's existing "missing index" fallback marks it unsatisfied. The
    explanation carried for each line is taken from a sample on the winning side.

    Returns a dict with the same shape as a single judge response, so it drops in
    wherever one structured verdict was consumed.
    """
    if not samples:
        return {"final_answer_correct": False, "rubric": []}
    if len(samples) == 1:
        return samples[0]

    final_correct = _strict_majority(
        [bool(sample.get("final_answer_correct", False)) for sample in samples]
    )

    votes: dict[int, list[bool]] = {}
    yes_explanation: dict[int, str | None] = {}
    no_explanation: dict[int, str | None] = {}
    for sample in samples:
        for entry in sample.get("rubric", []):
            index = entry.get("index")
            if index is None:
                continue
            satisfied = bool(entry.get("satisfied", False))
            votes.setdefault(index, []).append(satisfied)
            explanation = entry.get("explanation")
            if satisfied and index not in yes_explanation:
                yes_explanation[index] = explanation
            elif not satisfied and index not in no_explanation:
                no_explanation[index] = explanation

    rubric = []
    for index in sorted(votes):
        satisfied = _strict_majority(votes[index])
        rubric.append(
            {
                "index": index,
                "satisfied": satisfied,
                "explanation": (yes_explanation if satisfied else no_explanation).get(index),
            }
        )

    return {"final_answer_correct": final_correct, "rubric": rubric}
