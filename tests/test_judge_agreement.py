"""Tests for the alternative-annotator inter-judge agreement test.

The grader's inter-judge-agreement gap ("callers should grade with at least two
non-evaluated judges and report Cohen's kappa; this module grades with one judge
per call") is closed by ``inter_judge_agreement`` in the grader, which pairs
binary rubric labels across judges and runs the test in ``judge_agreement``.

The first test goes through the grader (the wiring); the rest cover the
statistics directly.
"""

from __future__ import annotations

import pytest

from big_finance_harness.grader import inter_judge_agreement
from big_finance_harness.judge_agreement import (
    alternative_annotator_test,
    cohen_kappa,
    observed_agreement,
)
from big_finance_harness.types import GradedRubricLine, GradedRun


def _graded(question_id: str, judge: str, line_earned: bool, final_correct: bool) -> GradedRun:
    """A one-rubric-line graded run; the binary labels we pair across judges."""
    return GradedRun(
        question_id=question_id,
        model="anthropic:claude-opus-4-7",
        judge=judge,
        final_answer="$114.3 billion",
        reference_answer="$114.3 billion",
        final_answer_correct=final_correct,
        rubric_lines=[GradedRubricLine(text="locates the 10-K", points=1, earned=line_earned)],
        rubric_points_earned=1 if line_earned else 0,
        rubric_points_possible=1,
        rubric_lines_earned=1 if line_earned else 0,
        rubric_lines_possible=1,
    )


def _bench_inputs():
    """Two judges over four questions whose 8-length label vectors give kappa=0.75.

    Judge A labels: [1,1,1,1,0,0,0,0]; judge B labels: [1,1,1,0,0,0,0,0]
    (one disagreement on Q2's final-answer bit) -> 7/8 observed agreement,
    p_e=0.5, kappa=(0.875-0.5)/0.5=0.75.
    """
    judge_a = [
        _graded("q1", "judgeA", True, True),
        _graded("q2", "judgeA", True, True),
        _graded("q3", "judgeA", False, False),
        _graded("q4", "judgeA", False, False),
    ]
    judge_b = [
        _graded("q1", "judgeB", True, True),
        _graded("q2", "judgeB", True, False),  # disagrees with A on final answer
        _graded("q3", "judgeB", False, False),
        _graded("q4", "judgeB", False, False),
    ]
    return judge_a + judge_b


def test_inter_judge_agreement_pairs_grader_labels():
    """Wiring: grader pairs binary labels across judges and runs the test.

    kappa_min=0.9 (above the 0.75 point estimate) forces RETAIN.
    """
    results = inter_judge_agreement(_bench_inputs(), kappa_min=0.9)
    assert list(results) == ["judgeA|judgeB"]

    res = results["judgeA|judgeB"]
    assert res.reference_judge == "judgeA"
    assert res.alternative_judge == "judgeB"
    # One rubric line + one final-answer bit per question, 4 questions.
    assert res.n == 8
    assert res.observed_agreement == pytest.approx(7 / 8)
    assert res.cohen_kappa == pytest.approx(0.75)
    assert res.positive_agreement == pytest.approx(0.75)
    assert res.negative_agreement == pytest.approx(1.0)
    # kappa_min sits above the point estimate, so the lower bound cannot clear it.
    assert -1.0 <= res.kappa_lower <= 1.0
    assert res.decision == "RETAIN"


def test_inter_judge_agreement_replaces_when_judges_agree():
    """Two judges that agree perfectly -> kappa 1.0 -> REPLACE."""
    perfectly_agreeing = [
        _graded("q1", "judgeA", True, True),
        _graded("q2", "judgeA", False, False),
        _graded("q1", "judgeB", True, True),
        _graded("q2", "judgeB", False, False),
    ]
    res = inter_judge_agreement(perfectly_agreeing)["judgeA|judgeB"]
    assert res.cohen_kappa == pytest.approx(1.0)
    assert res.kappa_lower == pytest.approx(1.0)
    assert res.decision == "REPLACE"


def test_cohen_kappa_hand_computed_value():
    assert cohen_kappa([1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]) == pytest.approx(0.75)


def test_cohen_kappa_perfect_and_constant():
    # Identical non-trivial labels -> perfect agreement.
    assert cohen_kappa([1, 0, 1, 0, 1], [1, 0, 1, 0, 1]) == pytest.approx(1.0)
    # Both constant on the same label -> degenerate-but-perfect (1.0), not NaN.
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == pytest.approx(1.0)


def test_observed_agreement():
    assert observed_agreement([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert observed_agreement([1, 1, 1, 0], [1, 0, 0, 0]) == pytest.approx(0.5)


def test_alternative_annotator_test_validation():
    with pytest.raises(ValueError):
        alternative_annotator_test([], [])
    with pytest.raises(ValueError):
        alternative_annotator_test([1, 0], [1])
    with pytest.raises(ValueError):
        alternative_annotator_test([1, 0], [0, 1], confidence=1.5)


def test_decision_flips_with_threshold():
    """High agreement (kappa~0.9 over 40 items) clears a low bar but not a high one.

    Balanced marginals keep kappa well off the degenerate 0/1 endpoints so the
    seeded bootstrap lower bound lands solidly between 0 and the point estimate.
    """
    reference = [1] * 20 + [0] * 20
    alternative = [1] * 18 + [0] * 22  # 2 disagreements -> kappa = 0.9
    assert cohen_kappa(reference, alternative) == pytest.approx(0.9)
    retain = alternative_annotator_test(reference, alternative, kappa_min=0.95)
    replace = alternative_annotator_test(reference, alternative, kappa_min=0.0)
    assert retain.decision == "RETAIN"
    assert replace.decision == "REPLACE"
