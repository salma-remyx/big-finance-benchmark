"""Tests for the cross-trial final-answer consistency metric.

Builds real ``GradedRun`` records (the schema the grader emits) across the
harness's ``trial_idx`` axis, serializes them to a grades JSONL the way the run
pipeline does, then reads them back through ``read_grade_records`` and checks
the measurement -- exercising the integration end to end against the existing
``big_finance_harness.types`` data model rather than self-testing the new
module in isolation.
"""

from __future__ import annotations

from pathlib import Path

from big_finance_harness.trial_consistency import (
    compute_consistency,
    normalize_final_answer,
    read_grade_records,
)
from big_finance_harness.types import GradedRubricLine, GradedRun


def _graded(
    qid: str, trial_idx: int, final_answer: str, correct: bool, judge: str = "gemini"
) -> GradedRun:
    """A minimal ``GradedRun`` carrying only what the consistency metric reads."""
    return GradedRun(
        question_id=qid,
        trial_idx=trial_idx,
        model="vertex:gemini-3.1-pro-preview",
        judge=judge,
        final_answer=final_answer,
        reference_answer="$114.3 billion",
        final_answer_correct=correct,
        rubric_lines=[GradedRubricLine(text="x", points=1, earned=correct)],
        rubric_points_earned=1 if correct else 0,
        rubric_points_possible=1,
        rubric_lines_earned=1 if correct else 0,
        rubric_lines_possible=1,
    )


def test_normalize_final_answer_collapses_surface_variation():
    # Currency symbol, casing, trailing period, and thousands separators
    # should not break equality.
    assert normalize_final_answer("$114.3 Billion.") == "114.3 billion"
    assert normalize_final_answer("114.3 billion") == "114.3 billion"
    assert normalize_final_answer("  $1,234  ") == "1234"
    assert normalize_final_answer("$1,234,567") == "1234567"
    assert normalize_final_answer(None) == ""
    assert normalize_final_answer("") == ""


def test_read_and_compute_consistency_over_graded_run_jsonl(tmp_path: Path):
    # One model label, three questions. bf-1's trials diverge; bf-2's agree;
    # bf-3 has a single trial and must drop out of the rate denominator.
    runs = [
        _graded("bf-1", 0, "$114.3 billion", True),
        _graded("bf-1", 1, "$114.3 Billion.", True),
        _graded("bf-1", 2, "$118.0 billion", False),
        _graded("bf-2", 0, "42%", True),
        _graded("bf-2", 1, "42%", True),
        _graded("bf-2", 2, "42%", True),
        _graded("bf-3", 0, "7", True),
    ]
    grades_path = tmp_path / "gemini.grades.jsonl"
    grades_path.write_text("\n".join(r.model_dump_json() for r in runs) + "\n", encoding="utf-8")

    records = read_grade_records(tmp_path)
    # bf-3's single trial is loaded, just excluded from the consistency rate.
    assert len(records) == 7
    assert {r.model_label for r in records} == {"gemini"}

    summary = compute_consistency(records)
    assert summary.n_groups == 2  # bf-3 (1 trial) excluded

    by_qid = {g.question_id: g for g in summary.per_group}

    # bf-1: two trials normalize to "114.3 billion", one to "118.0 billion"
    # -> text-inconsistent; correctness also inconsistent (True, True, False).
    bf1 = by_qid["bf-1"]
    assert bf1.n_trials == 3
    assert bf1.text_consistent is False
    assert bf1.correctness_consistent is False
    assert bf1.distinct_answers == 2

    # bf-2: three identical trials.
    bf2 = by_qid["bf-2"]
    assert bf2.text_consistent is True
    assert bf2.correctness_consistent is True
    assert bf2.distinct_answers == 1

    # One of two groups is text-inconsistent -> 0.5 headline rate.
    assert summary.n_text_inconsistent == 1
    assert summary.text_inconsistency_rate == 0.5
