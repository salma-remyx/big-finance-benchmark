"""Tests for rubric measurability filtering and bank assembly.

The integration test goes through the existing grading entry point
(`big_finance_harness.grader.grade`) with a measurability-filtered rubric
subset, verifying the judge sees and scores only the selected lines. The
unit tests cover the Beta-Bernoulli posterior and the greedy bank selector
directly.
"""

from __future__ import annotations

import csv
import json

import pytest

from big_finance_harness import grader as grader_module
from big_finance_harness.grader import grade
from big_finance_harness.rubric_calibration import (
    fit_item_response,
    measurability_report,
    select_rubric_bank,
    write_rubric_calibration_csv,
)
from big_finance_harness.types import (
    DatasetItem,
    RubricLine,
    RunRecord,
    StepRecord,
)


def _make_item() -> DatasetItem:
    return DatasetItem(
        id="bf-cal-001",
        query="What was Apple's FY2023 operating income?",
        reference_answer="$114.3 billion",
        rubric=[
            RubricLine(text="Identifies AAPL as ticker", points=1),
            RubricLine(text="Locates FY2023 10-K", points=2),
            RubricLine(text="Reports operating income of $114.3 billion", points=5),
        ],
    )


def _make_run() -> RunRecord:
    return RunRecord(
        question_id="bf-cal-001",
        question="What was Apple's FY2023 operating income?",
        reference_answer="$114.3 billion",
        model="anthropic:claude-opus-4-7",
        harness_version="0.1.0",
        thinking="off",
        temperature=None,
        max_steps=30,
        steps=[
            StepRecord(
                step=0,
                assistant_text="Looking up Apple's FY2023 10-K.",
                tool_calls=[],
                tool_results=[],
                prompt_tokens=100,
                completion_tokens=20,
                wallclock_seconds=1.0,
            )
        ],
        final_answer="$114.3 billion",
        stop_reason="final_answer",
        total_prompt_tokens=100,
        total_completion_tokens=20,
        total_wallclock_seconds=1.0,
        started_at="2026-04-30T00:00:00+00:00",
        completed_at="2026-04-30T00:00:01+00:00",
    )


class _FakeResponse:
    def __init__(self, content: str):
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 10})()
        self._hidden_params = {"response_cost": 0.001}


@pytest.mark.asyncio
async def test_grade_with_rubric_subset_scores_only_selected_lines(monkeypatch):
    """The measurability-filtered rubric flows through grade(): the judge
    receives the 2-line bank, and points/line counts aggregate over the
    subset, not the full 3-line rubric."""
    judge_payload = {
        "final_answer_correct": True,
        "rubric": [
            {"index": 1, "satisfied": True, "explanation": "Reported $114.3B"},
            {"index": 2, "satisfied": True, "explanation": "Located the 10-K"},
        ],
    }
    seen_prompts: list[str] = []

    async def fake_acompletion(**kwargs):
        seen_prompts.append(kwargs["messages"][1]["content"])
        return _FakeResponse(json.dumps(judge_payload))

    monkeypatch.setattr(grader_module.litellm, "acompletion", fake_acompletion)

    graded = await grade(
        run=_make_run(),
        item=_make_item(),
        judge_model_id="vertex:gemini-3.1-pro-preview",
        rubric_indices=[1, 2],  # drop the degenerate line 0
    )

    # The judge saw only the two selected rubric lines.
    assert "Identifies AAPL as ticker" not in seen_prompts[0]
    assert "Locates FY2023 10-K" in seen_prompts[0]
    # Aggregation is over the subset: both lines earned → 2 of 2 lines, 7 of 7 points.
    assert graded.rubric_lines_earned == 2
    assert graded.rubric_lines_possible == 2
    assert graded.rubric_points_earned == 7
    assert graded.rubric_points_possible == 7
    # The filtered-out line does not appear in the graded output.
    assert [ln.text for ln in graded.rubric_lines] == [
        "Locates FY2023 10-K",
        "Reports operating income of $114.3 billion",
    ]


def _grade_dict(judge: str, model: str, earned: list[bool], trial_idx: int = 0) -> dict:
    return {
        "question_id": "bf-cal-001",
        "judge": judge,
        "model": model,
        "trial_idx": trial_idx,
        "rubric_lines": [
            {"text": f"line {i}", "points": 1, "earned": e} for i, e in enumerate(earned)
        ],
    }


def test_unanimous_line_on_three_judge_panel_is_not_measurable():
    """A line all three judges score identically across trials, with zero
    observed disagreement, has posterior mass near p_disagree=0 → filtered
    out. A line with observed disagreement stays measurable."""
    grades = []
    for trial in range(3):
        grades.append(_grade_dict("gemini", "m1", [True, True, False, True], trial))
        grades.append(_grade_dict("opus", "m1", [True, True, True, True], trial))
        grades.append(_grade_dict("gpt", "m1", [True, True, True, True], trial))
    report = measurability_report(question_id="bf-cal-001", grades=grades)
    # Line 0: 3 judges × 3 trials → 9 agreeing pairs, 0 disagreeing.
    line0 = report.lines[0]
    assert line0.n_agree == 9
    assert line0.n_disagree == 0
    assert line0.mean_disagreement < 0.10
    assert line0.measurable is False
    # Line 2: disagreement observed → measurable.
    assert report.lines[2].n_disagree > 0
    assert report.lines[2].measurable is True
    assert report.sufficient_judges is True


def test_two_judge_panel_keeps_unanimous_lines():
    """With 2 judges (the harness default), unanimity is weak evidence —
    the line is kept and the report flags insufficient judge redundancy."""
    grades = [
        _grade_dict("gemini", "m1", [True, True], 0),
        _grade_dict("opus", "m1", [True, True], 0),
        _grade_dict("gemini", "m1", [True, True], 1),
        _grade_dict("opus", "m1", [True, True], 1),
    ]
    report = measurability_report(question_id="bf-cal-001", grades=grades)
    assert report.lines[0].n_disagree == 0
    assert report.lines[0].measurable is True
    assert report.sufficient_judges is False


def test_bank_selection_drops_degenerate_lines():
    """A line passed (or failed) by every model carries no information at
    any ability → dropped from the compact bank; discriminative lines are
    kept and span the range."""
    # Lines 0 (all pass) and 2 (all fail) are degenerate; lines 1 and 3
    # split the model set differently.
    matrix = [
        [1, 1, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    bank = select_rubric_bank(matrix=matrix, n_lines=4)
    assert 0 not in bank  # every model passes line 0 → no information
    assert 2 not in bank  # every model fails line 2 → no information
    assert 1 in bank and 3 in bank


def test_irt_fit_ranks_models_by_ability():
    thetas, difficulties = fit_item_response(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    # Model 0 satisfies the most lines → highest ability.
    assert thetas[0] > thetas[2] > thetas[1]
    # Line 3 is failed by every model → hardest.
    assert difficulties[3] == max(difficulties)


def test_write_rubric_calibration_csv_roundtrip(tmp_path):
    grades = [
        _grade_dict("gemini", "m1", [True, False, True]),
        _grade_dict("opus", "m1", [True, False, False]),
        _grade_dict("gemini", "m2", [True, True, True]),
        _grade_dict("opus", "m2", [True, False, True]),
    ]
    out = tmp_path / "rubric_calibration.csv"
    reports = write_rubric_calibration_csv(grades=grades, out_path=out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 3
    assert {r["qid"] for r in rows} == {"bf-cal-001"}
    assert all(r["n_judges"] == "2" for r in rows)
    assert rows[0]["measurable"] == "True"  # 2-judge panel: kept with caveat
    assert len(reports) == 1
