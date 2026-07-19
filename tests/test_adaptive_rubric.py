"""Tests for the SedarEval-style adaptive rubric elaboration wired into the grader.

The integration is exercised through the existing ``grade()`` entry point (the call
site). The elaboration LLM call and the grading LLM call are both mocked, so no
provider is hit — the point is to verify the wiring: when ``adaptive_rubric=True`` the
judge prompt carries the structured rubric and grading still produces a correct
``GradedRun``.
"""

from __future__ import annotations

import json

import pytest

from big_finance_harness import grader as grader_module
from big_finance_harness.adaptive_rubric import (
    ElaboratedRubricLine,
    format_elaborated_rubric,
)
from big_finance_harness.grader import grade
from big_finance_harness.types import (
    DatasetItem,
    RubricLine,
    RunRecord,
    StepRecord,
    ToolResultBlock,
    ToolUseBlock,
)


def _make_run(question_id: str, final_answer: str | None = "$114.3 billion") -> RunRecord:
    return RunRecord(
        question_id=question_id,
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
                tool_calls=[ToolUseBlock(id="t1", name="edgar_search", input={"ticker": "AAPL"})],
                tool_results=[ToolResultBlock(tool_use_id="t1", content="...filings...")],
                prompt_tokens=100,
                completion_tokens=20,
                wallclock_seconds=1.0,
            )
        ],
        final_answer=final_answer,
        stop_reason="final_answer",
        total_prompt_tokens=100,
        total_completion_tokens=20,
        total_wallclock_seconds=1.0,
        started_at="2026-04-30T00:00:00+00:00",
        completed_at="2026-04-30T00:00:01+00:00",
    )


def _make_item() -> DatasetItem:
    return DatasetItem(
        id="bf-test-001",
        query="What was Apple's FY2023 operating income?",
        reference_answer="$114.3 billion",
        rubric=[
            RubricLine(text="Identifies AAPL as ticker", points=1),
            RubricLine(text="Locates FY2023 10-K", points=2),
            RubricLine(text="Reports operating income of $114.3 billion", points=5),
        ],
    )


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int, cost: float):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)
        self._hidden_params = {"response_cost": cost}


@pytest.mark.asyncio
async def test_grade_with_adaptive_rubric_passes_structured_rubric_to_judge(monkeypatch):
    """With adaptive_rubric=True the judge prompt carries the elaborated structure."""

    async def fake_elaborate(*, item, judge_model_id, **_kwargs):
        return [
            ElaboratedRubricLine(
                index=1,
                primary_criteria=["ticker is AAPL"],
                secondary_criteria=[],
                deduction_points=["uses a different ticker"],
            ),
            ElaboratedRubricLine(
                index=2,
                primary_criteria=["locates the FY2023 10-K filing"],
                secondary_criteria=["cites the filing URL"],
                deduction_points=["wrong fiscal year"],
            ),
            ElaboratedRubricLine(
                index=3,
                primary_criteria=["operating income equals $114.3 billion"],
                secondary_criteria=[],
                deduction_points=["wrong sign", "wrong units"],
            ),
        ]

    # `grade` binds `elaborate_rubric` into its own namespace, so patch it there.
    monkeypatch.setattr(grader_module, "elaborate_rubric", fake_elaborate)

    captured: dict[str, object] = {}
    judge_payload = {
        "final_answer_correct": True,
        "rubric": [
            {"index": 1, "satisfied": True, "explanation": "ok"},
            {"index": 2, "satisfied": True, "explanation": "ok"},
            {"index": 3, "satisfied": True, "explanation": "ok"},
        ],
    }
    fake_response = _FakeResponse(
        content=json.dumps(judge_payload),
        prompt_tokens=3000,
        completion_tokens=200,
        cost=0.02,
    )

    async def fake_acompletion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return fake_response

    monkeypatch.setattr(grader_module.litellm, "acompletion", fake_acompletion)

    graded = await grade(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="vertex:gemini-3.1-pro-preview",
        adaptive_rubric=True,
    )

    # Grading still produces correct results — the elaboration only restructures the
    # judge's view, the point aggregation is unchanged.
    assert graded.rubric_lines_earned == 3
    assert graded.rubric_lines_possible == 3
    assert graded.rubric_points_earned == 8
    assert graded.rubric_points_possible == 8
    # The judge prompt carried the SedarEval structured rubric.
    user_msg = captured["messages"][1]["content"]
    assert "primary (must hold): ticker is AAPL" in user_msg
    assert "secondary: cites the filing URL" in user_msg
    assert "mark unsatisfied if: wrong sign; wrong units" in user_msg


@pytest.mark.asyncio
async def test_grade_without_adaptive_rubric_keeps_plain_rubric(monkeypatch):
    """Default (flag off) behavior is unchanged: judge sees the plain rubric list."""

    captured: dict[str, object] = {}
    judge_payload = {
        "final_answer_correct": True,
        "rubric": [{"index": 1, "satisfied": True, "explanation": "ok"}],
    }
    fake_response = _FakeResponse(
        content=json.dumps(judge_payload), prompt_tokens=500, completion_tokens=10, cost=0.001
    )

    async def fake_acompletion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return fake_response

    monkeypatch.setattr(grader_module.litellm, "acompletion", fake_acompletion)

    await grade(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="vertex:gemini-3.1-pro-preview",
    )

    user_msg = captured["messages"][1]["content"]
    # No elaboration markers; the plain numbered rubric is shown verbatim.
    assert "primary (must hold)" not in user_msg
    assert "1. Identifies AAPL as ticker" in user_msg


def test_format_elaborated_rubric_falls_back_for_missing_lines():
    """A line the elaboration omits falls back to plain text — never dropped."""

    rubric = [
        RubricLine(text="Identifies AAPL as ticker", points=1),
        RubricLine(text="Locates FY2023 10-K", points=2),
        RubricLine(text="Reports operating income of $114.3 billion", points=5),
    ]
    elaborated = [
        ElaboratedRubricLine(
            index=1,
            primary_criteria=["ticker is AAPL"],
            secondary_criteria=[],
            deduction_points=["uses a different ticker"],
        ),
        # index 2 missing on purpose.
        ElaboratedRubricLine(
            index=3,
            primary_criteria=["operating income equals $114.3 billion"],
            secondary_criteria=["shows the calculation"],
            deduction_points=["wrong sign", "wrong units"],
        ),
    ]

    out = format_elaborated_rubric(rubric, elaborated)

    # Line 1 fully elaborated.
    assert "primary (must hold): ticker is AAPL" in out
    assert "mark unsatisfied if: uses a different ticker" in out
    # Line 2 (missing) falls back to the plain numbered line, no structured markers.
    assert "2. Locates FY2023 10-K" in out
    line2_block = out.split("2. Locates FY2023 10-K", 1)[1].split("3. ", 1)[0]
    assert "primary" not in line2_block
    assert "mark unsatisfied if" not in line2_block
    # Line 3 carries secondary + multiple deduction points joined by '; '.
    assert "secondary: shows the calculation" in out
    assert "mark unsatisfied if: wrong sign; wrong units" in out
