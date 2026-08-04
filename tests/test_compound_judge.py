"""Tests for the compound (majority-vote) judge path.

The integration tests drive ``grade()`` -- the existing grading entry point in
``big_finance_harness.grader`` -- with ``compound_samples>1``, stubbing
``litellm.acompletion`` to return a scripted sequence of disagreeing judge verdicts,
and assert that the wired-in majority-vote path produces the aggregated verdict while
summing per-call token/cost accounting. The unit tests pin the aggregator's
strict-majority + tie semantics directly.
"""

from __future__ import annotations

import json

import pytest

from big_finance_harness import grader as grader_module
from big_finance_harness.compound_judge import aggregate_judge_samples
from big_finance_harness.grader import grade
from big_finance_harness.types import (
    DatasetItem,
    RubricLine,
    RunRecord,
    StepRecord,
    ToolResultBlock,
    ToolUseBlock,
)


def _make_run() -> RunRecord:
    return RunRecord(
        question_id="bf-test-001",
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
        final_answer="$114.3 billion",
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
    def __init__(self, payload: dict, prompt_tokens: int, completion_tokens: int, cost: float):
        self.choices = [_FakeChoice(json.dumps(payload))]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)
        self._hidden_params = {"response_cost": cost}


@pytest.mark.asyncio
async def test_grade_compound_samples_majority_votes_and_sums_accounting(monkeypatch):
    """Three disagreeing verdicts reduce to the majority verdict; costs sum x3.

    Votes across the three samples:
      final_answer_correct: True, False, True -> True
      rubric 1 (ticker):    True,  True,  True  -> True
      rubric 2 (10-K):      False, True, False -> False (1 of 3)
      rubric 3 (income):    True,  True, False -> True  (2 of 3)
    So 2 of 3 rubric lines are earned (1 + 5 = 6 of 8 points).
    """
    payloads = [
        {
            "final_answer_correct": True,
            "rubric": [
                {"index": 1, "satisfied": True, "explanation": "AAPL cited"},
                {"index": 2, "satisfied": False, "explanation": "no 10-K"},
                {"index": 3, "satisfied": True, "explanation": "$114.3B reported"},
            ],
        },
        {
            "final_answer_correct": False,
            "rubric": [
                {"index": 1, "satisfied": True, "explanation": "ticker ok"},
                {"index": 2, "satisfied": True, "explanation": "located 10-K"},
                {"index": 3, "satisfied": True, "explanation": "income matches"},
            ],
        },
        {
            "final_answer_correct": True,
            "rubric": [
                {"index": 1, "satisfied": True, "explanation": "AAPL"},
                {"index": 2, "satisfied": False, "explanation": "missed filing"},
                {"index": 3, "satisfied": False, "explanation": "wrong number"},
            ],
        },
    ]
    responses = [
        _FakeResponse(p, prompt_tokens=1000, completion_tokens=50, cost=0.005) for p in payloads
    ]
    calls = {"n": 0}

    async def fake_acompletion(**_kwargs):
        i = calls["n"]
        calls["n"] += 1
        return responses[i]

    monkeypatch.setattr(grader_module.litellm, "acompletion", fake_acompletion)

    graded = await grade(
        run=_make_run(),
        item=_make_item(),
        judge_model_id="vertex:gemini-3.1-pro-preview",
        compound_samples=3,
    )

    assert calls["n"] == 3
    # Majority-voted verdict.
    assert graded.final_answer_correct is True
    assert [line.earned for line in graded.rubric_lines] == [True, False, True]
    assert graded.rubric_lines_earned == 2
    assert graded.rubric_points_earned == 6
    assert graded.rubric_points_possible == 8
    # Accounting summed across the three judge calls.
    assert graded.judge_prompt_tokens == 3000
    assert graded.judge_completion_tokens == 150
    assert graded.judge_cost_usd == pytest.approx(0.015)
    # Explanation carried from a sample on the winning (True) side of line 3.
    assert graded.rubric_lines[2].judge_explanation in ("$114.3B reported", "income matches")


@pytest.mark.asyncio
async def test_grade_default_compound_samples_is_a_single_call(monkeypatch):
    """compound_samples defaults to 1: exactly one judge call, no aggregation."""

    payload = {
        "final_answer_correct": True,
        "rubric": [
            {"index": 1, "satisfied": True, "explanation": "ok"},
            {"index": 2, "satisfied": False, "explanation": "no"},
            {"index": 3, "satisfied": True, "explanation": "ok"},
        ],
    }
    fake = _FakeResponse(payload, prompt_tokens=2500, completion_tokens=180, cost=0.012)
    calls = {"n": 0}

    async def fake_acompletion(**_kwargs):
        calls["n"] += 1
        return fake

    monkeypatch.setattr(grader_module.litellm, "acompletion", fake_acompletion)

    graded = await grade(
        run=_make_run(),
        item=_make_item(),
        judge_model_id="vertex:gemini-3.1-pro-preview",
    )
    assert calls["n"] == 1
    assert graded.judge_prompt_tokens == 2500
    assert graded.judge_cost_usd == pytest.approx(0.012)


def test_aggregate_judge_samples_strict_majority_and_tie_breaks_conservatively():
    samples = [
        {
            "final_answer_correct": True,
            "rubric": [
                {"index": 1, "satisfied": True, "explanation": "y"},
                {"index": 2, "satisfied": True, "explanation": "y"},
            ],
        },
        {
            "final_answer_correct": False,
            "rubric": [
                {"index": 1, "satisfied": True, "explanation": "y"},
                {"index": 2, "satisfied": False, "explanation": "n"},
            ],
        },
    ]
    agg = aggregate_judge_samples(samples)
    # final: 1 of 2 -> tie -> False (conservative).
    assert agg["final_answer_correct"] is False
    by_index = {entry["index"]: entry for entry in agg["rubric"]}
    # line 1: 2 of 2 True -> True; line 2: 1 of 2 -> tie -> False.
    assert by_index[1]["satisfied"] is True
    assert by_index[2]["satisfied"] is False
    # Explanation taken from a sample on the winning side.
    assert by_index[1]["explanation"] == "y"


def test_aggregate_judge_samples_passes_single_sample_through():
    single = {
        "final_answer_correct": True,
        "rubric": [{"index": 1, "satisfied": True, "explanation": "x"}],
    }
    assert aggregate_judge_samples([single]) is single
