"""Tests for the CM-LRS "bankability" scorecard and its wiring into `grade()`.

We mock `litellm.acompletion` rather than calling a real judge — both `grade()` and
`score_cm_lrs()` turn a structured judge response into a typed result, and that
translation is what we verify here, independent of any provider.
"""

from __future__ import annotations

import json

import pytest

from big_finance_harness import grader as grader_module
from big_finance_harness.cm_lrs import (
    CM_LRS_DIMENSIONS,
    _aggregate_cm_lrs,
    score_cm_lrs,
)
from big_finance_harness.grader import grade
from big_finance_harness.types import (
    CmLrsDimensionScore,
    DatasetItem,
    RubricLine,
    RunRecord,
    StepRecord,
    ToolResultBlock,
    ToolUseBlock,
)

_DIM_KEYS = [d.key for d in CM_LRS_DIMENSIONS]


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


class _FakeACompletion:
    """Returns queued responses in call order, recording each call's kwargs."""

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _rubric_payload() -> dict:
    return {
        "final_answer_correct": True,
        "rubric": [
            {"index": 1, "satisfied": True, "explanation": "Trace mentions AAPL"},
            {"index": 2, "satisfied": True, "explanation": "Located the 10-K"},
            {"index": 3, "satisfied": True, "explanation": "Reported $114.3B"},
        ],
    }


def _cm_lrs_payload(scores: dict[str, int]) -> dict:
    """Build a CM-LRS judge payload covering the given dimension scores."""

    return {
        "dimensions": [
            {"key": key, "score": scores[key], "rationale": f"scored {scores[key]}"}
            for key in _DIM_KEYS
            if key in scores
        ]
    }


@pytest.mark.asyncio
async def test_grade_with_cm_lrs_runs_both_scoring_passes(monkeypatch):
    """grade(cm_lrs=True) runs the rubric pass then the CM-LRS pass and attaches it."""

    scores = {
        "factual_accuracy": 5,
        "evidence_traceability": 4,
        "numerical_consistency": 5,
        "workflow_completeness": 3,
        "source_discipline": 4,
        "decision_usefulness": 2,
        "reviewability": 4,
    }
    rubric_response = _FakeResponse(
        content=json.dumps(_rubric_payload()),
        prompt_tokens=2500,
        completion_tokens=180,
        cost=0.012,
    )
    cm_lrs_response = _FakeResponse(
        content=json.dumps(_cm_lrs_payload(scores)),
        prompt_tokens=1800,
        completion_tokens=240,
        cost=0.009,
    )
    fake = _FakeACompletion([rubric_response, cm_lrs_response])
    monkeypatch.setattr(grader_module.litellm, "acompletion", fake)

    graded = await grade(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="vertex:gemini-3.1-pro-preview",
        cm_lrs=True,
    )

    # The canonical rubric grade still works as before.
    assert graded.final_answer_correct is True
    assert graded.rubric_points_earned == 8
    # Two judge calls fired: rubric then CM-LRS.
    assert len(fake.calls) == 2
    assert fake.calls[0]["response_format"]["json_schema"]["name"] == "rubric_grading"
    assert fake.calls[1]["response_format"]["json_schema"]["name"] == "cm_lrs_scoring"
    # CM-LRS scorecard attached.
    assert graded.cm_lrs is not None
    assert [d.key for d in graded.cm_lrs.dimensions] == _DIM_KEYS
    assert {d.score for d in graded.cm_lrs.dimensions} == {2, 3, 4, 5}
    # Equal-weight aggregate = arithmetic mean of the seven 0-5 scores.
    assert graded.cm_lrs.aggregate == pytest.approx(sum(scores.values()) / 7, abs=0.001)
    assert graded.cm_lrs.judge == "vertex:gemini-3.1-pro-preview"
    assert graded.cm_lrs.judge_cost_usd == pytest.approx(0.009)
    assert graded.cm_lrs.judge_prompt_tokens == 1800


@pytest.mark.asyncio
async def test_grade_without_cm_lrs_leaves_field_none(monkeypatch):
    """Default grade() path does not fire a second call and leaves cm_lrs unset."""

    fake = _FakeACompletion([_FakeResponse(json.dumps(_rubric_payload()), 100, 10, 0.001)])
    monkeypatch.setattr(grader_module.litellm, "acompletion", fake)

    graded = await grade(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="vertex:gemini-3.1-pro-preview",
    )

    assert len(fake.calls) == 1  # rubric only
    assert graded.cm_lrs is None


@pytest.mark.asyncio
async def test_score_cm_lrs_applies_custom_weights_and_alias(monkeypatch):
    """score_cm_lrs honors a weight map (zero drops a dimension) and judge_alias."""

    scores = {k: 4 for k in _DIM_KEYS}  # all 4s
    response = _FakeResponse(json.dumps(_cm_lrs_payload(scores)), 500, 60, 0.004)
    fake = _FakeACompletion([response])
    monkeypatch.setattr(grader_module.litellm, "acompletion", fake)

    # Weight everything to 0 except decision_usefulness (1.0) → aggregate == its score.
    scored = await score_cm_lrs(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="anthropic:claude-opus-4-7",
        judge_alias="judge-A",
        weights={k: 0.0 for k in _DIM_KEYS} | {"decision_usefulness": 1.0},
    )

    assert scored.judge == "judge-A"
    by_key = {d.key: d.score for d in scored.dimensions}
    assert by_key["decision_usefulness"] == 4
    # Only decision_usefulness carries weight, so the aggregate is its score alone.
    assert scored.aggregate == pytest.approx(4.0, abs=0.001)
    # Echoed weighting reflects the zeros and the lone 1.0.
    assert scored.weights["factual_accuracy"] == 0.0
    assert scored.weights["decision_usefulness"] == 1.0


@pytest.mark.asyncio
async def test_score_cm_lrs_missing_dimension_scores_zero(monkeypatch):
    """A dimension the judge omits is scored 0 with rationale 'missing'."""

    partial = dict.fromkeys(_DIM_KEYS, 5)
    del partial["numerical_consistency"]  # judge omits this one
    response = _FakeResponse(json.dumps(_cm_lrs_payload(partial)), 500, 60, 0.004)
    monkeypatch.setattr(grader_module.litellm, "acompletion", _FakeACompletion([response]))

    scored = await score_cm_lrs(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="vertex:gemini-3.1-pro-preview",
    )

    dims = {d.key: d for d in scored.dimensions}
    assert dims["numerical_consistency"].score == 0
    assert dims["numerical_consistency"].rationale == "missing"
    # Six 5s and one 0 → mean 30/7.
    assert scored.aggregate == pytest.approx(30 / 7, abs=0.001)


def test_aggregate_clamps_and_means():
    """Pure unit test: equal weights give the mean; out-of-range scores are clamped."""

    scored = [CmLrsDimensionScore(key=k, name=k, score=3) for k in ("a", "b", "c")]
    agg, weights = _aggregate_cm_lrs(scored, None)
    assert agg == 3.0
    assert weights == {"a": 1.0, "b": 1.0, "c": 1.0}

    # Custom weights: a=3 (w2), b=6 clamped to 5 (w1), c dropped (w0).
    scored[1] = CmLrsDimensionScore(key="b", name="b", score=6)
    agg, _ = _aggregate_cm_lrs(scored, {"a": 2.0, "b": 1.0, "c": 0.0})
    # (3*2 + 5*1) / (2 + 1) = 11 / 3
    assert agg == pytest.approx(11 / 3, abs=0.001)
