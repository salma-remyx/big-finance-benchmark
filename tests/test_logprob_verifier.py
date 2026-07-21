"""Integration tests for the LLM-as-a-Verifier continuous-scoring pass.

The primary test exercises the wiring in `big_finance_harness.grader.grade` (a
non-new module): with `continuous_score=True`, the grader runs an extra logprob
verifier pass and attaches calibrated continuous scores to the `GradedRun`.
`litellm.acompletion` is mocked — no network — returning a structured JSON for the
boolean grade and a token-logprob distribution for the verifier.
"""

from __future__ import annotations

import json
import math

import pytest

from big_finance_harness import grader as grader_module
from big_finance_harness import logprob_verifier
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


class _FakeUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 100
        self.completion_tokens = 5


class _StructuredResponse:
    """Boolean judge response: OpenAI-shape structured JSON in message.content."""

    def __init__(self, payload: dict) -> None:
        choice = type(
            "Choice",
            (),
            {
                "message": type("Msg", (), {"content": json.dumps(payload)})(),
            },
        )()
        self.choices = [choice]
        self.usage = _FakeUsage()
        self._hidden_params = {"response_cost": 0.001}


class _LogprobToken:
    def __init__(self, token: str, logprob: float, top: list[dict]) -> None:
        self.token = token
        self.logprob = logprob
        self.top_logprobs = top


class _LogprobResponse:
    """Verifier response: a first-token logprob distribution, OpenAI chat shape."""

    def __init__(self, token: str, logprob: float, top: list[dict]) -> None:
        info = _LogprobToken(token, logprob, top)
        logprobs = type("Logprobs", (), {"content": [info]})()
        choice = type("Choice", (), {"logprobs": logprobs})()
        self.choices = [choice]
        self.usage = _FakeUsage()
        self._hidden_params = {"response_cost": 0.0005}


def _lp(top_tokens: list[tuple[str, float]]) -> list[dict]:
    return [{"token": t, "logprob": math.log(p)} for t, p in top_tokens]


@pytest.mark.asyncio
async def test_grade_with_continuous_score_attaches_calibrated_scores(monkeypatch):
    """`continuous_score=True` adds continuous scores on top of the boolean grade.

    Final-answer verification is yes-dominant on a graded scale (yes/partial/no at
    0.8/0.15/0.05) → E[score] = 0.8·1 + 0.15·0.5 + 0.05·0 = 0.875. Each rubric line
    is no-dominant (no/yes/partial at 0.9/0.05/0.05) → E[score] = 0.075. The boolean
    grade (all satisfied) is unaffected.
    """

    async def fake_acompletion(**kwargs):
        if kwargs.get("response_format") is not None:
            # Boolean judge call: structured JSON.
            return _StructuredResponse(
                {
                    "final_answer_correct": True,
                    "rubric": [
                        {"index": 1, "satisfied": True, "explanation": "ok"},
                        {"index": 2, "satisfied": True, "explanation": "ok"},
                        {"index": 3, "satisfied": True, "explanation": "ok"},
                    ],
                }
            )
        # Verifier call: route by the criterion in the user prompt so the assertion
        # is independent of concurrent-call ordering.
        user = kwargs["messages"][-1]["content"]
        if "rubric step" in user.lower():
            return _LogprobResponse(
                " no", math.log(0.9), _lp([(" no", 0.9), (" yes", 0.05), (" partial", 0.05)])
            )
        return _LogprobResponse(
            " yes", math.log(0.8), _lp([(" yes", 0.8), (" partial", 0.15), (" no", 0.05)])
        )

    monkeypatch.setattr(grader_module.litellm, "acompletion", fake_acompletion)

    graded = await grade(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="vertex:gemini-3.1-pro-preview",
        continuous_score=True,
    )

    # Boolean grade unchanged.
    assert graded.final_answer_correct is True
    assert graded.rubric_lines_earned == 3

    # Continuous verifier scores attached.
    assert graded.verifier_scores is not None
    assert graded.verifier_scores.judge_model == "vertex:gemini-3.1-pro-preview"
    assert (
        graded.verifier_scores.rubric_lines is not None
        and len(graded.verifier_scores.rubric_lines) == 3
    )

    # Headline continuous verdict is high; per-criterion scores are low and graded.
    assert graded.verifier_scores.final_answer.score == pytest.approx(0.875, abs=1e-9)
    assert graded.verifier_scores.final_answer.label == "final_answer"
    for line in graded.verifier_scores.rubric_lines:
        assert line.label.startswith("rubric:")
        assert line.score == pytest.approx(0.075, abs=1e-9)
        # Probabilities over the matched scoring tokens sum to 1.
        assert sum(m["probability"] for m in line.matched_tokens) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_grade_without_continuous_score_has_no_verifier_scores(monkeypatch):
    """The flag is opt-in: the default path leaves verifier_scores None."""

    async def fake_acompletion(**_kwargs):
        return _StructuredResponse(
            {
                "final_answer_correct": True,
                "rubric": [{"index": 1, "satisfied": True, "explanation": "ok"}],
            }
        )

    monkeypatch.setattr(grader_module.litellm, "acompletion", fake_acompletion)

    graded = await grade(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="vertex:gemini-3.1-pro-preview",
    )
    assert graded.verifier_scores is None


def test_expectation_is_none_when_no_scoring_token_matches():
    """If none of the emitted tokens are in the scoring vocabulary, score is None.

    Pure check of the core expectation helper (imports the new module directly).
    """
    candidates = [{"token": "maybe_so", "logprob": 0.0}, {"token": "unsure", "logprob": -1.0}]
    score, detail = logprob_verifier._expectation(
        candidates, logprob_verifier.DEFAULT_SCORING_TOKENS
    )
    assert score is None
    assert detail == []


def test_expectation_binary_yes_matches_single_token_calibrated():
    """A confident single 'yes' token maps cleanly to a ~1.0 continuous score.

    With yes at logprob -1e-3 and no at -20, p_yes ≈ 0.999999998 → E[score] ≈ 1.0.
    """
    score, _ = logprob_verifier._expectation(
        [{"token": " yes", "logprob": -1e-3}, {"token": " no", "logprob": -20.0}],
        logprob_verifier.DEFAULT_SCORING_TOKENS,
    )
    assert score == pytest.approx(1.0, abs=1e-6)
