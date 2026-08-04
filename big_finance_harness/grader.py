"""Rubric-based judge.

The grader takes the trace of a run, the dataset item the run targets, and a judge model
identifier. It returns a `GradedRun` with one binary score per rubric line plus a binary
final-answer-correctness score.

The judge is asked, in a single structured call per question, to:
  1. Decide whether the model's `final_answer` matches the reference answer.
  2. For each rubric line, decide whether the trace evidences that the line was satisfied.

The judge sees: the question, the reference answer, the rubric, the model's final answer,
and the assistant text + tool calls + tool results from the full trace. The judge does
NOT see the points associated with each line — points are aggregated client-side after
grading so the judge can't be biased toward heavy-weight items.

Inter-judge agreement: callers should grade with at least two non-evaluated judges and
report Cohen's kappa; this module grades with one judge per call. An opt-in compound
path (`grade(compound_samples=N)`) instead draws N independent verdicts per question
and reduces them by majority vote, trading extra judge-time compute for a more
reliable single-judge verdict.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import os

import litellm

from big_finance_harness.compound_judge import aggregate_judge_samples
from big_finance_harness.models.base import (
    _to_litellm_model,
    _vertex_location_for,
    parse_model_id,
)
from big_finance_harness.types import (
    DatasetItem,
    GradedRubricLine,
    GradedRun,
    RubricLine,
    RunRecord,
    StepRecord,
)

JUDGE_SYSTEM = """\
You are an impartial grader for a financial-research agent benchmark. You evaluate whether
the agent satisfied each step of an analyst rubric and whether its final answer matches
the reference answer. You are strict but fair: a rubric line is satisfied only if the
trace contains positive evidence for it.

Return only the JSON object specified by the response schema. Do not add commentary.
"""


_MAX_TRACE_CHARS = 150_000  # ~37k tokens cap on the trace passed to the judge.
_TOOL_RESULT_CAP = 4_000  # ~1k tokens per tool result.
_TOOL_ARGS_CAP = 1_500  # ~375 tokens per tool call's args.

# Per-judge concurrency caps. Without these caps, the orchestrator's
# `--grade-concurrency × N-models` quickly exceeds the judge's PT bucket. Vertex
# Anthropic PT 429s climb sharply above ~12 concurrent calls; Vertex Gemini
# tolerates 40+. Tune per provider when adding new judges.
_JUDGE_CAPS: dict[str, int] = {
    "vertex-anthropic": 12,
    "vertex": 40,
    "openai": 20,
    "anthropic": 10,
    "gateway": 30,
}
_JUDGE_SEMAPHORES: dict[str, asyncio.Semaphore] = {}


def _judge_semaphore(judge_model_id: str) -> asyncio.Semaphore:
    if judge_model_id not in _JUDGE_SEMAPHORES:
        provider = judge_model_id.split(":", 1)[0] if ":" in judge_model_id else "default"
        cap = _JUDGE_CAPS.get(provider, 10)
        _JUDGE_SEMAPHORES[judge_model_id] = asyncio.Semaphore(cap)
    return _JUDGE_SEMAPHORES[judge_model_id]


def _format_trace(steps: list[StepRecord]) -> str:
    """Render the trace as text for the judge, with per-element and total-size caps.

    For very long runs the unbounded trace can exceed the judge's context window.
    The per-element cap is set to cover the 95th percentile of observed
    tool_result sizes; the judge's `assistant_text` (which we never cap)
    typically already cites the relevant tool-result content.

    On global overflow we keep head + tail and drop the middle (the model's first few
    tool calls show how it approached the problem and the last few show the
    conclusion — both more informative for grading than the middle).
    """

    lines: list[str] = []
    for s in steps:
        lines.append(f"=== step {s.step} ===")
        if s.assistant_text:
            lines.append(f"assistant: {s.assistant_text}")
        for tc in s.tool_calls:
            args = json.dumps(tc.input, ensure_ascii=False)
            if len(args) > _TOOL_ARGS_CAP:
                args = args[:_TOOL_ARGS_CAP] + "..."
            lines.append(f"tool_call {tc.name}({args})")
        for tr in s.tool_results:
            content = tr.content
            if len(content) > _TOOL_RESULT_CAP:
                content = content[:_TOOL_RESULT_CAP] + "..."
            err = " [ERROR]" if tr.is_error else ""
            lines.append(f"tool_result{err}: {content}")
    text = "\n".join(lines)
    if len(text) <= _MAX_TRACE_CHARS:
        return text
    # Keep head + tail; drop middle. The model's first few tool calls show how it
    # approached the problem and the last few show the conclusion — both more
    # informative for grading than the middle.
    head = text[: _MAX_TRACE_CHARS // 2]
    tail = text[-_MAX_TRACE_CHARS // 2 :]
    return f"{head}\n\n... [trace truncated for length] ...\n\n{tail}"


def _format_rubric_for_judge(rubric: list[RubricLine]) -> str:
    return "\n".join(f"{i + 1}. {line.text}" for i, line in enumerate(rubric))


def _judge_user_prompt(
    question: str,
    reference_answer: str,
    rubric: list[RubricLine],
    final_answer: str | None,
    trace: str,
) -> str:
    return f"""\
QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RUBRIC (one line per analyst step; numbered):
{_format_rubric_for_judge(rubric)}

AGENT'S FINAL ANSWER:
{final_answer or "[no final answer was produced]"}

AGENT'S TRACE (assistant text, tool calls, tool results):
{trace}

For each rubric line, return a boolean indicating whether the trace and final answer
together evidence that the line was satisfied. Also return a boolean indicating whether
the final answer matches the reference answer (numerically equivalent values count as
matching; minor formatting differences are acceptable; sign and units must match).
"""


def _build_response_schema(num_rubric_lines: int) -> dict[str, Any]:
    """Response schema for the judge.

    NOTE: We intentionally don't use `minItems`/`maxItems` on the rubric array — Vertex
    Gemini's structured-output validator rejects schemas where (rubric items × array
    bounds) produce "too many states for serving" with `400 INVALID_ARGUMENT`. This
    fired on questions with 80+ rubric lines. The grader's `by_index.get(i+1, ...)`
    fallback handles short/long arrays gracefully, so strict bounds are unnecessary.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["final_answer_correct", "rubric"],
        "properties": {
            "final_answer_correct": {"type": "boolean"},
            "rubric": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "satisfied", "explanation"],
                    "properties": {
                        "index": {"type": "integer"},
                        "satisfied": {"type": "boolean"},
                        "explanation": {"type": "string"},
                    },
                },
            },
        },
    }


async def _run_judge_once(
    kwargs: dict[str, object],
    judge_model_id: str,
) -> tuple[dict[str, Any], int, int, float | None]:
    """Make one structured judge call, parse its JSON, and capture token/cost accounting.

    Handles the per-judge concurrency semaphore and the provider "temperature
    deprecated" retry. ``kwargs`` is copied so the retry's ``pop("temperature")`` can't
    race a concurrent sibling when the compound path fans out several calls at once.
    Returns ``(parsed_verdict, prompt_tokens, completion_tokens, cost_usd)`` so the
    single-call and compound paths share one accounting surface.
    """
    kwargs = dict(kwargs)
    sem = _judge_semaphore(judge_model_id)
    async with sem:
        try:
            response = await litellm.acompletion(**kwargs)
        except (litellm.BadRequestError, litellm.InternalServerError) as e:
            msg = str(e).lower()
            if "temperature" in msg and "deprecated" in msg:
                kwargs.pop("temperature", None)
                response = await litellm.acompletion(**kwargs)
            else:
                raise
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    hidden = getattr(response, "_hidden_params", {}) or {}
    cost = hidden.get("response_cost")
    cost_usd = float(cost) if cost is not None else None
    return parsed, prompt_tokens, completion_tokens, cost_usd


async def grade(
    *,
    run: RunRecord,
    item: DatasetItem,
    judge_model_id: str,
    max_output_tokens: int = 16384,
    judge_alias: str | None = None,
    compound_samples: int = 1,
) -> GradedRun:
    """Grade a run with the given judge.

    `judge_alias`: if provided, the stored `GradedRun.judge` field uses this string
    instead of `judge_model_id`. Useful when substituting a same-family model and
    wanting downstream analysis to treat the grades as a single judge bucket.

    `compound_samples`: when greater than 1, scale judge-time compute by running the
    structured judge `compound_samples` times and reducing the verdicts by majority
    vote (see `big_finance_harness.compound_judge`). Token and judge-cost accounting
    is summed across the calls. Defaults to 1, which reproduces the single-call judge
    exactly.
    """
    if run.question_id != item.id:
        raise ValueError(f"run/item id mismatch: run={run.question_id} item={item.id}")
    if compound_samples < 1:
        raise ValueError(f"compound_samples must be >= 1, got {compound_samples}")

    provider, snapshot = parse_model_id(judge_model_id)
    judge_model = _to_litellm_model(provider, snapshot)

    trace = _format_trace(run.steps)
    user_prompt = _judge_user_prompt(
        question=item.query,
        reference_answer=item.reference_answer,
        rubric=item.rubric,
        final_answer=run.final_answer,
        trace=trace,
    )
    schema = _build_response_schema(len(item.rubric))

    # LiteLLM normalizes structured output across providers via
    # response_format={type: json_schema}. With drop_params=True, providers that don't
    # support strict json_schema fall back to JSON-mode + post-validation.
    kwargs: dict[str, object] = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_output_tokens,
        "temperature": 0,
        # Retry rate limits and transient errors with LiteLLM's built-in exponential
        # backoff. 20 retries gives ~15-20 min cumulative wait under default backoff,
        # enough to ride out sustained quota pressure during a many-model parallel
        # grade phase.
        "num_retries": 20,
        # Per-call timeout (seconds). Bounded worst case for hung judge calls.
        "request_timeout": 1800,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "rubric_grading",
                "strict": True,
                "schema": schema,
            },
        },
    }
    # Mirror the model client's Vertex routing — judge models on vertex/vertex-anthropic
    # need explicit project + location passed per call. See
    # `big_finance_harness/models/base.py` for the dedicated-PT header semantics and
    # `VERTEX_DISABLE_DEDICATED` escape hatch.
    if provider in ("vertex", "vertex-anthropic"):
        project = os.environ.get("VERTEXAI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project:
            kwargs["vertex_project"] = project
        kwargs["vertex_location"] = _vertex_location_for(provider)
        if not os.environ.get("VERTEX_DISABLE_DEDICATED"):
            kwargs["extra_headers"] = {"X-Vertex-AI-LLM-Request-Type": "dedicated"}
    # Run the structured judge `compound_samples` times and reduce the independent
    # verdicts by majority vote (`big_finance_harness.compound_judge`). With the
    # default of one sample this collapses to a single call reduced to itself; with
    # N>1 it scales judge-time compute for a more reliable verdict. The aggregated
    # verdict has the same shape as a single judge response, so the translation below
    # is unchanged; token and cost accounting is summed across the calls.
    sample_results = await asyncio.gather(
        *[_run_judge_once(kwargs, judge_model_id) for _ in range(compound_samples)]
    )
    parsed = aggregate_judge_samples([verdict for verdict, _, _, _ in sample_results])
    judge_prompt_tokens = sum(prompt for _, prompt, _, _ in sample_results)
    judge_completion_tokens = sum(comp for _, _, comp, _ in sample_results)
    costs = [cost for _, _, _, cost in sample_results if cost is not None]
    judge_cost_usd = sum(costs) if costs else None

    final_correct: bool = bool(parsed.get("final_answer_correct", False))
    by_index = {entry["index"]: entry for entry in parsed.get("rubric", [])}

    graded: list[GradedRubricLine] = []
    points_earned = 0
    points_possible = 0
    lines_earned = 0
    for i, line in enumerate(item.rubric):
        entry = by_index.get(i + 1, {"satisfied": False, "explanation": "missing"})
        satisfied = bool(entry.get("satisfied", False))
        graded.append(
            GradedRubricLine(
                text=line.text,
                points=line.points,
                earned=satisfied,
                judge_explanation=entry.get("explanation"),
            )
        )
        points_possible += line.points
        if satisfied:
            points_earned += line.points
            lines_earned += 1

    return GradedRun(
        question_id=item.id,
        trial_idx=run.trial_idx,
        model=run.model,
        judge=judge_alias if judge_alias else judge_model_id,
        final_answer=run.final_answer,
        reference_answer=item.reference_answer,
        final_answer_correct=final_correct,
        rubric_lines=graded,
        rubric_points_earned=points_earned,
        rubric_points_possible=points_possible,
        rubric_lines_earned=lines_earned,
        rubric_lines_possible=len(item.rubric),
        judge_prompt_tokens=judge_prompt_tokens,
        judge_completion_tokens=judge_completion_tokens,
        judge_cost_usd=judge_cost_usd,
    )
