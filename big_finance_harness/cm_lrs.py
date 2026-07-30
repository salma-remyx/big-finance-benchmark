"""Capital Markets LLM Reliability Score (CM-LRS) — "bankability" scorecard.

Adapted from CM-LRS (arXiv:2607.21340). The canonical `grader.grade()` scores an
agent run with a binary rubric (each analyst step satisfied or not) plus a binary
final-answer-correctness flag. CM-LRS adds a complementary signal: a 0-5 score per
reliability dimension, evaluated at the *workflow-output* layer rather than the
question-answer layer. Plausibility is cheap; bankability is the bar.

The seven dimensions, each scored 0-5 against a rubric anchored on signals a
reviewer in a regulated setting would use:

  1. factual_accuracy      — material claims correct vs. the reference answer / sources
  2. evidence_traceability — claims traceable to retrieved tool results (EDGAR / web)
  3. numerical_consistency — figures in the answer match the source / computation figures
  4. workflow_completeness — the analyst steps implied by the task were covered, not just fluency
  5. source_discipline     — correct, current, primary sources with clean citation hygiene
  6. decision_usefulness   — the output is actionable for a decision-maker
  7. reviewability         — a reviewer can follow the reasoning and reproduce the chain

The aggregate is a *tunable* weighted mean: a workflow that cares only about
extraction can up-weight factual/numerical dimensions and drop decision-usefulness.
This module scores with one judge per call; callers wanting inter-judge agreement
should run it across multiple non-evaluated judges and report agreement, as with
the base grader.

This is a near-clone of the judge pattern in `grader.grade()` — same structured
`litellm.acompletion` call, the same trace formatter, the same per-judge
concurrency semaphore and Vertex routing — differing only in prompt and schema.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import litellm

from big_finance_harness.grader import _format_trace, _judge_semaphore
from big_finance_harness.models.base import (
    _to_litellm_model,
    _vertex_location_for,
    parse_model_id,
)
from big_finance_harness.types import (
    CmLrsDimensionScore,
    CmLrsScore,
    DatasetItem,
    RunRecord,
)

CM_LRS_JUDGE_SYSTEM = """\
You are an impartial reviewer scoring a capital-markets research output for \
"bankability" — whether it could be defended in front of a counter-party or a \
regulator, with the documents in hand. Score each of the seven reliability \
dimensions 0-5 against its rubric, using only positive evidence in the trace and \
final answer. 5 means the bar is met without caveat; 0 means it is not met at all. \
Return only the JSON object specified by the response schema. Do not add commentary.
"""


@dataclass(frozen=True)
class CmLrsDimension:
    """Static definition of one CM-LRS dimension: key, label, and rubric anchor."""

    key: str
    name: str
    rubric: str


# The seven dimensions in the order the paper presents them. Keys are stable
# identifiers used in the judge schema and in caller-supplied weight maps.
CM_LRS_DIMENSIONS: list[CmLrsDimension] = [
    CmLrsDimension(
        key="factual_accuracy",
        name="Factual Accuracy",
        rubric=(
            "Every material claim in the final output is correct against the reference "
            "answer and authoritative sources. 5 = no inaccuracies; 0 = material claims "
            "are wrong."
        ),
    ),
    CmLrsDimension(
        key="evidence_traceability",
        name="Evidence Traceability",
        rubric=(
            "Material claims are traceable to retrieved evidence (tool results shown in "
            "the trace). 5 = every claim cites a supporting passage; 0 = claims are "
            "unsourced."
        ),
    ),
    CmLrsDimension(
        key="numerical_consistency",
        name="Numerical Consistency",
        rubric=(
            "Figures in the final answer match the source figures and any computation in "
            "the trace; units, signs, and magnitude agree. 5 = fully consistent; 0 = "
            "figures conflict or are unsupported."
        ),
    ),
    CmLrsDimension(
        key="workflow_completeness",
        name="Workflow Completeness",
        rubric=(
            "The analyst steps implied by the task were actually carried out (retrieval, "
            "computation, synthesis), not merely glossed with fluent text. 5 = all steps "
            "evidenced; 0 = major steps missing."
        ),
    ),
    CmLrsDimension(
        key="source_discipline",
        name="Source Discipline",
        rubric=(
            "Sources used are correct, current, and primary (e.g. the right SEC filing, "
            "the right period); citations are clean and unambiguous. 5 = impeccable "
            "discipline; 0 = wrong or stale sources."
        ),
    ),
    CmLrsDimension(
        key="decision_usefulness",
        name="Decision Usefulness",
        rubric=(
            "The output is actionable for a decision-maker facing the question — "
            "directly answers it with the right caveats. 5 = ready to act on; 0 = "
            "unusable for a decision."
        ),
    ),
    CmLrsDimension(
        key="reviewability",
        name="Reviewability / Auditability",
        rubric=(
            "A reviewer can follow the reasoning end to end and reproduce the chain from "
            "sources to conclusion. 5 = fully auditable; 0 = opaque or unreproducible."
        ),
    ),
]


def _cm_lrs_user_prompt(
    question: str,
    reference_answer: str,
    final_answer: str | None,
    trace: str,
) -> str:
    dims = "\n".join(f"- {d.key} ({d.name}): {d.rubric}" for d in CM_LRS_DIMENSIONS)
    return f"""\
QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

AGENT'S FINAL ANSWER:
{final_answer or "[no final answer was produced]"}

AGENT'S TRACE (assistant text, tool calls, tool results):
{trace}

Score each of the seven CM-LRS reliability dimensions 0-5 against its rubric. Ground
evidence_traceability and source_discipline in the tool results shown in the trace;
ground numerical_consistency in the figures that appear there; and judge
workflow_completeness by whether the agent actually performed the analyst steps the
task implies, not by how fluent the prose is.

The dimensions and their rubrics:
{dims}
"""


def _cm_lrs_response_schema() -> dict[str, Any]:
    """Response schema for the CM-LRS judge.

    The dimensions array is intentionally unbounded (no minItems/maxItems) and the
    `score` integer carries no minimum/maximum, mirroring `grader.grade()`'s schema:
    Vertex Gemini rejects bounded integer schemas with "too many states". The grader
    clamps scores to [0, 5] client-side and tolerates a short/long dimension array.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["dimensions"],
        "properties": {
            "dimensions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["key", "score", "rationale"],
                    "properties": {
                        "key": {"type": "string"},
                        "score": {"type": "integer"},
                        "rationale": {"type": "string"},
                    },
                },
            },
        },
    }


def _aggregate_cm_lrs(
    scored: list[CmLrsDimensionScore],
    weights: dict[str, float] | None,
) -> tuple[float, dict[str, float]]:
    """Weighted mean of the 0-5 dimension scores.

    `weights` maps dimension keys to non-negative weights. A key absent from the map
    defaults to 1.0, so passing no weights yields the plain arithmetic mean across
    all scored dimensions. Weights are clamped to be non-negative; a dimension whose
    weight is explicitly 0 is dropped from the denominator. Scores are clamped to
    [0, 5] (the judge schema is unbounded). Returns the aggregate rounded to 4dp and
    the weighting actually applied (echoed on `CmLrsScore.weights` for reproducibility).
    """
    applied: dict[str, float] = {}
    for d in scored:
        w = 1.0 if weights is None else max(0.0, float(weights.get(d.key, 1.0)))
        applied[d.key] = w
    total = sum(applied.values())
    if total <= 0:
        return 0.0, applied
    weighted = sum(max(0, min(5, d.score)) * applied[d.key] for d in scored)
    return round(weighted / total, 4), applied


async def score_cm_lrs(
    *,
    run: RunRecord,
    item: DatasetItem,
    judge_model_id: str,
    weights: dict[str, float] | None = None,
    max_output_tokens: int = 8192,
    judge_alias: str | None = None,
) -> CmLrsScore:
    """Score a run's bankability on the 7-dimension CM-LRS rubric.

    `weights`: optional per-dimension weights for the tunable aggregate (see
    `_aggregate_cm_lrs`). `judge_alias`: if provided, the stored `CmLrsScore.judge`
    uses this string instead of `judge_model_id`, matching `grade()`'s aliasing.
    """
    provider, snapshot = parse_model_id(judge_model_id)
    judge_model = _to_litellm_model(provider, snapshot)

    trace = _format_trace(run.steps)
    user_prompt = _cm_lrs_user_prompt(
        question=item.query,
        reference_answer=item.reference_answer,
        final_answer=run.final_answer,
        trace=trace,
    )
    schema = _cm_lrs_response_schema()

    kwargs: dict[str, object] = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": CM_LRS_JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "num_retries": 20,
        "request_timeout": 1800,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "cm_lrs_scoring", "strict": True, "schema": schema},
        },
    }
    # Vertex routing mirrors `grader.grade()`; see that module for the dedicated-PT
    # header semantics and the VERTEX_DISABLE_DEDICATED escape hatch.
    if provider in ("vertex", "vertex-anthropic"):
        project = os.environ.get("VERTEXAI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project:
            kwargs["vertex_project"] = project
        kwargs["vertex_location"] = _vertex_location_for(provider)
        if not os.environ.get("VERTEX_DISABLE_DEDICATED"):
            kwargs["extra_headers"] = {"X-Vertex-AI-LLM-Request-Type": "dedicated"}
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
    judge_prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    judge_completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    hidden = getattr(response, "_hidden_params", {}) or {}
    judge_cost = hidden.get("response_cost")
    judge_cost_usd = float(judge_cost) if judge_cost is not None else None

    by_key: dict[str, dict[str, Any]] = {
        entry["key"]: entry for entry in parsed.get("dimensions", [])
    }
    scored: list[CmLrsDimensionScore] = []
    for dim in CM_LRS_DIMENSIONS:
        entry = by_key.get(dim.key, {"score": 0, "rationale": "missing"})
        try:
            score = int(entry.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        scored.append(
            CmLrsDimensionScore(
                key=dim.key,
                name=dim.name,
                score=score,
                rationale=entry.get("rationale"),
            )
        )
    aggregate, applied = _aggregate_cm_lrs(scored, weights)

    return CmLrsScore(
        question_id=item.id,
        trial_idx=run.trial_idx,
        model=run.model,
        judge=judge_alias if judge_alias else judge_model_id,
        dimensions=scored,
        aggregate=aggregate,
        weights=applied,
        final_answer=run.final_answer,
        judge_prompt_tokens=judge_prompt_tokens,
        judge_completion_tokens=judge_completion_tokens,
        judge_cost_usd=judge_cost_usd,
    )
