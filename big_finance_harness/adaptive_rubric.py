"""Self-adaptive rubric elaboration for the judge (SedarEval scheme).

The grader normally shows the judge a flat list of expert-authored rubric lines and
asks, per line, whether the trace evidences it. SedarEval (arXiv:2501.15595) shows
that structuring the rubric the judge sees — breaking each line into *primary
criteria* (must hold), *secondary criteria* (supporting), and *deduction points*
(specific failure modes that should mark the line unsatisfied) — improves judge
precision and stability, mirroring how human exams are scored.

This module elaborates the repo's EXISTING expert rubric into that structure using the
judge model itself (one structured-output call per question). It does NOT replace the
expert rubric, invent new ground truth, or touch the system-under-test agent — it only
restructures what the judge reads. Numeric point values are deliberately excluded from
the elaboration so the judge cannot be biased toward heavy-weight lines, preserving the
grader's existing bias-avoidance design (points are still aggregated client-side).

The elaboration is opt-in: ``grade(..., adaptive_rubric=True)`` calls
``elaborate_rubric`` once before the grading call and feeds the structured rubric into
the judge prompt. With the flag off, ``grade`` behaves exactly as before.
"""

from __future__ import annotations

import json
import os
from typing import Any

import litellm
from pydantic import BaseModel, Field

from big_finance_harness.models.base import (
    _to_litellm_model,
    _vertex_location_for,
    parse_model_id,
)
from big_finance_harness.types import DatasetItem, RubricLine

ELABORATION_SYSTEM = """\
You adapt an expert-authored analyst rubric into a structured scoring guide for an
LLM judge, in the style of a human exam rubric.

For each rubric line you receive, produce:
  - primary_criteria: the one or two conditions that MUST hold for the line to count
    (restate the line's core requirement concretely for THIS question).
  - secondary_criteria: supporting considerations that strengthen the evidence but are
    not alone sufficient (e.g. citing the right filing, showing the calculation).
  - deduction_points: specific, concrete failure modes that should make the judge mark
    the line NOT satisfied (e.g. wrong sign, wrong units, wrong fiscal year, wrong
    entity/ticker, stale data, off-by-one reporting period).

Rules:
  - ELABORATE the given expert lines. Do NOT invent new rubric lines, new requirements,
    or new ground truth beyond what the expert rubric and reference answer imply.
  - Do NOT mention numeric point values or weights anywhere — the judge must not see
    them.
  - Keep each criterion / failure-mode short (one clause). Aim for 1-2 primary, 0-3
    secondary, and 1-4 deduction points per line.

Return only the JSON object specified by the response schema. Do not add commentary.
"""


class ElaboratedRubricLine(BaseModel):
    """One expert rubric line restructured into the SedarEval scoring scheme."""

    index: int
    primary_criteria: list[str] = Field(default_factory=list)
    secondary_criteria: list[str] = Field(default_factory=list)
    deduction_points: list[str] = Field(default_factory=list)


def _elaboration_user_prompt(item: DatasetItem) -> str:
    rubric_block = "\n".join(f"{i + 1}. {line.text}" for i, line in enumerate(item.rubric))
    return f"""\
QUESTION:
{item.query}

REFERENCE ANSWER:
{item.reference_answer}

EXPERT RUBRIC (one line per analyst step; numbered):
{rubric_block}

For each numbered rubric line above, return its structured scoring guide
(primary_criteria, secondary_criteria, deduction_points) adapted to THIS question.
"""


def _elaboration_schema() -> dict[str, Any]:
    """Response schema for the elaboration call.

    Mirrors the grader's schema choices: no ``minItems``/``maxItems`` on the array,
    which Vertex Gemini's structured-output validator rejects at scale. Missing or
    short arrays are reconciled against the expert rubric in ``elaborate_rubric``.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["rubric"],
        "properties": {
            "rubric": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "index",
                        "primary_criteria",
                        "secondary_criteria",
                        "deduction_points",
                    ],
                    "properties": {
                        "index": {"type": "integer"},
                        "primary_criteria": {"type": "array", "items": {"type": "string"}},
                        "secondary_criteria": {"type": "array", "items": {"type": "string"}},
                        "deduction_points": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    }


async def elaborate_rubric(
    *,
    item: DatasetItem,
    judge_model_id: str,
    max_output_tokens: int = 8192,
) -> list[ElaboratedRubricLine]:
    """Elaborate the expert rubric into the SedarEval structured form.

    One structured-output call to the judge model. Returns one ``ElaboratedRubricLine``
    per expert line, aligned to the expert rubric's order; any line the model omits or
    mis-indexes falls back to a trivial elaboration (the line's own text as the sole
    primary criterion) so the judge never loses sight of a rubric line.
    """
    provider, snapshot = parse_model_id(judge_model_id)
    judge_model = _to_litellm_model(provider, snapshot)

    kwargs: dict[str, object] = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": ELABORATION_SYSTEM},
            {"role": "user", "content": _elaboration_user_prompt(item)},
        ],
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "num_retries": 20,
        "request_timeout": 1800,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "rubric_elaboration",
                "strict": True,
                "schema": _elaboration_schema(),
            },
        },
    }
    # Mirror the grader's Vertex routing so the same judge snapshot is used for
    # elaboration as for grading. See `big_finance_harness/models/base.py`.
    if provider in ("vertex", "vertex-anthropic"):
        project = os.environ.get("VERTEXAI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project:
            kwargs["vertex_project"] = project
        kwargs["vertex_location"] = _vertex_location_for(provider)
        if not os.environ.get("VERTEX_DISABLE_DEDICATED"):
            kwargs["extra_headers"] = {"X-Vertex-AI-LLM-Request-Type": "dedicated"}

    # Share the grader's per-judge semaphore so elaboration counts against the same
    # concurrency budget as grading (one budget per judge model). Imported lazily to
    # avoid a load-time cycle: grader imports this module at its top level.
    from big_finance_harness.grader import _judge_semaphore

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
    by_index: dict[int, ElaboratedRubricLine] = {}
    for entry in parsed.get("rubric", []):
        idx = int(entry.get("index", 0))
        by_index[idx] = ElaboratedRubricLine(
            index=idx,
            primary_criteria=list(entry.get("primary_criteria") or []),
            secondary_criteria=list(entry.get("secondary_criteria") or []),
            deduction_points=list(entry.get("deduction_points") or []),
        )

    # Align to the expert rubric's ordering/indices. Missing lines fall back to a
    # trivial elaboration (the line's own text) so every line still reaches the judge.
    out: list[ElaboratedRubricLine] = []
    for i, line in enumerate(item.rubric):
        elab = by_index.get(i + 1)
        if elab is None:
            elab = ElaboratedRubricLine(index=i + 1, primary_criteria=[line.text])
        out.append(elab)
    return out


def format_elaborated_rubric(
    rubric: list[RubricLine],
    elaborated: list[ElaboratedRubricLine],
) -> str:
    """Render the structured (SedarEval-style) rubric as text for the judge.

    A line whose elaboration is empty falls back to the plain ``"N. <text>"`` form, so
    a partial or failed elaboration never drops a rubric line from the judge's view.
    """
    by_index = {e.index: e for e in elaborated}
    blocks: list[str] = []
    for i, line in enumerate(rubric):
        idx = i + 1
        elab = by_index.get(idx)
        if elab is None or not (
            elab.primary_criteria or elab.secondary_criteria or elab.deduction_points
        ):
            blocks.append(f"{idx}. {line.text}")
            continue
        parts: list[str] = [f"{idx}. {line.text}"]
        if elab.primary_criteria:
            parts.append("   primary (must hold): " + "; ".join(elab.primary_criteria))
        if elab.secondary_criteria:
            parts.append("   secondary: " + "; ".join(elab.secondary_criteria))
        if elab.deduction_points:
            parts.append("   mark unsatisfied if: " + "; ".join(elab.deduction_points))
        blocks.append("\n".join(parts))
    return "\n".join(blocks)
