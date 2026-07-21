"""LLM-as-a-Verifier continuous scoring — calibrated verdicts from scoring-token logprobs.

Adapted from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391). The paper's core contribution — computing the *expectation
over the distribution of scoring-token logits* to turn a discrete LM judge into a
continuous, calibrated verifier — is implemented here at full fidelity.

The harness's rubric grader (`big_finance_harness.grader.grade`) asks a judge, in
one structured call, to emit booleans for final-answer correctness and each rubric
line. That is exactly the "standard LM judge that prompts LLMs to produce discrete
scores" the paper improves upon. When ``grade(..., continuous_score=True)`` is set,
this module adds a second pass: for each criterion it asks the judge a focused
yes/no/partial question and reads the *logprobs* of the first generated token,
mapping every candidate scoring token to a numeric weight and returning

    E[score] = Σ_token softmax(logprob)_token · weight(token)   ∈ [0, 1].

This exercises the paper's "score granularity" and "criteria decomposition"
scaling axes: granularity scales with how many tokens the weight map spans
(binary yes/no vs. a graded partial scale), and decomposition scores each rubric
line independently. ``num_samples > 1`` exercises the "repeated evaluation" axis
(variance reduction by averaging several temperature-sampled passes).

Mode 2 (adapted port) — substitutions of the paper's auxiliaries with
target-native equivalents:

* Scoring-token logprobs come from LiteLLM's ``logprobs``/``top_logprobs`` on the
  same OpenAI/Vertex judge path ``grade`` already routes, rather than a bespoke
  provider client. Token strings are normalized (lowercased, leading/trailing
  whitespace stripped) before matching the weight map, since providers differ on
  how they emit them.
* The paper's cost-efficient candidate-*ranking* algorithm (best-of-N selection),
  its Claude-Code task-progress extension, and its RL dense-feedback results
  (SAC/GRPO sample efficiency) are intentionally out of scope: the harness grades
  single runs (no candidate set to rank) and hosts no trainer to feed dense
  rewards into. Those contributions map to different surfaces than this grader.

Attribution: arXiv:2607.05391. This module implements the verifier; it does not
reproduce the paper's reported benchmark numbers.
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any

import litellm

from big_finance_harness.models.base import (
    _to_litellm_model,
    _vertex_location_for,
    parse_model_id,
)
from big_finance_harness.types import (
    DatasetItem,
    RunRecord,
    VerifierCriterionScore,
    VerifierScores,
)

# A graded scale (not just binary yes/no) is the paper's "score granularity" axis:
# finer granularity separates borderline-correct solutions better than a boolean.
DEFAULT_SCORING_TOKENS: dict[str, float] = {
    "yes": 1.0,
    "correct": 1.0,
    "true": 1.0,
    "satisfied": 1.0,
    "partial": 0.5,
    "partially": 0.5,
    "maybe": 0.5,
    "uncertain": 0.5,
    "no": 0.0,
    "incorrect": 0.0,
    "false": 0.0,
    "unsatisfied": 0.0,
}

_VERIFIER_SYSTEM = (
    "You are an impartial verifier for a financial-research agent benchmark. "
    "Reply with exactly one token from the allowed set and nothing else."
)

# The verifier sees a short trace excerpt; the boolean judge already sees the full
# trace. Keeping the excerpt small bounds the per-criterion verifier cost.
_TRACE_EXCERPT_CHARS = 6_000


def _normalize_token(token: Any) -> str:
    return str(token).strip().lower()


def _lp_value(entry: Any, key: str) -> Any:
    # top_logprobs entries are dicts in OpenAI shape but may surface as attribute
    # objects on some LiteLLM versions — accept both.
    if isinstance(entry, dict):
        return entry.get(key)
    return getattr(entry, key, None)


def _first_token_logprobs(choice: Any) -> list[dict[str, Any]]:
    """Return candidate {token, logprob} entries for the first generated token.

    LiteLLM normalizes chat-completion logprobs to ``choice.logprobs.content`` — a
    list of token infos each carrying ``token``, ``logprob``, and ``top_logprobs``
    (a list of {token, logprob}). This is the shape all providers surface through
    ``acompletion`` (the only call path the harness uses), so we parse that and
    fold the sampled token in when a provider omits it from ``top_logprobs``.
    """
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        return []
    content = getattr(logprobs, "content", None)
    if not content:
        return []
    first = content[0]
    candidates = [
        {
            "token": _lp_value(t, "token"),
            "logprob": float(_lp_value(t, "logprob") or 0.0),
        }
        for t in (getattr(first, "top_logprobs", None) or [])
    ]
    gen_tok = getattr(first, "token", None)
    if gen_tok is not None:
        seen = {_normalize_token(c["token"]) for c in candidates}
        if _normalize_token(gen_tok) not in seen:
            candidates.append(
                {"token": gen_tok, "logprob": float(getattr(first, "logprob", 0.0) or 0.0)}
            )
    return candidates


def _expectation(
    candidates: list[dict[str, Any]], scoring_tokens: dict[str, float]
) -> tuple[float | None, list[dict[str, Any]]]:
    """E[score] over the softmax of the matched scoring-token logprobs.

    Only tokens present in ``scoring_tokens`` (after normalization) contribute, and
    their probabilities are renormalized over the matched set — so the result is a
    proper expectation in the score range, typically [0, 1]. Returns ``(None, [])``
    when no emitted token matched the scoring vocabulary.
    """
    matched: list[tuple[Any, float, float]] = []
    for c in candidates:
        weight = scoring_tokens.get(_normalize_token(c["token"]))
        if weight is not None:
            matched.append((c["token"], float(weight), float(c["logprob"])))
    if not matched:
        return None, []
    max_lp = max(m[2] for m in matched)
    weights = [math.exp(m[2] - max_lp) for m in matched]
    total = sum(weights)
    score = sum(weights[i] * matched[i][1] for i in range(len(matched))) / total
    detail = [
        {
            "token": matched[i][0],
            "score": matched[i][1],
            "logprob": matched[i][2],
            "probability": weights[i] / total,
        }
        for i in range(len(matched))
    ]
    return score, detail


def _final_answer_prompt(item: DatasetItem, run: RunRecord, trace: str) -> str:
    return f"""\
You are verifying whether an agent's final answer is correct.

QUESTION:
{item.query}

REFERENCE ANSWER:
{item.reference_answer}

AGENT'S FINAL ANSWER:
{run.final_answer or "[no final answer was produced]"}

Excerpt of the agent's trace:
{trace[:_TRACE_EXCERPT_CHARS]}

Does the agent's final answer match the reference answer? Numerically equivalent
values count as matching; minor formatting is acceptable; sign and units must match.

Reply with a SINGLE token — one of: yes, no, partial. Do not add any other text.
"""


def _rubric_prompt(item: DatasetItem, run: RunRecord, trace: str, line_text: str) -> str:
    return f"""\
You are verifying one analyst rubric step from a financial-research agent run.

QUESTION:
{item.query}

REFERENCE ANSWER:
{item.reference_answer}

RUBRIC STEP TO VERIFY:
{line_text}

AGENT'S FINAL ANSWER:
{run.final_answer or "[no final answer was produced]"}

Excerpt of the agent's trace:
{trace[:_TRACE_EXCERPT_CHARS]}

Is this rubric step satisfied by the trace and final answer together? A step is
satisfied only if the trace contains positive evidence for it.

Reply with a SINGLE token — one of: yes, no, partial. Do not add any other text.
"""


async def _verify_criterion(
    *,
    judge_model_id: str,
    user_prompt: str,
    scoring_tokens: dict[str, float],
    top_logprobs: int,
    num_samples: int,
    temperature: float,
    max_output_tokens: int,
) -> tuple[float | None, float | None, list[dict[str, Any]], int]:
    """Run ``num_samples`` logprob passes for one criterion; return aggregated score.

    Returns ``(mean_score, variance, first_sample_detail, n_successful)``. The
    detail is captured from the first sample only, to bound payload size.
    """
    # Local import: grader imports this module lazily inside grade(), so importing
    # grader here at call time avoids a load-time cycle and shares the per-model
    # judge concurrency budget with the boolean grader.
    from big_finance_harness.grader import _judge_semaphore

    provider, snapshot = parse_model_id(judge_model_id)
    judge_model = _to_litellm_model(provider, snapshot)

    async def one_sample() -> tuple[float | None, list[dict[str, Any]]]:
        kwargs: dict[str, object] = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": _VERIFIER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "num_retries": 20,
            "request_timeout": 1800,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        # Mirror the grader's Vertex routing — same provider path the boolean judge uses.
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
        return _expectation(_first_token_logprobs(response.choices[0]), scoring_tokens)

    samples = await asyncio.gather(*(one_sample() for _ in range(num_samples)))
    scores = [s for s, _ in samples if s is not None]
    if not scores:
        return None, None, [], 0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores) if len(scores) > 1 else 0.0
    first_detail = next((d for _, d in samples if d), [])
    return mean, variance, first_detail, len(scores)


async def verify_criteria(
    *,
    judge_model_id: str,
    item: DatasetItem,
    run: RunRecord,
    trace: str,
    scoring_tokens: dict[str, float] | None = None,
    top_logprobs: int = 20,
    num_samples: int = 1,
    temperature: float | None = None,
    max_output_tokens: int = 512,
) -> VerifierScores:
    """Score final-answer correctness and every rubric line via the logprob verifier.

    Each criterion is scored independently (criteria decomposition) as a continuous
    E[score] ∈ [0, 1]. ``num_samples > 1`` turns on repeated evaluation (variance
    reduction); in that case the per-criterion sampling temperature defaults to 0.7
    unless overridden.
    """
    tokens = scoring_tokens if scoring_tokens is not None else DEFAULT_SCORING_TOKENS
    temp = temperature if temperature is not None else (0.7 if num_samples > 1 else 0.0)

    async def score_final() -> VerifierCriterionScore:
        value, variance, detail, n = await _verify_criterion(
            judge_model_id=judge_model_id,
            user_prompt=_final_answer_prompt(item, run, trace),
            scoring_tokens=tokens,
            top_logprobs=top_logprobs,
            num_samples=num_samples,
            temperature=temp,
            max_output_tokens=max_output_tokens,
        )
        return VerifierCriterionScore(
            label="final_answer",
            score=value,
            num_samples=n,
            variance=variance,
            matched_tokens=detail,
        )

    async def score_line(idx: int, text: str) -> VerifierCriterionScore:
        value, variance, detail, n = await _verify_criterion(
            judge_model_id=judge_model_id,
            user_prompt=_rubric_prompt(item, run, trace, text),
            scoring_tokens=tokens,
            top_logprobs=top_logprobs,
            num_samples=num_samples,
            temperature=temp,
            max_output_tokens=max_output_tokens,
        )
        return VerifierCriterionScore(
            label=f"rubric:{idx + 1}",
            score=value,
            num_samples=n,
            variance=variance,
            matched_tokens=detail,
        )

    # All criteria verified concurrently; each is itself a focused single-token call.
    final_answer, *rubric_lines = await asyncio.gather(
        score_final(),
        *(score_line(i, line.text) for i, line in enumerate(item.rubric)),
    )

    return VerifierScores(
        judge_model=judge_model_id,
        scoring_tokens=tokens,
        top_logprobs=top_logprobs,
        final_answer=final_answer,
        rubric_lines=list(rubric_lines),
    )
