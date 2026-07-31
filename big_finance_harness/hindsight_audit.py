"""Parametric-hindsight audit for time-indexed financial decision tasks.

Adapted from *HindsightBench: A Black-Box Behavioral Audit Protocol for Parametric
Hindsight in Time-Indexed LLM Decision Tasks* (arXiv:2607.18867v1). The paper profiles
"parametric hindsight" -- a model leaking parametric knowledge of a *realized* outcome
into a *historical* decision task -- via a black-box, probe-level protocol (no backtests,
no logprobs, no corpus access). This module ports the protocol's core mechanism onto the
harness's existing text-API client (`ModelClient`, obtained via `make_client`).

Core mechanism kept at fidelity:

  * The four-arm date-manipulation matrix -- `revealed` / `date_only` / `masked` /
    `transplanted` -- rewrites the *same* decision task with different date cues.
  * Dual memory probes -- *date recovery* (can the model name the decision date?) and
    *outcome recall* (does the model state the realized outcome when asked directly?).
  * Per-probe metrics: `trigger_strength` (revealed minus masked leakage -- does naming
    the date *trigger* hindsight?), `transplant_effect` (revealed minus transplanted --
    is the *specific* true date the trigger?), `recoverability`, and `recall`.

Mode-2 substitutions (vs. the paper):

  * The paper's 258-node vintage-correct macro panel is replaced by a small built-in
    `DEFAULT_PROBES` set of clearly-historical financial hindsight scenarios (with an
    optional JSONL override). The two metrics that *require* the macro panel --
    behaviorally-effective knowledge cutoff and the recall-accuracy dissociation
    coefficient -- are intentionally out of scope here; they need a sweep across many
    cutoff dates per model and belong in a downstream evaluation PR.
  * The paper's disclosed parser is replaced by a parameter-free
    substring / numeric-overlap leakage detector (`leakage_score`).
  * The paper's standalone audit/regen framework is replaced by an opt-in audit phase
    wired into the orchestrator (`scripts/run_eval_set.py --hindsight-audit`) that emits
    one `<label>.hindsight.jsonl` per model, sibling to `grades.jsonl`.

Serving-invariance caveats from the paper (pin quantization and thinking regime; disclose
parser and sampling policy) apply to any real audit run: pass a fixed `thinking` level and
`temperature` and record them alongside the row.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field

from big_finance_harness.models.base import ModelClient, ThinkingLevel
from big_finance_harness.types import Message, ModelResponse, TextBlock

# Neutral analyst persona. Deliberately NOT the ReAct agent system prompt (these probes
# are single-turn, tool-free, black-box -- the paper's probe-level-cost contract).
_PROBE_SYSTEM_PROMPT = "You are a financial analyst. Answer the user's question concisely."

Arm = Literal["revealed", "date_only", "masked", "transplanted"]
ARMS: tuple[Arm, ...] = ("revealed", "date_only", "masked", "transplanted")

_YEAR_RE = re.compile(r"(19|20)\d{2}")


class HindsightProbe(BaseModel):
    """One time-indexed decision task with a realized post-cutoff outcome.

    `task` is written WITHOUT the cutoff date and WITHOUT the outcome (the masked base).
    `outcome` is the realized event that occurs strictly after `cutoff_date` -- exactly
    what a model with hindsight would leak. `placebo_date` defaults to an anachronistic
    (pre-outcome) date derived from `cutoff_date`; override for a custom control.
    """

    id: str
    task: str
    cutoff_date: str
    outcome: str
    recall_question: str | None = None
    placebo_date: str | None = None


class HindsightAuditRow(BaseModel):
    """Per-(model, probe) audit row, written one-per-line to `<label>.hindsight.jsonl`."""

    probe_id: str
    model: str  # client.snapshot (the id passed at invocation)
    resolved_model: str | None = None
    cutoff_date: str
    placebo_date: str
    n_trials: int
    # Mean leakage per arm across trials, in [0, 1].
    revealed_leakage: float
    date_only_leakage: float
    masked_leakage: float
    transplanted_leakage: float
    # Headline metrics from the paper (first four of its six).
    trigger_strength: float  # revealed - masked: does naming the date trigger hindsight?
    transplant_effect: float  # revealed - transplanted: is the *specific* true date the trigger?
    recoverability: float  # date-recovery probe accuracy (0/1 mean across trials)
    recall: float  # outcome-recall probe (mean leakage across trials)
    # Cost accounting so the paper can report per-probe audit cost.
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    cost_usd: float | None = None
    # Raw arm responses (one list per arm, across trials) for auditability/replay.
    arms: dict[str, list[str]] = Field(default_factory=dict)


def _shift_year(date_str: str, delta: int) -> str:
    """Return `date_str` with its first 4-digit year shifted by `delta` (anachronism)."""
    match = _YEAR_RE.search(date_str)
    if match is None:
        return date_str
    year = int(match.group(0)) + delta
    return date_str[: match.start()] + str(year) + date_str[match.end() :]


def _placebo_date_for(probe: HindsightProbe) -> str:
    return probe.placebo_date or _shift_year(probe.cutoff_date, -5)


def build_arms(probe: HindsightProbe) -> dict[Arm, str]:
    """Construct the four-arm date-manipulation matrix prompts for one probe.

    - revealed:    strong "as of {cutoff}" framing, twice.
    - date_only:   bare parenthetical date cue, no "as of" framing.
    - masked:      the date is stripped entirely.
    - transplanted: revealed framing with the cutoff swapped for a placebo date.
    """
    cutoff = probe.cutoff_date
    placebo = _placebo_date_for(probe)
    task = probe.task.strip()
    return {
        "revealed": (
            f"As of {cutoff}, {task}\n\n"
            f"Answer based only on information available on or before {cutoff}."
        ),
        "date_only": f"{task}\n\n(Context date: {cutoff}.)",
        "masked": f"{task}\n\nAnswer based only on currently available information.",
        "transplanted": (
            f"As of {placebo}, {task}\n\n"
            f"Answer based only on information available on or before {placebo}."
        ),
    }


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def _numeric_tokens(text: str) -> list[str]:
    return _NUM_RE.findall(text)


def leakage_score(response: str, outcome: str) -> float:
    """Parameter-free, disclosed parser: how much does `response` leak `outcome`?

    Returns 1.0 for a full normalized-substring match of the outcome, else the fraction
    of the outcome's numeric tokens that appear in the response (numbers are the
    highest-signal part of a financial outcome), else the fraction of overlapping word
    tokens. Always in [0, 1].
    """
    norm_response = _normalize(response)
    norm_outcome = _normalize(outcome)
    if not norm_outcome:
        return 0.0
    if norm_outcome in norm_response:
        return 1.0
    outcome_nums = _numeric_tokens(norm_outcome)
    if outcome_nums:
        response_nums = set(_numeric_tokens(norm_response))
        hits = sum(1 for n in outcome_nums if n in response_nums)
        return hits / len(outcome_nums)
    outcome_tokens = norm_outcome.split()
    if not outcome_tokens:
        return 0.0
    response_tokens = set(norm_response.split())
    hits = sum(1 for token in outcome_tokens if token in response_tokens)
    return hits / len(outcome_tokens)


def _date_recovery_prompt(probe: HindsightProbe) -> str:
    return (
        f"Read the following analyst task and state the date the decision is being made. "
        f"Reply with only the date.\n\nTask: {probe.task}"
    )


def _recall_prompt(probe: HindsightProbe) -> str:
    question = probe.recall_question or (
        f"What was the outcome of: {probe.task} (resolved after {probe.cutoff_date})?"
    )
    return question


async def _probe(client: ModelClient, user_text: str, **chat_kwargs: object) -> ModelResponse:
    messages = [Message(role="user", content=[TextBlock(text=user_text)])]
    return await client.chat(
        system=_PROBE_SYSTEM_PROMPT,
        messages=messages,
        tools=[],
        **chat_kwargs,  # type: ignore[arg-type]
    )


async def run_hindsight_audit(
    *,
    client: ModelClient,
    probes: list[HindsightProbe],
    n_trials: int = 1,
    temperature: float | None = 0.0,
    thinking: ThinkingLevel = "off",
    max_output_tokens: int = 1024,
) -> list[HindsightAuditRow]:
    """Run the four-arm audit + dual memory probes for every probe against one model.

    Each probe issues (4 arms + date recovery + outcome recall) calls per trial through
    `client.chat` -- the same black-box text API the harness uses for the ReAct agent,
    obtained via `make_client`. Returns one `HindsightAuditRow` per probe.
    """
    chat_kwargs = {
        "temperature": temperature,
        "thinking": thinking,
        "max_output_tokens": max_output_tokens,
    }
    rows: list[HindsightAuditRow] = []
    for probe in probes:
        arm_prompts = build_arms(probe)
        arm_responses: dict[str, list[str]] = {arm: [] for arm in ARMS}
        recover_hits = 0
        recall_scores: list[float] = []
        total_prompt = 0
        total_completion = 0
        cost: float | None = None
        resolved_model: str | None = None

        for _ in range(max(1, n_trials)):
            for arm in ARMS:
                resp = await _probe(client, arm_prompts[arm], **chat_kwargs)
                arm_responses[arm].append(resp.text or "")
                total_prompt += resp.prompt_tokens
                total_completion += resp.completion_tokens
                if resp.cost_usd is not None:
                    cost = (cost or 0.0) + resp.cost_usd
                if resolved_model is None and resp.resolved_model:
                    resolved_model = resp.resolved_model

            # Date-recovery memory probe.
            rec = await _probe(client, _date_recovery_prompt(probe), **chat_kwargs)
            total_prompt += rec.prompt_tokens
            total_completion += rec.completion_tokens
            if rec.cost_usd is not None:
                cost = (cost or 0.0) + rec.cost_usd
            if _normalize(probe.cutoff_date) and _normalize(probe.cutoff_date) in _normalize(
                rec.text or ""
            ):
                recover_hits += 1

            # Outcome-recall memory probe.
            rcl = await _probe(client, _recall_prompt(probe), **chat_kwargs)
            total_prompt += rcl.prompt_tokens
            total_completion += rcl.completion_tokens
            if rcl.cost_usd is not None:
                cost = (cost or 0.0) + rcl.cost_usd
            recall_scores.append(leakage_score(rcl.text or "", probe.outcome))

        def _mean_leak(arm: Arm) -> float:
            scores = [leakage_score(t, probe.outcome) for t in arm_responses[arm]]
            return sum(scores) / len(scores) if scores else 0.0

        revealed = _mean_leak("revealed")
        masked = _mean_leak("masked")
        transplanted = _mean_leak("transplanted")
        date_only = _mean_leak("date_only")

        rows.append(
            HindsightAuditRow(
                probe_id=probe.id,
                model=client.snapshot,
                resolved_model=resolved_model,
                cutoff_date=probe.cutoff_date,
                placebo_date=_placebo_date_for(probe),
                n_trials=max(1, n_trials),
                revealed_leakage=round(revealed, 4),
                date_only_leakage=round(date_only, 4),
                masked_leakage=round(masked, 4),
                transplanted_leakage=round(transplanted, 4),
                trigger_strength=round(revealed - masked, 4),
                transplant_effect=round(revealed - transplanted, 4),
                recoverability=round(recover_hits / max(1, n_trials), 4),
                recall=round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0,
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
                cost_usd=cost,
                arms=arm_responses,
            )
        )
    return rows


def write_audit_jsonl(path: str | Path, rows: list[HindsightAuditRow]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def read_audit_jsonl(path: str | Path) -> Iterator[HindsightAuditRow]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield HindsightAuditRow.model_validate_json(line)


def load_probes(path: str | Path) -> list[HindsightProbe]:
    """Load `HindsightProbe` rows from a JSONL file (one probe per line)."""
    p = Path(path)
    probes: list[HindsightProbe] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                probes.append(HindsightProbe.model_validate(json.loads(line)))
    return probes


# Built-in default probe set -- clearly-historical, widely-documented financial events
# with an unambiguous pre-cutoff decision point and a realized post-cutoff outcome.
# These are seeds for a real audit; supply a JSONL via `load_probes` for a custom panel.
DEFAULT_PROBES: list[HindsightProbe] = [
    HindsightProbe(
        id="svb-2023",
        task=(
            "Assess the solvency and deposit-safety risk of Silicon Valley Bank "
            "(SIVB) and advise whether a corporate depositor should move funds out."
        ),
        cutoff_date="2023-03-08",
        outcome=(
            "Silicon Valley Bank was shut down by California regulators on March 10, "
            "2023 and placed into FDIC receivership after a bank run."
        ),
        recall_question="What happened to Silicon Valley Bank in mid-March 2023?",
    ),
    HindsightProbe(
        id="ftx-2022",
        task=(
            "Assess the solvency and counterparty risk of the crypto exchange FTX and "
            "whether a user should withdraw their assets."
        ),
        cutoff_date="2022-11-07",
        outcome=(
            "FTX filed for Chapter 11 bankruptcy on November 11, 2022 and founder "
            "Sam Bankman-Fried resigned as CEO."
        ),
        recall_question="What happened to FTX in November 2022?",
    ),
    HindsightProbe(
        id="credit-suisse-2023",
        task=(
            "Assess the viability and counterparty risk of Credit Suisse and whether "
            "exposure should be reduced."
        ),
        cutoff_date="2023-03-15",
        outcome=(
            "UBS agreed to acquire Credit Suisse in an emergency rescue brokered by the "
            "Swiss government on March 19, 2023."
        ),
        recall_question="What was the resolution of Credit Suisse in March 2023?",
    ),
    HindsightProbe(
        id="covid-crash-2020",
        task=(
            "Assess the near-term downside risk for the S&P 500 and whether an equity "
            "investor should de-risk."
        ),
        cutoff_date="2020-02-19",
        outcome=(
            "The S&P 500 fell roughly 34% from its February 19, 2020 peak to its "
            "March 23, 2020 trough during the COVID-19 crash."
        ),
        recall_question="How far did the S&P 500 fall during the March 2020 COVID-19 crash?",
    ),
]


__all__ = [
    "ARMS",
    "DEFAULT_PROBES",
    "HindsightAuditRow",
    "HindsightProbe",
    "build_arms",
    "leakage_score",
    "load_probes",
    "read_audit_jsonl",
    "run_hindsight_audit",
    "write_audit_jsonl",
]
