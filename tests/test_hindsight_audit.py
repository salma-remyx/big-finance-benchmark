"""Integration tests for the HindsightBench parametric-hindsight audit.

Drives `run_hindsight_audit` through a scripted `ModelClient` -- the same black-box
text-API contract `make_client` produces -- to verify the four-arm date matrix, the
leakage detector, and the headline metrics (trigger strength / transplant effect /
recoverability / recall) compute correctly when a model leaks the realized outcome on
the revealed arm but not the masked arm.
"""

from __future__ import annotations

import pytest

from big_finance_harness.hindsight_audit import (
    DEFAULT_PROBES,
    HindsightProbe,
    HindsightAuditRow,
    build_arms,
    leakage_score,
    read_audit_jsonl,
    run_hindsight_audit,
    write_audit_jsonl,
)
from big_finance_harness.models.base import ModelClient, ThinkingLevel
from big_finance_harness.types import Message, ModelResponse, ToolSpec

_PROBE = HindsightProbe(
    id="test-event",
    task="Assess the risk of holding ExampleCorp shares.",
    cutoff_date="2023-03-08",
    outcome="ExampleCorp was placed into FDIC receivership on March 10, 2023.",
    recall_question="What happened to ExampleCorp in mid-March 2023?",
)


class _DateTriggeredClient(ModelClient):
    """Leaks the realized outcome ONLY when the revealed-arm date cue is present.

    Mimics parametric hindsight triggered by naming the true cutoff date: the response
    states the post-cutoff outcome when the prompt says "as of {cutoff}" (revealed arm)
    but hedges generically on every other arm / probe.
    """

    snapshot = "anthropic:claude-test-2026-01-01"

    def __init__(self, probe: HindsightProbe) -> None:
        self.probe = probe
        self.calls = 0

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float | None = None,
        thinking: ThinkingLevel = "off",
        max_output_tokens: int = 1024,
    ) -> ModelResponse:
        self.calls += 1
        user_text = "".join(b.text for m in messages for b in m.content if hasattr(b, "text"))
        revealed_marker = f"as of {self.probe.cutoff_date}"
        is_recall = (
            "ExampleCorp" in user_text
            and self.probe.outcome not in user_text
            and ("happened" in user_text.lower())
        )
        if revealed_marker in user_text.lower():
            text = self.probe.outcome  # hindsight leak on the revealed arm
        elif is_recall:
            text = self.probe.outcome  # direct recall also surfaces it
        else:
            text = "Insufficient information to assess; risk appears moderate."  # no leak
        return ModelResponse(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            prompt_tokens=5,
            completion_tokens=7,
        )


class _CleanClient(ModelClient):
    """Never leaks -- hedged answers and a correct date-recovery on every call."""

    snapshot = "anthropic:claude-test-2026-01-01"

    def __init__(self, probe: HindsightProbe) -> None:
        self.probe = probe

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float | None = None,
        thinking: ThinkingLevel = "off",
        max_output_tokens: int = 1024,
    ) -> ModelResponse:
        user_text = "".join(b.text for m in messages for b in m.content if hasattr(b, "text"))
        # Date-recovery probe: answer with the cutoff date so recoverability is exercised.
        if "state the date" in user_text.lower():
            text = self.probe.cutoff_date
        else:
            text = "I have no view on this."  # no leakage, ever
        return ModelResponse(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            prompt_tokens=5,
            completion_tokens=4,
        )


@pytest.mark.asyncio
async def test_trigger_strength_positive_when_date_triggers_leak():
    client = _DateTriggeredClient(_PROBE)
    rows = await run_hindsight_audit(client=client, probes=[_PROBE], n_trials=1)
    assert len(rows) == 1
    row = rows[0]
    # Revealed arm leaked; masked arm did not -> trigger strength is maximal.
    assert row.revealed_leakage == 1.0
    assert row.masked_leakage == 0.0
    assert row.trigger_strength == 1.0
    # The transplanted arm uses a placebo date, so it is not the revealed marker -> no leak.
    assert row.transplanted_leakage == 0.0
    assert row.transplant_effect == 1.0
    # Direct outcome-recall probe surfaced the outcome.
    assert row.recall == 1.0
    # The audit drove the model through all arms + memory probes.
    assert client.calls == 6


@pytest.mark.asyncio
async def test_trigger_strength_zero_for_clean_model():
    client = _CleanClient(_PROBE)
    rows = await run_hindsight_audit(client=client, probes=[_PROBE], n_trials=1)
    row = rows[0]
    assert row.trigger_strength == 0.0
    assert row.transplant_effect == 0.0
    assert row.recall == 0.0
    # Date-recovery probe answered correctly -> recoverability is full.
    assert row.recoverability == 1.0


def test_build_arms_matrix_has_distinct_date_cues():
    arms = build_arms(_PROBE)
    assert f"As of {_PROBE.cutoff_date}" in arms["revealed"]
    assert _PROBE.cutoff_date not in arms["masked"]
    # Transplanted arm must NOT carry the true cutoff date (it uses a placebo).
    assert _PROBE.cutoff_date not in arms["transplanted"]
    assert "As of" in arms["transplanted"]


def test_leakage_score_numeric_and_substring_paths():
    full = leakage_score(
        "ExampleCorp was placed into FDIC receivership on March 10, 2023.", _PROBE.outcome
    )
    assert full == 1.0
    # Numeric overlap path: response carries only one of the outcome's two numbers
    # (10 and 2023), so the score is a true fraction.
    partial = leakage_score("It happened around the 10th.", _PROBE.outcome)
    assert 0.0 < partial < 1.0
    none = leakage_score("I have no view on this.", _PROBE.outcome)
    assert none == 0.0


@pytest.mark.asyncio
async def test_jsonl_roundtrip_preserves_metrics(tmp_path):
    client = _DateTriggeredClient(_PROBE)
    rows = await run_hindsight_audit(client=client, probes=[_PROBE], n_trials=1)
    out = tmp_path / "model.hindsight.jsonl"
    write_audit_jsonl(str(out), rows)
    restored = list(read_audit_jsonl(out))
    assert len(restored) == 1
    assert isinstance(restored[0], HindsightAuditRow)
    assert restored[0].trigger_strength == rows[0].trigger_strength
    assert restored[0].probe_id == _PROBE.id


@pytest.mark.asyncio
async def test_default_probes_are_well_formed():
    # The built-in panel must build a valid four-arm matrix for every probe.
    for probe in DEFAULT_PROBES:
        arms = build_arms(probe)
        assert probe.cutoff_date not in arms["masked"]
        assert probe.cutoff_date not in arms["transplanted"]
