"""Tests for the A^2E-style behavioral audit.

Two layers:
  - Pure unit tests of `audit_run` over hand-built `RunRecord` traces. These pin the
    four dimension formulas (efficiency / tool use / planning / error recovery).
  - An integration test through `grader.grade(audit=True)` (mocking the judge call)
    proving the audit is wired into the existing grading entry point.
"""

from __future__ import annotations

import json

import pytest

from big_finance_harness import grader as grader_module
from big_finance_harness.agent_audit import audit_run, audit_trace_file, read_audits
from big_finance_harness.grader import grade
from big_finance_harness.trace import TraceWriter
from big_finance_harness.types import (
    DatasetItem,
    RubricLine,
    RunRecord,
    StepRecord,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=name, input_schema={"type": "object"})


def _step(
    idx: int,
    *,
    calls: list[ToolUseBlock] | None = None,
    errors: list[bool] | None = None,
    prompt: int = 100,
    completion: int = 20,
) -> StepRecord:
    calls = calls or []
    results = [
        ToolResultBlock(
            tool_use_id=tc.id,
            content="ok" if not (errors and i < len(errors) and errors[i]) else "boom",
            is_error=bool(errors and i < len(errors) and errors[i]),
        )
        for i, tc in enumerate(calls)
    ]
    return StepRecord(
        step=idx,
        assistant_text=f"step {idx}",
        tool_calls=calls,
        tool_results=results,
        prompt_tokens=prompt,
        completion_tokens=completion,
        wallclock_seconds=1.0,
    )


def _run(
    steps: list[StepRecord],
    *,
    stop_reason: str = "final_answer",
    final_answer: str | None = "$1.0",
    max_steps: int = 10,
    num_retries: int = 0,
    tool_specs: list[ToolSpec] | None = None,
) -> RunRecord:
    return RunRecord(
        question_id="bf-audit-001",
        question="q",
        reference_answer="$1.0",
        model="anthropic:claude-test",
        harness_version="1.0.0",
        thinking="off",
        max_steps=max_steps,
        steps=steps,
        final_answer=final_answer,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        total_prompt_tokens=sum(s.prompt_tokens for s in steps),
        total_completion_tokens=sum(s.completion_tokens for s in steps),
        total_wallclock_seconds=float(len(steps)),
        num_retries=num_retries,
        tool_specs=tool_specs or [],
        started_at="2026-08-11T00:00:00+00:00",
        completed_at="2026-08-11T00:00:01+00:00",
    )


def _call(name: str, i: int = 0) -> ToolUseBlock:
    return ToolUseBlock(id=f"t{i}", name=name, input={})


def test_clean_converged_run_scores_high_everywhere():
    run = _run(
        [_step(0, calls=[_call("edgar_search", 0), _call("python_exec", 1)])],
        tool_specs=[
            _spec("edgar_search"),
            _spec("python_exec"),
            _spec("web_search"),
            _spec("fetch_url"),
        ],
    )
    audit = audit_run(run)

    assert audit.execution_efficiency.converged is True
    # 1 step of a 10-step budget answered => nearly full efficiency.
    assert audit.execution_efficiency.score == pytest.approx(0.9)
    assert audit.tool_use.n_tool_errors == 0
    assert audit.tool_use.tool_error_rate == 0.0
    assert audit.tool_use.score == 1.0
    assert audit.tool_use.distinct_tools_used == 2
    assert audit.tool_use.tool_diversity_ratio == pytest.approx(0.5)
    assert audit.task_planning.score == 1.0
    assert audit.error_recovery.score == 1.0
    assert 0.0 <= audit.overall_score <= 1.0


def test_tool_error_rate_drives_tool_use_score_down():
    # 2 calls, 1 error => error_rate 0.5 => tool_use score 0.5.
    run = _run(
        [_step(0, calls=[_call("edgar_search", 0)], errors=[False]),
         _step(1, calls=[_call("edgar_search", 1)], errors=[True])],
        stop_reason="max_steps",
        final_answer=None,
    )
    audit = audit_run(run)
    assert audit.tool_use.n_tool_calls == 2
    assert audit.tool_use.n_tool_errors == 1
    assert audit.tool_use.tool_error_rate == pytest.approx(0.5)
    assert audit.tool_use.score == pytest.approx(0.5)
    # Hit the step budget without an answer: planning 0.25, efficiency 0.0 (not converged).
    assert audit.task_planning.score == pytest.approx(0.25)
    assert audit.execution_efficiency.score == 0.0


def test_error_recovery_when_converging_despite_errors():
    # Step 0 errors, step 1 recovers and the run still produces a final answer.
    run = _run(
        [_step(0, calls=[_call("edgar_search", 0)], errors=[True]),
         _step(1, calls=[_call("edgar_search", 1)], errors=[False])],
        stop_reason="final_answer",
        final_answer="$1.0",
        num_retries=2,
    )
    audit = audit_run(run)
    assert audit.error_recovery.n_tool_errors == 1
    assert audit.error_recovery.n_retries == 2
    assert audit.error_recovery.recovered_despite_errors is True
    assert audit.error_recovery.ended_in_error is False
    assert audit.error_recovery.score == 1.0


def test_error_recovery_partial_when_run_gives_up_after_error():
    # Step 0 errors and is the last step; run does NOT converge => partial recovery.
    run = _run(
        [_step(0, calls=[_call("edgar_search", 0)], errors=[True])],
        stop_reason="max_steps",
        final_answer=None,
    )
    audit = audit_run(run)
    assert audit.error_recovery.recovered_despite_errors is False
    # Only error step had no following step => 0 share.
    assert audit.error_recovery.score == 0.0


def test_crashed_run_scores_zero_on_planning_and_marks_ended_in_error():
    run = _run([_step(0, calls=[_call("edgar_search", 0)])], stop_reason="error", final_answer=None)
    audit = audit_run(run)
    assert audit.task_planning.score == 0.0
    assert audit.error_recovery.ended_in_error is True


def test_audit_trace_file_round_trip(tmp_path):
    """audit_trace_file consumes the harness's traces JSONL (via TraceWriter) and emits
    one RunAudit per line — the self-contained trace-scoring pipeline over run output."""
    traces_path = tmp_path / "m.traces.jsonl"
    audit_path = tmp_path / "m.audit.jsonl"
    writer = TraceWriter(traces_path)
    writer.write(
        _run([_step(0, calls=[_call("edgar_search", 0)])], tool_specs=[_spec("edgar_search")])
    )
    writer.write(
        _run(
            [_step(0, calls=[_call("edgar_search", 0)], errors=[True])],
            stop_reason="error",
            final_answer=None,
            max_steps=10,
        )
    )

    summary = audit_trace_file(traces_path, audit_path)
    assert summary["n_audited"] == 2
    assert 0.0 <= summary["mean_overall_score"] <= 1.0

    audits = list(read_audits(traces_path))
    assert len(audits) == 2
    assert audits[0].task_planning.stop_reason == "final_answer"
    assert audits[1].error_recovery.ended_in_error is True
    # The written file round-trips through RunAudit too.
    assert audit_path.read_text().count("\n") == 2


@pytest.mark.asyncio
async def test_grade_attaches_audit_when_requested(monkeypatch):
    """grade(audit=True) wires audit_run into the existing grading entry point."""
    judge_payload = {
        "final_answer_correct": True,
        "rubric": [{"index": 1, "satisfied": True, "explanation": "ok"}],
    }

    class _Usage:
        prompt_tokens = 100
        completion_tokens = 10

    class _Resp:
        usage = _Usage()
        _hidden_params = {"response_cost": 0.001}

        class _Choice:
            class _Message:
                content = json.dumps(judge_payload)

            message = _Message()

        choices = [_Choice()]

    async def fake_acompletion(**_kwargs):
        return _Resp()

    monkeypatch.setattr(grader_module.litellm, "acompletion", fake_acompletion)

    run = _run([_step(0, calls=[_call("edgar_search", 0)])], tool_specs=[_spec("edgar_search")])
    item = DatasetItem(
        id="bf-audit-001",
        query="q",
        reference_answer="$1.0",
        rubric=[RubricLine(text="step", points=1)],
    )

    audited = await grade(
        run=run, item=item, judge_model_id="vertex:gemini-3.1-pro-preview", audit=True
    )
    assert audited.audit is not None
    assert set(audited.audit) == {
        "question_id",
        "model",
        "execution_efficiency",
        "tool_use",
        "task_planning",
        "error_recovery",
        "overall_score",
    }
    assert audited.audit["task_planning"]["stop_reason"] == "final_answer"
    assert 0.0 <= audited.audit["overall_score"] <= 1.0

    # Default (audit=False) leaves the field unset — existing callers unaffected.
    plain = await grade(run=run, item=item, judge_model_id="vertex:gemini-3.1-pro-preview")
    assert plain.audit is None
