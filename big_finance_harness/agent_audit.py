"""Behavioral audit of an agent run trace.

This is an adapted port of the multidimensional trace-scoring stage from
**A^2E (Agent Auditing Engine)** (arxiv:2608.07346). A^2E evaluates agent
*harnesses* along four axes that go beyond bare correctness:

  - **Execution efficiency** -- how parsimoniously the harness reaches an answer.
  - **Tool use**            -- the quality and diversity of tool interactions.
  - **Task planning**       -- whether the trajectory terminates in a coherent answer.
  - **Error recovery**      -- bounce-back from tool/API failures.

A^2E ships a bespoke Monitor + Agent Task Protocol (ATP) + benchmark suite to
capture and score those dimensions. This repo already captures a standardized
trace (`RunRecord` / `StepRecord`) via `run_question`, so the Monitor and ATP
are unnecessary here: every dimension reduces to a parameter-free computation
over fields already on the trace (steps, tool calls, tool results, stop_reason,
retries, tokens, wallclock, cost). That substitution is the only adaptation --
the four-dimensional scoring itself is preserved at full fidelity.

The scoring is deterministic and side-effect free: `audit_run(run)` returns a
`RunAudit`. `grade(audit=True)` in `big_finance_harness.grader` attaches the
result to the `GradedRun` so behavioral metrics flow out next to correctness;
`audit_trace_file` emits one `RunAudit` per line for offline analysis.

Each dimension exposes its raw counters plus a transparent ``score`` in [0, 1];
``overall_score`` is the unweighted mean of the four. The formulas are
documented inline rather than learned, so a reader can audit the auditor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

from big_finance_harness.trace import read_traces
from big_finance_harness.types import RunRecord, StepRecord


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class ExecutionEfficiency(BaseModel):
    """How much of the run's budget was spent to reach an answer."""

    n_steps: int
    max_steps: int
    step_budget_used_ratio: float  # n_steps / max_steps
    total_tokens: int
    tokens_per_step: float
    wallclock_seconds: float
    cost_usd: float | None
    converged: bool  # stop_reason == "final_answer"
    # Converging in fewer steps relative to the cap scores higher; not converging scores 0.
    score: float


class ToolUseMetrics(BaseModel):
    """Diversity and failure rate of tool interactions."""

    n_tool_calls: int
    n_tool_errors: int
    tool_error_rate: float  # n_tool_errors / n_tool_calls
    distinct_tools_used: int
    n_tools_available: int
    tool_diversity_ratio: float  # distinct_tools_used / n_tools_available
    tool_calls_per_step: float
    # Lower tool-error rate scores higher; no calls at all scores 0.
    score: float


class TaskPlanning(BaseModel):
    """Whether the trajectory terminated in a coherent conclusion."""

    stop_reason: str
    final_answer_produced: bool
    steps_before_first_tool_call: int
    # Clean convergence (final_answer) scores 1.0; a no-tool give-up 0.5; hitting any
    # budget (max_steps / context / token) 0.25; a crash (error) 0.0.
    score: float


class ErrorRecovery(BaseModel):
    """Bounce-back from tool / retry failures."""

    n_tool_errors: int
    n_retries: int
    ended_in_error: bool  # stop_reason == "error"
    recovered_despite_errors: bool  # converged despite hitting >=1 tool error
    # No errors => 1.0 (nothing to recover from). Converged despite errors => 1.0.
    # Otherwise the share of error-bearing steps that were followed by another step.
    score: float


class RunAudit(BaseModel):
    """Multidimensional behavioral audit of a single `RunRecord`."""

    question_id: str
    model: str
    execution_efficiency: ExecutionEfficiency
    tool_use: ToolUseMetrics
    task_planning: TaskPlanning
    error_recovery: ErrorRecovery
    overall_score: float  # unweighted mean of the four dimension scores


def _execution_efficiency(run: RunRecord) -> ExecutionEfficiency:
    n_steps = len(run.steps)
    ratio = (n_steps / run.max_steps) if run.max_steps else 0.0
    converged = run.stop_reason == "final_answer"
    total_tokens = run.total_prompt_tokens + run.total_completion_tokens
    tokens_per_step = (total_tokens / n_steps) if n_steps else 0.0
    score = _clamp01(1.0 - ratio) if converged else 0.0
    return ExecutionEfficiency(
        n_steps=n_steps,
        max_steps=run.max_steps,
        step_budget_used_ratio=ratio,
        total_tokens=total_tokens,
        tokens_per_step=tokens_per_step,
        wallclock_seconds=run.total_wallclock_seconds,
        cost_usd=run.cost_usd,
        converged=converged,
        score=score,
    )


def _count_tool_calls(steps: list[StepRecord]) -> tuple[int, int, set[str], int]:
    """Return (n_tool_calls, n_tool_errors, tools_used, steps_before_first_tool_call)."""
    n_calls = 0
    n_errors = 0
    used: set[str] = set()
    steps_before_first_tool = 0
    seen_tool = False
    for s in steps:
        calls_here = len(s.tool_calls)
        n_calls += calls_here
        for tc in s.tool_calls:
            used.add(tc.name)
        n_errors += sum(1 for tr in s.tool_results if tr.is_error)
        if not seen_tool:
            if calls_here > 0:
                seen_tool = True
            else:
                steps_before_first_tool += 1
    return n_calls, n_errors, used, steps_before_first_tool


def _tool_use(run: RunRecord) -> ToolUseMetrics:
    n_calls, n_errors, used, _ = _count_tool_calls(run.steps)
    n_available = len(run.tool_specs)
    error_rate = (n_errors / n_calls) if n_calls else 0.0
    diversity = (len(used) / n_available) if n_available else 0.0
    per_step = (n_calls / len(run.steps)) if run.steps else 0.0
    score = _clamp01(1.0 - error_rate) if n_calls else 0.0
    return ToolUseMetrics(
        n_tool_calls=n_calls,
        n_tool_errors=n_errors,
        tool_error_rate=error_rate,
        distinct_tools_used=len(used),
        n_tools_available=n_available,
        tool_diversity_ratio=diversity,
        tool_calls_per_step=per_step,
        score=score,
    )


_PLANNING_SCORE: dict[str, float] = {
    "final_answer": 1.0,
    "no_tool_call": 0.5,
    "max_steps": 0.25,
    "context_exceeded": 0.25,
    "token_budget": 0.25,
    "error": 0.0,
}


def _task_planning(run: RunRecord) -> TaskPlanning:
    _, _, _, steps_before = _count_tool_calls(run.steps)
    return TaskPlanning(
        stop_reason=run.stop_reason,
        final_answer_produced=run.final_answer is not None,
        steps_before_first_tool_call=steps_before,
        score=_PLANNING_SCORE.get(run.stop_reason, 0.0),
    )


def _error_recovery(run: RunRecord) -> ErrorRecovery:
    _, n_errors, _, _ = _count_tool_calls(run.steps)
    converged = run.stop_reason == "final_answer"
    ended_in_error = run.stop_reason == "error"
    recovered = n_errors > 0 and converged
    error_step_idxs = [
        i for i, s in enumerate(run.steps) if any(tr.is_error for tr in s.tool_results)
    ]
    if not error_step_idxs:
        score = 1.0
    elif recovered:
        score = 1.0
    else:
        # Share of error steps that were followed by at least one more step.
        followed = sum(1 for i in error_step_idxs if i + 1 < len(run.steps))
        score = followed / len(error_step_idxs)
    return ErrorRecovery(
        n_tool_errors=n_errors,
        n_retries=run.num_retries,
        ended_in_error=ended_in_error,
        recovered_despite_errors=recovered,
        score=score,
    )


def audit_run(run: RunRecord) -> RunAudit:
    """Score one run trace across A^2E's four behavioral dimensions."""
    eff = _execution_efficiency(run)
    tools = _tool_use(run)
    plan = _task_planning(run)
    rec = _error_recovery(run)
    overall = (eff.score + tools.score + plan.score + rec.score) / 4.0
    return RunAudit(
        question_id=run.question_id,
        model=run.model,
        execution_efficiency=eff,
        tool_use=tools,
        task_planning=plan,
        error_recovery=rec,
        overall_score=overall,
    )


def read_audits(path: str | Path) -> Iterator[RunAudit]:
    """Read a traces JSONL file and yield one ``RunAudit`` per trace."""
    for run in read_traces(path):
        yield audit_run(run)


def audit_trace_file(traces_path: str | Path, audit_path: str | Path) -> dict[str, object]:
    """Audit every trace in ``traces_path``, writing one ``RunAudit`` per line to
    ``audit_path``. Returns a small summary dict (for manifest / logging)."""
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    score_sum = 0.0
    with audit_path.open("w", encoding="utf-8") as f:
        for audit in read_audits(traces_path):
            f.write(audit.model_dump_json() + "\n")
            n += 1
            score_sum += audit.overall_score
    return {
        "traces_path": str(traces_path),
        "audit_path": str(audit_path),
        "n_audited": n,
        "mean_overall_score": round(score_sum / n, 4) if n else 0.0,
    }
