"""Integration tests for the task-level LLM router.

These exercise `TaskRouterClient` through the existing `run_question` seam (the same
`client: ModelClient` injection real evals use) plus the delayed-reward learning loop.
"""

from __future__ import annotations

import pytest

from big_finance_harness.agent import run_question
from big_finance_harness.models.base import ModelClient, ThinkingLevel
from big_finance_harness.models.task_router import TaskRouterClient, _Arm
from big_finance_harness.tools.base import Tool
from big_finance_harness.tools.final_answer import FinalAnswerTool
from big_finance_harness.types import Message, ModelResponse, TextBlock, ToolSpec, ToolUseBlock


class _ScriptedClient(ModelClient):
    """Stub backend that replays a fixed response list and counts how often it served."""

    def __init__(self, snapshot: str, responses: list[ModelResponse]):
        self.snapshot = snapshot
        self._responses = list(responses)
        self.calls = 0

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float | None = None,
        thinking: ThinkingLevel = "off",
        max_output_tokens: int = 65536,
    ) -> ModelResponse:
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


class _CalcTool(Tool):
    name = "calc"
    description = "Adds two numbers."
    input_schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    async def run(self, args):
        return str(args["a"] + args["b"])


def _resp(text: str = "ok", tool_calls: list[ToolUseBlock] | None = None) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=tool_calls or [],
        stop_reason="tool_use" if tool_calls else "end_turn",
        prompt_tokens=1,
        completion_tokens=1,
    )


def _two_arm_router(arm_a: ModelClient, arm_b: ModelClient, **kwargs) -> TaskRouterClient:
    return TaskRouterClient([_Arm("cheap", arm_a), _Arm("strong", arm_b)], **kwargs)


@pytest.mark.asyncio
async def test_router_drops_into_run_question_and_pins_one_arm():
    # Arm "cheap" (first) is selected at admission on the cold-start tie, then pinned.
    cheap = _ScriptedClient(
        "anthropic:claude-haiku-4-5-20251001",
        [
            _resp("thinking", [ToolUseBlock(id="c1", name="calc", input={"a": 2, "b": 2})]),
            _resp("done", [ToolUseBlock(id="c2", name="final_answer", input={"answer": "4"})]),
        ],
    )
    strong = _ScriptedClient("anthropic:claude-opus-4-7-20260416", [_resp()])

    router = _two_arm_router(cheap, strong)
    record = await run_question(
        question_id="q1",
        question="What is 2 + 2?",
        reference_answer="4",
        client=router,
        tools=[_CalcTool(), FinalAnswerTool()],
        system_prompt="test",
        max_steps=5,
    )

    assert record.stop_reason == "final_answer"
    assert record.final_answer == "4"
    assert len(record.steps) == 2
    # Per-task pinning: every step hit the same arm, the other arm never served.
    assert cheap.calls == 2
    assert strong.calls == 0
    assert router.selected_arm_for("What is 2 + 2?") == "cheap"
    # The active arm's snapshot propagates to the trace's `model` field.
    assert record.model == cheap.snapshot


@pytest.mark.asyncio
async def test_router_learns_from_delayed_terminal_reward():
    cheap = _ScriptedClient("anthropic:claude-haiku-4-5-20251001", [_resp()])
    strong = _ScriptedClient("anthropic:claude-opus-4-7-20260416", [_resp()])
    router = _two_arm_router(cheap, strong, alpha=1.0)

    q1 = "What is 2 + 2?"
    msg = [Message(role="user", content=[TextBlock(text=q1)])]
    await router.chat(system="", messages=msg, tools=[])

    # Delayed feedback: q1 was wrong and slow -> near-zero reward for "cheap". Updating
    # its covariance (not its mean) shrinks its confidence radius, so the under-trained
    # "strong" arm now carries the larger exploration bonus.
    router.feedback(q1, correct=False, latency_seconds=1000.0)

    q2 = "Compute Apple's fiscal 2024 gross margin from the 10-K."
    await router.chat(
        system="", messages=[Message(role="user", content=[TextBlock(text=q2)])], tools=[]
    )
    assert router.selected_arm_for(q2) == "strong"


def test_reward_is_accuracy_dominant_and_latency_aware():
    router = _two_arm_router(
        _ScriptedClient("a:a-2026-01-01", [_resp()]), _ScriptedClient("b:b-2026-01-01", [_resp()])
    )
    fast_correct = router.reward(True, 0.0)
    slow_correct = router.reward(True, 60.0)
    fast_wrong = router.reward(False, 0.0)
    slow_wrong = router.reward(False, 60.0)
    assert fast_correct == pytest.approx(1.0)
    assert slow_correct == pytest.approx(0.7)
    assert fast_wrong == pytest.approx(0.3)
    assert slow_wrong == pytest.approx(0.0)
    assert fast_correct > slow_correct > fast_wrong > slow_wrong


@pytest.mark.asyncio
async def test_router_policy_persists_across_instances(tmp_path):
    policy = tmp_path / "policy.json"
    cheap = _ScriptedClient("anthropic:claude-haiku-4-5-20251001", [_resp()])
    strong = _ScriptedClient("anthropic:claude-opus-4-7-20260416", [_resp()])

    router = _two_arm_router(cheap, strong, policy_path=policy)
    q = "What is 2 + 2?"
    await router.chat(
        system="", messages=[Message(role="user", content=[TextBlock(text=q)])], tools=[]
    )
    router.feedback(q, correct=True, latency_seconds=0.0)
    assert policy.exists()

    # A fresh router over the same arms loads the learned policy verbatim.
    reloaded = _two_arm_router(
        _ScriptedClient("anthropic:claude-haiku-4-5-20251001", [_resp()]),
        _ScriptedClient("anthropic:claude-opus-4-7-20260416", [_resp()]),
        policy_path=policy,
    )
    assert reloaded._bandit.to_dict() == router._bandit.to_dict()
