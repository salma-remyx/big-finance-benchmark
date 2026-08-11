"""Integration tests for task evolution.

These exercise the Explorer -> Formalizer -> Challenger pipeline through the repo's
existing public surface: the evolved output must be a `DatasetItem` that round-trips
through the same JSONL schema the eval pipeline loads, and it must run through the
existing `run_question` agent loop unchanged. Pure self-tests of the new module live
alongside, but the integration claim rests on the non-new-module imports below.
"""

from __future__ import annotations

import pytest

from big_finance_harness.agent import run_question
from big_finance_harness.models.base import ModelClient, ThinkingLevel
from big_finance_harness.task_evolution import Checkpoint, CheckpointDAG, evolve_item
from big_finance_harness.tools.final_answer import FinalAnswerTool
from big_finance_harness.types import (
    DatasetItem,
    Message,
    ModelResponse,
    RubricLine,
    ToolSpec,
    ToolUseBlock,
)


class _ScriptedTextClient(ModelClient):
    """Returns queued raw text bodies — one per pipeline phase call."""

    snapshot = "anthropic:claude-test-2026-01-01"

    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self.calls = 0

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float | None = None,
        thinking: ThinkingLevel = "off",
        max_output_tokens: int = 4096,
    ) -> ModelResponse:
        text = self._texts[self.calls]
        self.calls += 1
        return ModelResponse(
            text=text, tool_calls=[], stop_reason="end_turn", prompt_tokens=5, completion_tokens=5
        )


class _ScriptedAgentClient(ModelClient):
    snapshot = "anthropic:claude-test-2026-01-01"

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self.calls = 0

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float | None = None,
        thinking: ThinkingLevel = "off",
        max_output_tokens: int = 4096,
    ) -> ModelResponse:
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


def _simple_item() -> DatasetItem:
    return DatasetItem(
        id="bf-test-0001",
        query="What was Michael Steiner's weighted average EVI sale price in December 2016?",
        reference_answer="$14.21",
        rubric=[
            RubricLine(text="Records 12/06/2016 shares sold = 32,476.", points=1),
            RubricLine(text="Records 12/06/2016 sale price = $13.53.", points=1),
            RubricLine(text="Calculates weighted average = $14.21.", points=10),
        ],
        sources=["https://www.sec.gov/cgi-bin/browse-edgar"],
    )


def _phase_responses() -> list[str]:
    explorer = (
        '{"related_points": ["Steiner Form 4 filings for Dec 2016", '
        '"EVI 10-K filed 2017", "weighted average = sum(price*shares)/sum(shares)"], '
        '"summary": "Need per-transaction Dec 2016 dispositions."}'
    )
    formalizer = (
        '{"evolved_query": "Reconstruct the Dec 2016 EVI disposal schedule for Steiner from '
        'Form 4 filings and compute the share-weighted average sale price.", '
        '"checkpoints": ['
        '{"text": "Locate Steiner Form 4 filings for Dec 2016.", "depends_on": [], "base_points": 1}, '
        '{"text": "Record each Dec 2016 transaction date, shares, and price.", "depends_on": [0], "base_points": 1}, '
        '{"text": "Cross-check transaction totals against the 10-K.", "depends_on": [1], "base_points": 1}, '
        '{"text": "Compute the share-weighted average sale price.", "depends_on": [1], "base_points": 3}, '
        '{"text": "Round the final answer to the nearest cent.", "depends_on": [3], "base_points": 1}'
        "]}"
    )
    challenger = (
        '{"constraints": ["Cite the Form 4 accession number for each transaction.", '
        '"Show the weighted-average arithmetic."], '
        '"reference_framing": "share-weighted", "weight_overrides": {"3": 5}}'
    )
    return [explorer, formalizer, challenger]


@pytest.mark.asyncio
async def test_evolve_item_produces_pipeline_compatible_datasetitem():
    """The evolved item is a first-class DatasetItem: same JSONL schema, no change."""
    client = _ScriptedTextClient(_phase_responses())
    evolved = await evolve_item(item=_simple_item(), client=client)

    # Three phases fired, in order.
    assert [s.phase for s in evolved.evolution] == ["explore", "formalize", "challenge"]
    assert evolved.source_id == "bf-test-0001"
    assert evolved.item.id == "bf-test-0001.evo"

    # Round-trips through the exact loader path scripts/run_eval_set._load_dataset uses.
    reloaded = DatasetItem.model_validate_json(evolved.item.model_dump_json())
    assert reloaded == evolved.item

    # Rubric is the DAG reduced to RubricLine, structure-weighted: the synthesis
    # checkpoint (idx 3, up-weighted by the Challenger to base 5, depth 2 => 7)
    # outweighs the shallow lookup roots.
    assert evolved.item.rubric == evolved.dag.to_rubric()
    assert [r.points for r in evolved.item.rubric] == [1, 2, 3, 7, 4]
    assert max(r.points for r in evolved.item.rubric) == 7  # synthesis node heaviest

    # The deep-research query carries the Challenger's verification constraints.
    assert "Constraints:" in evolved.item.query
    assert "share-weighted" in evolved.item.reference_answer


@pytest.mark.asyncio
async def test_evolved_item_runs_through_existing_agent_loop():
    """The evolved DatasetItem drops into the existing per-item eval entry unchanged."""
    evolved = await evolve_item(item=_simple_item(), client=_ScriptedTextClient(_phase_responses()))

    # The eval pipeline calls run_question once per item; the evolved item must work there.
    agent_client = _ScriptedAgentClient(
        [
            ModelResponse(
                text="The weighted average sale price was $14.21.",
                tool_calls=[ToolUseBlock(id="c1", name="final_answer", input={"answer": "$14.21"})],
                stop_reason="tool_use",
                prompt_tokens=10,
                completion_tokens=6,
            )
        ]
    )
    record = await run_question(
        question_id=evolved.item.id,
        question=evolved.item.query,
        reference_answer=evolved.item.reference_answer,
        client=agent_client,
        tools=[FinalAnswerTool()],
        system_prompt="test",
        max_steps=3,
    )
    assert record.stop_reason == "final_answer"
    assert record.final_answer == "$14.21"
    assert record.question_id == evolved.item.id


@pytest.mark.asyncio
async def test_evolve_item_degrades_gracefully_on_unparseable_phases():
    """Bad model output must not crash the pipeline — it falls back to the source rubric."""
    client = _ScriptedTextClient(["no json here", "also not json", "still not json"])
    evolved = await evolve_item(item=_simple_item(), client=client)

    # Source answer preserved verbatim (no Challenger framing to append).
    assert evolved.item.reference_answer == "$14.21"
    # Linear fallback DAG: each source line chains to the previous, so structure
    # weighting (base + depth) strictly increases along the chain. The 10-point
    # synthesis line keeps its base weight (10 + depth 2 = 12), proving the
    # fallback preserves the source rubric's emphasis while still adding structure.
    assert [r.points for r in evolved.item.rubric] == [1, 2, 12]
    assert evolved.item.id == "bf-test-0001.evo"
    # Round-trips as a DatasetItem just like the happy path.
    assert DatasetItem.model_validate_json(evolved.item.model_dump_json()) == evolved.item


def test_checkpoint_dag_structure_weighting():
    """Diamond DAG: the synthesis root of the diamond earns depth-weighted points."""
    dag = CheckpointDAG(
        checkpoints=[
            Checkpoint(text="fetch A", base_points=1, depends_on=[]),  # depth 0 -> 1
            Checkpoint(text="fetch B", base_points=1, depends_on=[]),  # depth 0 -> 1
            Checkpoint(text="join A+B", base_points=2, depends_on=[0, 1]),  # depth 1 -> 3
            Checkpoint(text="finalize", base_points=1, depends_on=[2]),  # depth 2 -> 3
        ]
    )
    assert [dag.depth(i) for i in range(4)] == [0, 0, 1, 2]
    assert [r.points for r in dag.to_rubric()] == [1, 1, 3, 3]
