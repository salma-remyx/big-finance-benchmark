"""Integration tests for the evidence-grounding scorer.

Imports the repo's existing trace/dataset types (``big_finance_harness.types``)
and exercises the scorer against constructed ``RunRecord`` traces plus the
wiring CLI — proving the new module integrates with the harness's real types
rather than only self-testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from big_finance_harness.evidence_grounding import (
    extract_retrieved_passages,
    pairwise_accuracy,
    provenance_precision,
    score_evidence_grounding,
)
from big_finance_harness.types import (
    DatasetItem,
    RubricLine,
    RunRecord,
    StepRecord,
    ToolResultBlock,
    ToolUseBlock,
)
from scripts.eval_evidence_grounding import main as eval_main

# A gold supporting passage: the intended SEC-filing evidence (Acme's 2024
# revenue figure).
GOLD = "Acme Corporation total revenue 1234 million for fiscal year ended December 31 2024"
# A confusable same-filing excerpt the retriever might wrongly surface: a risk
# factor, not the revenue disclosure. Lexically distinct from GOLD so the
# token-overlap proxy discriminates the two.
HARD_NEG = (
    "Acme Corporation faces material competition and supply chain risks "
    "across its primary markets"
)
UNRELATED = "General market news and commentary unrelated to the filing."

_MODEL = "anthropic:claude-opus-4-7-20260416"


def _retrieval_step(
    step: int,
    pairs: list[tuple[str, str, str]],
    *,
    error_first: bool = False,
) -> StepRecord:
    """Build a step with retrieval tool calls/results: ``(tool_use_id, name, content)``.

    The tool_use_id links each ToolUseBlock to its ToolResultBlock, the same
    linkage the real agent loop produces.
    """
    calls = [ToolUseBlock(id=tid, name=name, input={"q": "x"}) for tid, name, _ in pairs]
    results = []
    for i, (tid, name, content) in enumerate(pairs):
        results.append(
            ToolResultBlock(
                tool_use_id=tid,
                content=content,
                is_error=bool(error_first and i == 0),
            )
        )
    return StepRecord(
        step=step,
        assistant_text="",
        tool_calls=calls,
        tool_results=results,
        prompt_tokens=1,
        completion_tokens=1,
        wallclock_seconds=0.1,
    )


def _build_run(
    question_id: str,
    *,
    retrieved: list[str],
    order: list[str] | None = None,
    include_final_answer: bool = True,
    error_first: bool = False,
) -> RunRecord:
    """Construct a run whose retrieved passages come from edgar/fetch/web tools.

    ``retrieved`` lists passage strings in retrieval order; each is emitted by a
    distinct tool so the extractor must respect both tool-name filtering and
    tool_use_id linkage.
    """
    tools = ["edgar_search", "fetch_url", "web_search"]
    order = order or retrieved
    pairs: list[tuple[str, str, str]] = [
        (f"r{i}", tools[i % len(tools)], passage) for i, passage in enumerate(order)
    ]
    steps = [_retrieval_step(0, pairs, error_first=error_first)]
    if include_final_answer:
        steps.append(
            StepRecord(
                step=1,
                assistant_text="done",
                tool_calls=[ToolUseBlock(id="fa", name="final_answer", input={"answer": "1234"})],
                tool_results=[ToolResultBlock(tool_use_id="fa", content="1234")],
                prompt_tokens=1,
                completion_tokens=1,
                wallclock_seconds=0.1,
            )
        )
    return RunRecord(
        question_id=question_id,
        question="What was Acme's 2024 revenue?",
        reference_answer="1234 million",
        model=_MODEL,
        harness_version="0.1.0",
        thinking="off",
        temperature=0.0,
        max_steps=30,
        steps=steps,
        final_answer="1234 million",
        stop_reason="final_answer",
        total_prompt_tokens=2,
        total_completion_tokens=2,
        total_wallclock_seconds=0.2,
        cost_usd=0.0,
        started_at="2026-04-30T00:00:00+00:00",
        completed_at="2026-04-30T00:00:01+00:00",
    )


def _item(question_id: str, sources: list[str] | None = None) -> DatasetItem:
    return DatasetItem(
        id=question_id,
        query="What was Acme's 2024 revenue?",
        reference_answer="1234 million",
        rubric=[RubricLine(text="state 2024 revenue", points=1)],
        sources=sources or [],
    )


def test_extract_retrieved_passages_filters_and_preserves_order():
    run = _build_run("q1", retrieved=[GOLD, HARD_NEG, UNRELATED])
    passages = extract_retrieved_passages(run)
    # final_answer content is excluded; the three retrieval results remain in order.
    assert passages == [GOLD, HARD_NEG, UNRELATED]


def test_extract_retrieved_passages_tool_filter_and_error_skip():
    run = _build_run("q1", retrieved=[GOLD, HARD_NEG, UNRELATED], error_first=True)
    only_edgar = extract_retrieved_passages(run, tool_names={"edgar_search"})
    # Only the first passage was emitted by edgar_search; it errored, so it's dropped.
    assert only_edgar == []
    all_valid = extract_retrieved_passages(run)
    assert all_valid == [HARD_NEG, UNRELATED]


def test_score_well_grounded_run():
    run = _build_run("q1", retrieved=[GOLD, HARD_NEG, UNRELATED])
    score = score_evidence_grounding(run, _item("q1", [GOLD]), hard_negatives=[HARD_NEG])
    assert score.n_retrieved == 3
    assert score.recall_at_k == 1.0  # gold covered by the first retrieval
    # One gold hit + one hard-negative hit among the two attributable passages.
    assert score.provenance_precision == pytest.approx(0.5)
    assert score.pairwise_accuracy == 1.0  # gold ranked above the negative
    assert GOLD in score.matched_gold
    assert HARD_NEG in score.matched_hard_negatives


def test_pairwise_accuracy_fails_when_negative_ranked_above_gold():
    run = _build_run("q1", retrieved=[HARD_NEG, UNRELATED, GOLD])
    acc = pairwise_accuracy([HARD_NEG, UNRELATED, GOLD], [GOLD], [HARD_NEG])
    assert acc == 0.0
    # Recall is still full — the gold passage IS retrieved within top-k.
    assert score_evidence_grounding(
        run, _item("q1", [GOLD]), hard_negatives=[HARD_NEG]
    ).recall_at_k == 1.0


def test_no_gold_evidence_scores_zero():
    run = _build_run("q1", retrieved=[UNRELATED])
    score = score_evidence_grounding(run, _item("q1", []))
    assert score.recall_at_k == 0.0
    assert score.provenance_precision == 0.0
    assert score.pairwise_accuracy == 0.0
    assert score.n_gold == 0


def test_provenance_precision_all_gold():
    # Only a gold-matching passage is retrieved: precision is perfect.
    assert provenance_precision([GOLD], [GOLD], [HARD_NEG]) == 1.0


def test_id_mismatch_raises():
    run = _build_run("q1", retrieved=[GOLD])
    with pytest.raises(ValueError, match="id mismatch"):
        score_evidence_grounding(run, _item("other", [GOLD]))


def test_cli_scores_traces_with_provenance_sidecar(tmp_path: Path):
    run = _build_run("q1", retrieved=[GOLD, HARD_NEG, UNRELATED])
    dataset_path = tmp_path / "ds.jsonl"
    dataset_path.write_text(_item("q1").model_dump_json() + "\n", encoding="utf-8")
    traces_path = tmp_path / "runs.traces.jsonl"
    traces_path.write_text(run.model_dump_json() + "\n", encoding="utf-8")
    provenance_path = tmp_path / "prov.jsonl"
    provenance_path.write_text(
        json.dumps({"id": "q1", "sources": [GOLD], "hard_negatives": [HARD_NEG]}) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "scores.jsonl"

    result = CliRunner().invoke(
        eval_main,
        [
            "--traces",
            str(traces_path),
            "--dataset",
            str(dataset_path),
            "--provenance",
            str(provenance_path),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "mean Recall@10" in result.output
    # The sidecar's gold passage overlays onto the item, so recall is perfect.
    assert "1.000" in result.output

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["question_id"] == "q1"
    assert records[0]["recall_at_k"] == 1.0
    assert records[0]["n_hard_negatives"] == 1
