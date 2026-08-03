"""Integration test for the opt-in grounding-check tool (adapted from MiniCheck).

Drives the new capability through the existing public tool factory
(``big_finance_harness.tools.default_tools``) and the agent's tool-dispatch helper
(``big_finance_harness.agent._dispatch_tool``), proving it is reachable from the
established agent surface while the canonical default inventory is unchanged.
"""

from __future__ import annotations

import json

import pytest

from big_finance_harness.agent import _dispatch_tool
from big_finance_harness.tools import default_tools
from big_finance_harness.tools.grounding_check import GroundingCheckTool
from big_finance_harness.types import ToolUseBlock

_DOC = (
    "Apple reported total revenue of $383.3 billion for fiscal year 2023, up from "
    "$394.3 billion in 2022. Net income was $96.9 billion. The board did not "
    "authorize a dividend increase this quarter."
)


def _grounding_tool() -> GroundingCheckTool:
    return next(
        t for t in default_tools(include_grounding_check=True) if t.name == "grounding_check"
    )


def _names(tools):
    return {t.name for t in tools}


def test_default_inventory_is_unchanged():
    # The canonical model-facing surface must not gain the grounding tool by default.
    assert "grounding_check" not in _names(default_tools())
    assert [t.name for t in default_tools()] == [
        "web_search",
        "edgar_search",
        "fetch_url",
        "python_exec",
        "final_answer",
    ]


def test_opt_in_flag_adds_grounding_check():
    tools = default_tools(include_grounding_check=True)
    assert "grounding_check" in _names(tools)
    assert len(tools) == 6


@pytest.mark.asyncio
async def test_dispatch_supported_claim_cites_evidence():
    tool = _grounding_tool()
    call = ToolUseBlock(
        id="call_supported",
        name="grounding_check",
        input={
            "claim": "Total revenue was $383.3 billion for fiscal year 2023.",
            "document": _DOC,
        },
    )
    result = await _dispatch_tool(tool, call)
    assert not result.is_error
    parsed = json.loads(result.content)
    assert parsed["label"] == "supported"
    assert parsed["evidence"], "supported claim should cite a backing sentence"
    assert "383.3" in parsed["evidence"][0]
    assert "revenue" in parsed["evidence"][0]


@pytest.mark.asyncio
async def test_dispatch_partially_supported_claim():
    tool = _grounding_tool()
    call = ToolUseBlock(
        id="call_partial",
        name="grounding_check",
        input={
            "claim": "Net income was $96.9 billion according to analysts, a record quarterly result.",
            "document": _DOC,
        },
    )
    result = await _dispatch_tool(tool, call)
    assert not result.is_error
    parsed = json.loads(result.content)
    assert parsed["label"] == "partially_supported"


@pytest.mark.asyncio
async def test_dispatch_invented_number_is_unsupported():
    # Same wording, but the figure does not appear anywhere in the document.
    tool = GroundingCheckTool()
    call = ToolUseBlock(
        id="call_invented",
        name="grounding_check",
        input={
            "claim": "Apple's 2023 net income was $150 billion.",
            "document": _DOC,
        },
    )
    result = await _dispatch_tool(tool, call)
    assert not result.is_error
    parsed = json.loads(result.content)
    assert parsed["label"] == "unsupported"


@pytest.mark.asyncio
async def test_dispatch_missing_document_surfaces_tool_error():
    # The agent loop converts ToolError into an is_error tool_result; it must not raise.
    tool = GroundingCheckTool()
    call = ToolUseBlock(
        id="call_bad",
        name="grounding_check",
        input={"claim": "something"},
    )
    result = await _dispatch_tool(tool, call)
    assert result.is_error
    assert "document" in result.content
