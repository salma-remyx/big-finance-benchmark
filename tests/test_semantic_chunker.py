"""Integration tests for LumberChunker-style semantic segmentation in fetch_url.

The fetch_url tool's query path chunks a document then BM25-ranks the chunks.
These tests exercise the opt-in wiring that swaps the fixed paragraph chunker
for the LLM-driven `semantic_chunk` segmenter, plus a few self-tests of the
segmenter itself. The integration test goes through the public FetchUrlTool
surface (a non-new module) so it proves the call-site edit actually invokes the
new code.
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from big_finance_harness.models.base import ModelClient, ThinkingLevel
from big_finance_harness.semantic_chunker import semantic_chunk
from big_finance_harness.tools.fetch_url import FetchUrlTool
from big_finance_harness.types import Message, ModelResponse, ToolSpec


class _ScriptedClient(ModelClient):
    """Stub model that always returns a canned integer and counts its calls."""

    snapshot = "anthropic:claude-test-2026-01-01"

    def __init__(self, answer: str = "2") -> None:
        self._answer = answer
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
        self.calls += 1
        return ModelResponse(
            text=self._answer,
            tool_calls=[],
            stop_reason="end_turn",
            prompt_tokens=0,
            completion_tokens=0,
        )


class _BoomClient(ModelClient):
    """A client that fails the test if it is ever called."""

    snapshot = "anthropic:claude-test-2026-01-01"

    async def chat(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("model client must not be called when semantic_chunking is off")


SAMPLE_HTML = """\
<html><body>
<p>Revenue grew across the segment. Services revenue hit a new high. Products held flat year over year.</p>
<p>The board declared a dividend. The payout ratio increased. The record date is set for next month.</p>
</body></html>
"""


@pytest.mark.asyncio
async def test_semantic_chunk_merges_same_topic_passages():
    # Six passages (one sentence each); the model always says the first two
    # share the first passage's topic, so they merge pairwise into 3 chunks.
    text = "Topic A one. Topic A two. Topic A three. Topic B one. Topic B two. Topic B three."
    client = _ScriptedClient(answer="2")
    chunks = await semantic_chunk(text, client, sentences_per_passage=1, group_size=6)

    assert client.calls > 0
    assert len(chunks) == 3
    # Each chunk merges exactly two passages (count=2); passage boundaries survive
    # as substrings regardless of the join character used between them.
    assert "Topic A one." in chunks[0]
    assert "Topic A two." in chunks[0]
    assert "Topic A one." not in chunks[1]
    assert "Topic B two." in chunks[2]


@pytest.mark.asyncio
async def test_semantic_chunk_unparseable_answer_defaults_to_single_passage():
    # A garbage answer must never merge unrelated passages: each becomes its own chunk.
    client = _ScriptedClient(answer="huh??")
    chunks = await semantic_chunk(
        "First sentence. Second sentence. Third sentence.",
        client,
        sentences_per_passage=1,
        group_size=3,
    )
    assert len(chunks) == 3


@pytest.mark.asyncio
async def test_semantic_chunk_respects_llm_call_cap():
    # max_llm_calls=0 flushes every passage verbatim without any LLM call.
    client = _ScriptedClient(answer="2")
    chunks = await semantic_chunk(
        "One. Two. Three. Four.",
        client,
        sentences_per_passage=1,
        group_size=2,
        max_llm_calls=0,
    )
    assert client.calls == 0
    assert chunks == ["One.", "Two.", "Three.", "Four."]


@pytest.mark.asyncio
async def test_fetch_url_uses_semantic_chunker_when_enabled(httpx_mock: HTTPXMock):
    """The call-site edit must invoke the new segmenter on the query path."""
    httpx_mock.add_response(
        url="https://example.com/filing",
        text=SAMPLE_HTML,
        headers={"content-type": "text/html"},
    )
    client = _ScriptedClient(answer="2")
    tool = FetchUrlTool(model_client=client, semantic_chunking=True, retrieve_k=2)
    out = await tool.run({"url": "https://example.com/filing", "query": "dividend"})

    # The LLM was actually consulted (segmenter ran), and BM25 still returns ranked chunks.
    assert client.calls > 0
    assert "chunk 1" in out
    assert "payout ratio" in out


@pytest.mark.asyncio
async def test_fetch_url_does_not_call_model_when_chunking_disabled(httpx_mock: HTTPXMock):
    """Default behavior is unchanged: no model client is consulted on the query path."""
    httpx_mock.add_response(
        url="https://example.com/filing",
        text=SAMPLE_HTML,
        headers={"content-type": "text/html"},
    )
    # A model client is attached but semantic_chunking is off -> it must not run.
    tool = FetchUrlTool(model_client=_BoomClient())
    out = await tool.run({"url": "https://example.com/filing", "query": "dividend"})
    assert "chunk 1" in out
