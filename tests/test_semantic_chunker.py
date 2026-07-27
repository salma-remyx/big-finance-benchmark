"""Integration tests for LumberChunker-style semantic segmentation in fetch_url.

The fetch_url tool's query path chunks a document then BM25-ranks the chunks.
These tests exercise the opt-in wiring that swaps the fixed paragraph chunker
for the LLM-driven `semantic_chunk` segmenter, plus self-tests of the segmenter
itself. The mocked model answers with paragraph IDs in the reference's
'Answer: ID XXXX' format, and the tests assert the splits land at exactly those
mock-specified paragraphs. The integration test goes through the public
FetchUrlTool surface (a non-new module) so it proves the call-site edit
actually invokes the segmenter.
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from big_finance_harness.models.base import ModelClient, ThinkingLevel
from big_finance_harness.semantic_chunker import semantic_chunk
from big_finance_harness.tools.fetch_url import FetchUrlTool
from big_finance_harness.types import Message, ModelResponse, ToolSpec


class _ScriptedClient(ModelClient):
    """Stub model returning canned 'Answer: ID XXXX' responses, in order."""

    snapshot = "anthropic:claude-test-2026-01-01"

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
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
        answer = self._answers.pop(0) if self._answers else ""
        return ModelResponse(
            text=answer,
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


# Eight two-word paragraphs. ID-tagged each is 4 words ≈ 5 tokens, so with
# target_tokens=11 each accumulation holds out the third paragraph and prompts
# with a two-paragraph block.
PARAGRAPHS = [
    "Alpha one.",
    "Alpha two.",
    "Beta one.",
    "Beta two.",
    "Gamma one.",
    "Gamma two.",
    "Delta one.",
    "Delta two.",
]
TEXT = "\n\n".join(PARAGRAPHS)

SAMPLE_TEXT = (
    "Revenue grew across the segment.\n\n"
    "Services revenue hit a new high.\n\n"
    "Products held flat year over year.\n\n"
    "The board declared a dividend.\n\n"
    "The payout ratio increased.\n\n"
    "The record date is set for next month.\n\n"
    "The dividend yield leads the sector."
)


@pytest.mark.asyncio
async def test_semantic_chunk_splits_at_mock_named_ids():
    # The model names IDs 1, 2, 3 as split points, so each chunk boundary must
    # land exactly at the mock-specified paragraph.
    client = _ScriptedClient(["Answer: ID 1", "Answer: ID 2", "Answer: ID 3"])
    chunks = await semantic_chunk(TEXT, client, target_tokens=11)

    assert client.calls == 3
    assert chunks == [
        "Alpha one.",
        "Alpha two.",
        "Beta one.",
        "Beta two.\n\nGamma one.\n\nGamma two.\n\nDelta one.\n\nDelta two.",
    ]


@pytest.mark.asyncio
async def test_semantic_chunk_restarts_at_named_paragraph():
    # After a split, accumulation restarts at the named paragraph: it must
    # reappear at the start of the next prompt, never inside the just-emitted
    # chunk.
    seen_prompts: list[str] = []
    seen_systems: list[str] = []

    class _RecordingClient(_ScriptedClient):
        async def chat(self, system, messages, tools, **kwargs):  # type: ignore[no-untyped-def]
            seen_systems.append(system)
            seen_prompts.append(messages[0].content[0].text)
            return await super().chat(system, messages, tools, **kwargs)

    client = _RecordingClient(["Answer: ID 1"])
    chunks = await semantic_chunk(TEXT, client, target_tokens=11)

    assert chunks[0] == "Alpha one."
    assert chunks[1] == "\n\n".join(PARAGRAPHS[1:])
    assert client.calls == 2
    # The prompted block is ID-tagged paragraphs, per the reference prompt.
    assert "ID 0: Alpha one." in seen_prompts[0]
    assert seen_prompts[1].index("ID 1: Alpha two.") < seen_prompts[1].index("ID 2:")
    assert "Find the first paragraph (not the first one)" in seen_systems[0]


@pytest.mark.asyncio
async def test_semantic_chunk_unparseable_answer_merges_forward():
    # An unparseable answer records no boundary: the block merges forward into
    # the current chunk and no content is dropped.
    client = _ScriptedClient(["I cannot answer that.", "Answer: ID 3"])
    chunks = await semantic_chunk(TEXT, client, target_tokens=11)

    assert client.calls == 2
    assert chunks == [
        "Alpha one.\n\nAlpha two.\n\nBeta one.",
        "Beta two.\n\nGamma one.\n\nGamma two.\n\nDelta one.\n\nDelta two.",
    ]


@pytest.mark.asyncio
async def test_semantic_chunk_first_paragraph_id_is_rejected():
    # The reference prompt forbids naming the first paragraph; if the model
    # names it anyway, no boundary is recorded (block merges forward).
    client = _ScriptedClient(["Answer: ID 0"])
    chunks = await semantic_chunk(TEXT, client, target_tokens=11)

    assert chunks == [TEXT]


@pytest.mark.asyncio
async def test_semantic_chunk_short_document_never_queries():
    # The reference stops querying within the trailing paragraphs: a document
    # of five or fewer paragraphs is emitted whole without any LLM call.
    text = "\n\n".join(PARAGRAPHS[:4])
    assert await semantic_chunk(text, _BoomClient()) == [text]


@pytest.mark.asyncio
async def test_semantic_chunk_empty_text():
    assert await semantic_chunk("", _BoomClient()) == []


@pytest.mark.asyncio
async def test_fetch_url_uses_semantic_chunker_when_enabled(httpx_mock: HTTPXMock):
    """The call-site edit must invoke the segmenter and honor the named split."""
    httpx_mock.add_response(
        url="https://example.com/filing",
        text=SAMPLE_TEXT,
        headers={"content-type": "text/plain"},
    )
    # The model names ID 3, splitting revenue paragraphs from dividend ones.
    client = _ScriptedClient(["Answer: ID 3"])
    tool = FetchUrlTool(model_client=client, semantic_chunking=True, retrieve_k=2)
    out = await tool.run({"url": "https://example.com/filing", "query": "dividend"})

    # The LLM was consulted and the document split at the named paragraph: the
    # dividend content and the revenue content surface as separate chunks.
    # (With only two chunks BM25's IDF is exactly 0 here, so assert the split
    # structure rather than the ranking.)
    assert client.calls == 1
    assert "chunk 1" in out
    sections = out.split("--- chunk ")
    assert len(sections) == 3  # preamble + two ranked chunks
    dividend_section = next(s for s in sections if "payout ratio" in s)
    assert "Revenue grew" not in dividend_section
    revenue_section = next(s for s in sections if "Revenue grew" in s)
    assert "payout ratio" not in revenue_section


@pytest.mark.asyncio
async def test_fetch_url_does_not_call_model_when_chunking_disabled(httpx_mock: HTTPXMock):
    """Default behavior is unchanged: no model client is consulted on the query path."""
    httpx_mock.add_response(
        url="https://example.com/filing",
        text=SAMPLE_TEXT,
        headers={"content-type": "text/plain"},
    )
    # A model client is attached but semantic_chunking is off -> it must not run.
    tool = FetchUrlTool(model_client=_BoomClient())
    out = await tool.run({"url": "https://example.com/filing", "query": "dividend"})
    assert "chunk 1" in out
