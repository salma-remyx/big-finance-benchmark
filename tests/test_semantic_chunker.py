"""Integration tests for LumberChunker-style semantic segmentation in fetch_url.

The fetch_url tool's query path chunks a document then BM25-ranks the chunks.
These tests exercise the opt-in wiring that swaps the fixed paragraph chunker
for the LLM-driven `semantic_chunk` segmenter, plus self-tests of the segmenter
itself. The mocked model answers with the *sentence* where the second chunk
should start (the paper's prompt formulation), and the tests assert the splits
land at exactly those mock-specified sentences. The integration test goes
through the public FetchUrlTool surface (a non-new module) so it proves the
call-site edit actually invokes the segmenter.
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from big_finance_harness.models.base import ModelClient, ThinkingLevel
from big_finance_harness.semantic_chunker import semantic_chunk
from big_finance_harness.tools.fetch_url import FetchUrlTool
from big_finance_harness.types import Message, ModelResponse, ToolSpec


class _ScriptedClient(ModelClient):
    """Stub model returning canned split-point sentences, in order."""

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


SAMPLE_HTML = """\
<html><body>
<p>Revenue grew across the segment. Services revenue hit a new high. Products held flat year over year.</p>
<p>The board declared a dividend. The payout ratio increased. The record date is set for next month.</p>
</body></html>
"""


@pytest.mark.asyncio
async def test_semantic_chunk_splits_at_mock_named_sentences():
    # Two-word sentences; target_words=3 triggers a query after two sentences.
    # The model names "Alpha two." then "Beta one." as split points, so each
    # chunk boundary must land exactly at the mock-specified sentence.
    text = "Alpha one. Alpha two. Beta one. Beta two."
    client = _ScriptedClient(["Alpha two.", "Beta one."])
    chunks = await semantic_chunk(text, client, target_words=3)

    assert client.calls == 2
    assert chunks == ["Alpha one.", "Alpha two.", "Beta one. Beta two."]


@pytest.mark.asyncio
async def test_semantic_chunk_carries_remainder_into_next_block():
    # After a split, the tail of the block must seed the next accumulation
    # cycle: the sentence the model named reappears at the start of a later
    # prompt, never inside the just-emitted chunk.
    text = "Alpha one. Alpha two. Beta one. Beta two."
    seen_prompts: list[str] = []

    class _RecordingClient(_ScriptedClient):
        async def chat(self, system, messages, tools, **kwargs):  # type: ignore[no-untyped-def]
            seen_prompts.append(messages[0].content[0].text)
            return await super().chat(system, messages, tools, **kwargs)

    client = _RecordingClient(["Beta one."])
    chunks = await semantic_chunk(text, client, target_words=5)

    assert chunks == ["Alpha one. Alpha two.", "Beta one. Beta two."]
    assert client.calls == 1
    # The queried block contained all four sentences (target 5 words reached
    # after three, but the two-sentence minimum plus word budget absorbed all).
    assert "Beta one." in seen_prompts[0]


@pytest.mark.asyncio
async def test_semantic_chunk_first_sentence_answer_means_no_split():
    # The model answering with the block's first sentence is the "do not split"
    # signal: the whole accumulated block is emitted as one chunk.
    text = "One fish. Two fish. Red fish. Blue fish."
    client = _ScriptedClient(["One fish."])
    chunks = await semantic_chunk(text, client, target_words=3)

    assert client.calls == 1
    assert chunks == ["One fish. Two fish.", "Red fish. Blue fish."]


@pytest.mark.asyncio
async def test_semantic_chunk_near_verbatim_answer_fuzzy_matches():
    # A fragment of a sentence (the paper's hallucination case) must still map
    # to the closest valid sentence via string matching.
    text = "Alpha one. Alpha two. Beta one. Beta two."
    client = _ScriptedClient(['"Beta one"'])
    chunks = await semantic_chunk(text, client, target_words=5)

    assert chunks == ["Alpha one. Alpha two.", "Beta one. Beta two."]


@pytest.mark.asyncio
async def test_semantic_chunk_refusal_emits_block_whole():
    # An unmatchable refusal must never merge across blocks or drop content:
    # the queried block is emitted whole.
    text = "One fish. Two fish. Red fish. Blue fish."
    client = _ScriptedClient(["I cannot split this text."])
    chunks = await semantic_chunk(text, client, target_words=3)

    assert chunks == ["One fish. Two fish.", "Red fish. Blue fish."]


@pytest.mark.asyncio
async def test_semantic_chunk_respects_llm_call_cap():
    # max_llm_calls=0 emits target-sized blocks verbatim without any LLM call.
    text = "One. Two. Three. Four. Five. Six."
    client = _ScriptedClient(["Two."])
    chunks = await semantic_chunk(text, client, target_words=2, max_llm_calls=0)

    assert client.calls == 0
    assert chunks == ["One. Two.", "Three. Four.", "Five. Six."]


@pytest.mark.asyncio
async def test_semantic_chunk_empty_text():
    assert await semantic_chunk("", _BoomClient()) == []


@pytest.mark.asyncio
async def test_fetch_url_uses_semantic_chunker_when_enabled(httpx_mock: HTTPXMock):
    """The call-site edit must invoke the segmenter and honor the named split."""
    httpx_mock.add_response(
        url="https://example.com/filing",
        text=SAMPLE_HTML,
        headers={"content-type": "text/html"},
    )
    # retrieve_chunk_tokens doubles as the target block size in words; 25 makes
    # the six-sentence sample trigger exactly one split query.
    client = _ScriptedClient(["The board declared a dividend."])
    tool = FetchUrlTool(
        model_client=client, semantic_chunking=True, retrieve_k=2, retrieve_chunk_tokens=25
    )
    out = await tool.run({"url": "https://example.com/filing", "query": "dividend"})

    # The LLM was consulted and the document split at the named sentence: the
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
        text=SAMPLE_HTML,
        headers={"content-type": "text/html"},
    )
    # A model client is attached but semantic_chunking is off -> it must not run.
    tool = FetchUrlTool(model_client=_BoomClient())
    out = await tool.run({"url": "https://example.com/filing", "query": "dividend"})
    assert "chunk 1" in out
