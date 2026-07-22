"""LLM-driven variable-size document segmentation for fetch_url retrieval.

Adapted from LumberChunker (Pipón-Noriega et al., 2024,
https://arxiv.org/abs/2406.17526). LumberChunker splits a document into short
passages of consecutive sentences, then iteratively prompts an LLM to locate the
point in a window of passages where the content begins to shift. The leading
passages that share the first passage's topic are merged into one variable-size
chunk; the next window starts immediately after the shift. The result is a
``list[str]`` of semantically coherent segments that drop into ``fetch_url``'s
BM25 in-document retrieval in place of the fixed-size paragraph chunker, which
matters for long SEC filings where fixed token windows routinely cut a topic in
half.

Mode: direct port of the paper's core mechanism (LLM-located content shifts),
reusing the harness's own ``ModelClient`` for the LLM calls. The single
auxiliary substitution is the sentence tokenizer — the paper uses spaCy; we use
a dependency-free regex splitter so the harness pulls in no extra NLP
dependency. That is preprocessing only; it does not change the segmentation
mechanism.
"""

from __future__ import annotations

import re

from big_finance_harness.models.base import ModelClient
from big_finance_harness.types import Message, TextBlock

# Passages are groups of this many consecutive sentences. The paper used 3.
DEFAULT_SENTENCES_PER_PASSAGE = 3
# Number of consecutive passages shown to the LLM per prompt. A larger window
# gives the model more context to find a shift at the cost of more tokens/call.
DEFAULT_GROUP_SIZE = 7
# Cap on LLM calls per document. A long SEC filing can yield hundreds of
# passages; this bounds cost. Passages past the cap are emitted verbatim so no
# content is silently dropped.
DEFAULT_MAX_LLM_CALLS = 40

# Break on whitespace following sentence-ending punctuation. Deliberately simple
# — good enough for prose-heavy filings without a spaCy/nltk dependency.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

_SYSTEM_PROMPT = (
    "You segment documents. You will receive a numbered list of consecutive "
    "passages taken in order from one document. Find the largest passage number "
    "N (1 <= N <= K) such that passages 1 through N all discuss the same topic "
    "as passage 1. Passage N+1, if present, shifts to a different topic. If every "
    "passage shares passage 1's topic, answer K. Respond with ONLY the integer N."
)


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences with a dependency-free regex."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]


def build_passages(sentences: list[str], sentences_per_passage: int) -> list[str]:
    """Group ``sentences`` into passages of ``sentences_per_passage`` each."""
    if sentences_per_passage < 1:
        raise ValueError("sentences_per_passage must be >= 1")
    return [
        " ".join(sentences[i : i + sentences_per_passage])
        for i in range(0, len(sentences), sentences_per_passage)
    ]


def _format_window(passages: list[str]) -> str:
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def _parse_topic_count(raw: str, window_size: int) -> int:
    """Parse the LLM's integer answer into a count in ``[1, window_size]``.

    Defaults to 1 (emit a single passage) when the model is unparseable, so a
    bad answer never merges unrelated passages into one chunk.
    """
    nums = re.findall(r"\d+", raw)
    if not nums:
        return 1
    return max(1, min(int(nums[0]), window_size))


async def _ask_topic_count(
    model: ModelClient, passages: list[str], *, max_output_tokens: int
) -> int:
    response = await model.chat(
        system=_SYSTEM_PROMPT,
        messages=[Message(role="user", content=[TextBlock(text=_format_window(passages))])],
        tools=[],
        temperature=0.0,
        max_output_tokens=max_output_tokens,
    )
    return _parse_topic_count(response.text, len(passages))


async def semantic_chunk(
    text: str,
    model: ModelClient,
    *,
    sentences_per_passage: int = DEFAULT_SENTENCES_PER_PASSAGE,
    group_size: int = DEFAULT_GROUP_SIZE,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    max_output_tokens: int = 16,
) -> list[str]:
    """Segment ``text`` into variable-size chunks using an LLM.

    Mirrors LumberChunker: passages of consecutive sentences are formed, then a
    sliding window is shown to the LLM. Each call returns how many leading
    passages share the first passage's topic; those passages are merged into one
    chunk and the window advances past them. The number of LLM calls is capped
    at ``max_llm_calls`` — once the cap is hit the remaining passages are
    emitted as-is so no text is lost.
    """
    passages = build_passages(split_sentences(text), sentences_per_passage)
    if not passages:
        return []
    if group_size < 2:
        # No shift can be detected without at least two passages to compare.
        return list(passages)

    chunks: list[str] = []
    i = 0
    calls = 0
    while i < len(passages):
        window = passages[i : i + group_size]
        if len(window) == 1:
            # Lone trailing passage: nothing to compare it against.
            chunks.append(window[0])
            break
        if calls >= max_llm_calls:
            # Cap reached: flush all remaining passages verbatim, no more calls.
            chunks.extend(passages[i:])
            break
        count = await _ask_topic_count(model, window, max_output_tokens=max_output_tokens)
        calls += 1
        chunks.append("\n\n".join(window[:count]))
        i += count
    return chunks
