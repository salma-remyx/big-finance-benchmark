"""LLM-driven variable-size document segmentation for fetch_url retrieval.

Adapted from LumberChunker (Pipón-Noriega et al., 2024,
https://arxiv.org/abs/2406.17526; reference implementation:
https://github.com/joaodsmarques/LumberChunker). The document's paragraphs are
tagged with sequential IDs ("ID 0: <text>", "ID 1: <text>", ...) and
consecutive paragraphs accumulate into a block until it reaches the target
size θ — 550 tokens, approximated as 1.2 tokens per word exactly as the
reference's ``count_words`` does. The LLM is then prompted (system prompt
verbatim from the reference):

    "Find the first paragraph (not the first one) where the content clearly
    changes compared to the previous paragraphs. ... Return the ID of the
    paragraph with the content shift as in the exemplified format:
    'Answer: ID XXXX'."

The returned ID is parsed with a regex and recorded as a boundary: paragraphs
before it form one chunk and accumulation restarts at the named paragraph.
When the answer cannot be parsed, no boundary is recorded and the block
merges forward into the current chunk, so no content is ever dropped. The
result is a ``list[str]`` of semantically coherent, variable-size segments
that drop into ``fetch_url``'s BM25 in-document retrieval in place of the
fixed-size paragraph chunker, which matters for long SEC filings where fixed
token windows routinely cut a topic in half.

Mode: direct port of the reference's segmentation loop (ID-tagged paragraph
accumulation, the reference prompt, regex ID parsing, merge-forward failure
semantics), reusing the harness's own ``ModelClient`` for the LLM calls. The
single auxiliary substitution is paragraph pre-segmentation — the reference
consumes pre-segmented book paragraphs; we split fetched text on blank lines.
That is preprocessing only; it does not change the segmentation mechanism.
"""

from __future__ import annotations

import re

from big_finance_harness.models.base import ModelClient
from big_finance_harness.types import Message, TextBlock

# Target accumulated block size θ in tokens. The paper sweeps θ over
# 450–1000; the reference implementation uses 550.
DEFAULT_TARGET_TOKENS = 550
# The reference stops querying within the last few paragraphs of a document;
# the trailing paragraphs simply extend the final chunk.
_TAIL_PARAGRAPHS = 5

_ANSWER_RE = re.compile(r"Answer: ID \w+")
_DIGITS_RE = re.compile(r"\d+")

# Verbatim from the reference implementation (LumberChunker-Segmentation.py).
_SYSTEM_PROMPT = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""


def split_paragraphs(text: str) -> list[str]:
    """Split ``text`` into paragraphs on blank-line boundaries."""
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _approx_tokens(text: str) -> int:
    """Approximate the token count as 1.2 tokens per word (reference count_words)."""
    return round(1.2 * len(text.split()))


def _parse_answer_id(response: str) -> int | None:
    """Parse the paragraph ID out of an 'Answer: ID XXXX' response."""
    match = _ANSWER_RE.search(response)
    if match is None:
        return None
    digits = _DIGITS_RE.search(match.group(0))
    return int(digits.group()) if digits else None


async def semantic_chunk(
    text: str,
    model: ModelClient,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
) -> list[str]:
    """Segment ``text`` into variable-size chunks using an LLM.

    Mirrors the reference loop: paragraphs accumulate into a block until it
    reaches ``target_tokens`` approximate tokens, then the LLM is asked for
    the ID of the first paragraph where the content clearly changes. The
    block splits at that paragraph — the leading part becomes a chunk and
    accumulation restarts at the named paragraph. When the answer cannot be
    parsed, no boundary is recorded and the block merges forward into the
    current chunk; either way no content is dropped.
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    tagged = [f"ID {idx}: {p}" for idx, p in enumerate(paragraphs)]
    n = len(tagged)
    boundaries: list[int] = []
    start = 0
    while start < n - _TAIL_PARAGRAPHS:
        # Accumulate until the block reaches the target size θ (paper §2: the
        # LLM is queried once the growing chunk hits the target size).
        i = 0
        block_tokens = 0
        while block_tokens < target_tokens and start + i < n - 1:
            i += 1
            block_tokens = _approx_tokens("\n".join(tagged[start : start + i]))
        # The paragraph that crossed the target is held out of the prompt and
        # seeds the next accumulation cycle (reference: the prompt covers
        # paragraphs [start, start+i-1) unless only one accumulated).
        end = start + max(i - 1, 1)
        document = "\n".join(tagged[start:end])

        response = await model.chat(
            system=_SYSTEM_PROMPT,
            messages=[Message(role="user", content=[TextBlock(text=f"\nDocument:\n{document}")])],
            tools=[],
            temperature=0.1,
        )
        answer_id = _parse_answer_id(response.text)
        if answer_id is None or not (start < answer_id < end):
            # Unparseable answer (or one naming a paragraph outside the
            # prompted block, including the forbidden first one): record no
            # boundary — the block merges forward into the current chunk.
            start = end
        else:
            boundaries.append(answer_id)
            start = answer_id

    chunks: list[str] = []
    prev = 0
    for boundary in [*boundaries, n]:
        chunks.append("\n\n".join(paragraphs[prev:boundary]))
        prev = boundary
    return chunks
