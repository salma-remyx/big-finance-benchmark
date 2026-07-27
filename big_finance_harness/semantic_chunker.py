"""LLM-driven variable-size document segmentation for fetch_url retrieval.

Adapted from LumberChunker (Pipón-Noriega et al., 2024,
https://arxiv.org/abs/2406.17526). LumberChunker accumulates consecutive
sentences into a block until it reaches a target chunk size in words, then
prompts an LLM: "If you divide the following text into two chunks, what is the
first sentence where the second chunk should start?" The block splits at the
sentence the model names; the leading part becomes one chunk and the remainder
carries forward as the start of the next accumulation cycle. The result is a
``list[str]`` of semantically coherent, variable-size segments that drop into
``fetch_url``'s BM25 in-document retrieval in place of the fixed-size paragraph
chunker, which matters for long SEC filings where fixed token windows routinely
cut a topic in half.

Mode: direct port of the paper's core mechanism (target-size accumulation plus
LLM-located split points at sentence granularity), reusing the harness's own
``ModelClient`` for the LLM calls. The single auxiliary substitution is the
sentence tokenizer — the paper uses spaCy; we use a dependency-free regex
splitter so the harness pulls in no extra NLP dependency. That is preprocessing
only; it does not change the segmentation mechanism.
"""

from __future__ import annotations

import re

from big_finance_harness.models.base import ModelClient
from big_finance_harness.types import Message, TextBlock

# Target accumulated block size in words. Once the buffer reaches this size the
# LLM is asked where to split it. The paper evaluates targets up to ~450 words.
DEFAULT_TARGET_WORDS = 450
# Cap on LLM calls per document. A long SEC filing can yield hundreds of
# target-size blocks; this bounds cost. Text past the cap is emitted in
# target-sized blocks so no content is silently dropped.
DEFAULT_MAX_LLM_CALLS = 40
# The model answers with a full sentence copied from the text, so the response
# budget must fit the longest plausible sentence, not just an integer.
DEFAULT_MAX_OUTPUT_TOKENS = 256

# Break on whitespace following sentence-ending punctuation. Deliberately simple
# — good enough for prose-heavy filings without a spaCy/nltk dependency.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\w+")

_SYSTEM_PROMPT = (
    "You segment documents into semantically coherent chunks. Given a block of "
    "consecutive sentences from one document, locate the point where the content "
    "begins to shift to a new topic."
)


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences with a dependency-free regex."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _build_split_prompt(block: str) -> str:
    """The paper's prompt formulation, verbatim in spirit (§2 of the paper)."""
    return (
        f"Text:\n{block}\n\n"
        "If you divide the above text into two chunks, what is the first sentence "
        "where the second chunk should start? Respond with ONLY that sentence, "
        "copied verbatim from the text. If the text is a single coherent passage "
        "that should not be divided, respond with the first sentence of the text."
    )


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip("\"'“”‘’ ")


def _match_split_sentence(response: str, sentences: list[str]) -> int | None:
    """Locate the sentence the LLM named as the split point.

    Returns the index of the matched sentence, or ``None`` when the response
    names the first sentence (the model's "do not split" signal) or cannot be
    matched to any sentence (refusal or hallucination) — both mean "emit the
    block whole". Matching is exact first, then containment, then word overlap,
    per the paper's suggestion to map the output to the closest valid sentence
    via standard string matching.
    """
    answer = _normalize(response)
    if not answer:
        return None
    lowered = answer.lower()
    normed = [_normalize(s) for s in sentences]
    # Exact match: the model copied a sentence verbatim.
    for idx, sent in enumerate(normed):
        if sent.lower() == lowered:
            return idx if idx > 0 else None
    # Containment: a sentence appears inside a longer quoted answer. Sentence 0
    # is excluded — matching it means "do not split".
    for idx, sent in enumerate(normed[1:], start=1):
        if sent.lower() in lowered:
            return idx
    # Fuzzy fallback: best word-overlap match for near-verbatim answers.
    answer_words = set(_WORD_RE.findall(lowered))
    if not answer_words:
        return None
    best_idx: int | None = None
    best_score = 0.0
    for idx, sent in enumerate(normed):
        words = set(_WORD_RE.findall(sent.lower()))
        if not words:
            continue
        score = len(answer_words & words) / min(len(answer_words), len(words))
        if score > best_score:
            best_idx, best_score = idx, score
    if best_idx is None or best_idx == 0 or best_score < 0.6:
        return None
    return best_idx


async def _ask_split_index(
    model: ModelClient, sentences: list[str], *, max_output_tokens: int
) -> int | None:
    block = " ".join(sentences)
    response = await model.chat(
        system=_SYSTEM_PROMPT,
        messages=[Message(role="user", content=[TextBlock(text=_build_split_prompt(block))])],
        tools=[],
        temperature=0.0,
        max_output_tokens=max_output_tokens,
    )
    return _match_split_sentence(response.text, sentences)


async def semantic_chunk(
    text: str,
    model: ModelClient,
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> list[str]:
    """Segment ``text`` into variable-size chunks using an LLM.

    Mirrors LumberChunker's iterative loop: sentences accumulate into a block
    until it reaches ``target_words`` words, then the LLM is asked for the first
    sentence where a second chunk should start. The block splits at that
    sentence — the leading part becomes a chunk and the remainder carries
    forward as the start of the next block. When the LLM names the first
    sentence, refuses to split, or hallucinates a sentence not in the block, the
    block is emitted whole. LLM calls are capped at ``max_llm_calls`` — past the
    cap the remaining text is emitted in target-sized blocks so no content is
    silently dropped.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    calls = 0
    i = 0
    while i < len(sentences):
        # Accumulate until the block reaches the target chunk size (paper §2.1:
        # the LLM is queried once the growing chunk hits the target size k).
        while i < len(sentences) and (buffer_words < target_words or len(buffer) < 2):
            buffer.append(sentences[i])
            buffer_words += _word_count(sentences[i])
            i += 1
        if i < len(sentences) and len(buffer) >= 2 and calls < max_llm_calls:
            calls += 1
            split_idx = await _ask_split_index(model, buffer, max_output_tokens=max_output_tokens)
            if split_idx is None:
                # No content shift found (or an unmatchable answer): keep whole.
                chunks.append(" ".join(buffer))
                buffer = []
                buffer_words = 0
            else:
                chunks.append(" ".join(buffer[:split_idx]))
                buffer = buffer[split_idx:]
                buffer_words = _word_count(" ".join(buffer))
        else:
            # End of text, or the LLM call cap is reached: flush the block.
            # Past the cap this degrades to target-sized blocks; nothing is lost.
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buffer_words = 0
    return chunks
