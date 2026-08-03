"""Grounding check — adapted from MiniCheck (Tang et al., 2024).

MiniCheck (https://arxiv.org/abs/2404.10774) frames LLM fact-checking against a
grounding document as a ``(document, claim) -> supported / not`` contract: split a
generation into atomic claims and judge each against the evidence with a small,
specialized model, rather than spending many large-LLM calls. The harness's agent
retrieves documents (edgar_search / fetch_url / web_search) and emits a
``final_answer`` with nothing that verifies the answer is actually grounded in that
evidence — this module fills that gap.

This is a **Mode 2 (adapted port)**. We keep the paper's contract and its
decompose-then-aggregate shape. The paper's learned NLI fact-checker — a
fine-tuned classifier that needs downloaded weights the harness does not host — is
replaced by a dependency-free, deterministic entailment proxy: for each atomic
claim we retrieve the best-matching document sentence and score support from
content-token coverage, numeric consistency (financial claims hinge on their
numbers), and negation polarity. The paper's separate fact-checking benchmark /
eval framework is intentionally cut — evaluation wiring belongs downstream.

The contract is exposed both as a library callable (``GroundingCheck.check``) and
as an agent-optional ``Tool`` (``GroundingCheckTool``). The tool is deliberately
NOT in ``default_tools()`` by default: it compounds the same evidence-traceability
axis as the grader's CM-LRS score without altering the canonical model-facing
tool surface. Opt in via ``default_tools(include_grounding_check=True)`` or by
constructing the tool directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from big_finance_harness.tools.base import Tool, ToolError

# A small, finance-aware stopword set. Kept inline so the proxy stays stdlib-only.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from into over under
    is are was were be been being this that these those it its as their his her our your
    we us you they them i he she which who whom whose what when where why how all any both
    each few more most other some such no nor not only own same so than too very can will
    just don should now s t
    """.split()
)

# Negation tokens flip entailment polarity: "X" vs "not X" are not the same claim.
_NEGATIONS = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "without",
        "none",
        "n't",
        "unable",
        "cannot",
        "fails",
        "failed",
        "denies",
        "denied",
        "absent",
    }
)

# Magnitude suffixes scaled to a canonical numeric value so "$410.5 million" and
# "410,500,000" compare equal. Single-letter suffixes (m/b/k) are accepted because
# they are standard in filings; the numeric check also falls back to raw-digit
# matching, so a mis-scaled suffix degrades gracefully rather than failing.
_MAGNITUDES = {
    "trillion": 1e12,
    "billion": 1e9,
    "million": 1e6,
    "thousand": 1e3,
    "bn": 1e9,
    "mn": 1e6,
    "m": 1e6,
    "b": 1e9,
    "k": 1e3,
}

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")
_NUMBER_RE = re.compile(
    r"(?P<num>-?\$?\d[\d,]*(?:\.\d+)?)\s*(?P<mag>trillion|billion|million|"
    r"thousand|bn|mn|k|b|m)?",
    re.IGNORECASE,
)

# Score -> label thresholds. A claim must clear a high bar to be "supported"; a
# weaker signal lands as "partially_supported"; a missing number or near-zero
# coverage is "unsupported".
_SUPPORTED = 0.75
_PARTIAL = 0.4


@dataclass
class SubclaimResult:
    """Verdict for one atomic claim extracted from the generation."""

    text: str
    label: str  # "supported" | "partially_supported" | "unsupported"
    score: float
    evidence: str


@dataclass
class GroundingResult:
    """Aggregate verdict for a (document, claim) pair — the MiniCheck contract."""

    label: str  # "supported" | "partially_supported" | "unsupported"
    score: float  # mean across subclaims
    min_score: float  # weakest subclaim drives the conservative overall label
    subclaims: list[SubclaimResult] = field(default_factory=list)

    @property
    def evidence(self) -> list[str]:
        return [s.evidence for s in self.subclaims if s.evidence]

    def to_json(self) -> str:
        return json.dumps(
            {
                "label": self.label,
                "score": round(self.score, 4),
                "min_score": round(self.min_score, 4),
                "n_subclaims": len(self.subclaims),
                "subclaims": [
                    {
                        "text": s.text,
                        "label": s.label,
                        "score": round(s.score, 4),
                        "evidence": s.evidence,
                    }
                    for s in self.subclaims
                ],
                "evidence": self.evidence,
            },
            ensure_ascii=False,
        )


def _content_tokens(text: str) -> set[str]:
    return {tok for tok in _WORD_RE.findall(text.lower()) if tok not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()]
    return [s for s in sentences if any(c.isalnum() for c in s)]


def _extract_numbers(text: str) -> list[tuple[float, str]]:
    """Return ``(scaled_value, raw_digits)`` per numeric mention in ``text``.

    ``raw_digits`` keeps the original decimal form so a mis-scaled magnitude can
    still match on the digit substring alone.
    """
    out: list[tuple[float, str]] = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group("num")
        cleaned = raw.lstrip("-$").replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        mag = (m.group("mag") or "").lower()
        value *= _MAGNITUDES.get(mag, 1.0)
        out.append((value, cleaned))
    return out


def _numbers_present(
    claim_nums: list[tuple[float, str]], doc_nums: list[tuple[float, str]]
) -> bool:
    """True if every claim number is corroborated somewhere in the document."""
    for c_val, c_raw in claim_nums:
        found = False
        for d_val, d_raw in doc_nums:
            # Scaled match within 0.5% relative tolerance (or a tiny absolute
            # tolerance for figures near zero) — handles "$410.5M" == "410,500,000".
            if abs(c_val - d_val) <= max(0.01, abs(c_val) * 0.005):
                found = True
                break
            # Raw-digit substring fallback: survives magnitude mis-scaling and
            # formatting variants ("410.5" in "$410.5 million").
            if c_raw and c_raw in d_raw:
                found = True
                break
        if not found:
            return False
    return True


def _has_negation(text: str) -> bool:
    return bool(_NEGATIONS & set(_WORD_RE.findall(text.lower())))


def _score_subclaim(
    claim: str, doc_sentences: list[str], doc_nums: list[tuple[float, str]]
) -> tuple[float, str]:
    """Return ``(support_score in [0, 1], evidence_sentence)`` for one claim."""
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return 0.0, ""

    best_sent = ""
    best_hit = 0
    for sent in doc_sentences:
        hit = len(claim_tokens & _content_tokens(sent))
        if hit > best_hit:
            best_hit = hit
            best_sent = sent

    coverage = best_hit / len(claim_tokens)
    score = coverage

    # Financial claims live or die on their numbers: an invented or wrong figure
    # makes the claim unsupported regardless of word overlap.
    claim_nums = _extract_numbers(claim)
    if claim_nums and not _numbers_present(claim_nums, doc_nums):
        score = 0.0

    # Negation polarity mismatch ("rose" vs "did not rise") halves support when
    # the rest of the claim otherwise matches.
    if best_sent and coverage >= 0.4 and _has_negation(claim) != _has_negation(best_sent):
        score *= 0.5

    return score, best_sent


def _label_for(score: float) -> str:
    if score >= _SUPPORTED:
        return "supported"
    if score >= _PARTIAL:
        return "partially_supported"
    return "unsupported"


class GroundingCheck:
    """Library entry point for the (document, claim) -> grounding verdict contract.

    Decomposes ``claim`` into atomic sentences, scores each against ``document``,
    and aggregates. The overall label is the *weakest* subclaim's label — a claim
    is only grounded if every part of it is.
    """

    def check(self, document: str, claim: str) -> GroundingResult:
        sub_texts = _split_sentences(claim) or [claim.strip()]
        doc_sentences = _split_sentences(document)
        doc_nums = _extract_numbers(document)

        subclaims: list[SubclaimResult] = []
        for sub in sub_texts:
            score, evidence = _score_subclaim(sub, doc_sentences, doc_nums)
            subclaims.append(
                SubclaimResult(
                    text=sub,
                    label=_label_for(score),
                    score=score,
                    evidence=evidence,
                )
            )

        scores = [s.score for s in subclaims] or [0.0]
        mean_score = sum(scores) / len(scores)
        min_score = min(scores)
        return GroundingResult(
            label=_label_for(min_score),
            score=mean_score,
            min_score=min_score,
            subclaims=subclaims,
        )


class GroundingCheckTool(Tool):
    """Agent-optional tool wrapping :class:`GroundingCheck`.

    Lets the agent fact-check a draft claim against a retrieved document before
    submitting it as a final answer, returning a label, score, and the cited
    evidence span. Library callers should usually call ``GroundingCheck().check``
    directly; this class exists so the capability is reachable from the same tool
    surface the rest of the harness uses.
    """

    name = "grounding_check"
    description = (
        "Fact-check a claim against a grounding document you have already retrieved "
        "(e.g. via fetch_url or edgar_search). Returns whether the claim is "
        "'supported', 'partially_supported', or 'unsupported' by the document, a "
        "numeric support score, and the document sentence that best backs the "
        "claim. Use it to verify a figure or statement is actually grounded in "
        "your evidence before submitting final_answer — a claim whose numbers do "
        "not appear in the document is reported as unsupported."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "claim": {
                "type": "string",
                "description": "The statement or figure to verify.",
            },
            "document": {
                "type": "string",
                "description": "The grounding document text to check the claim against.",
            },
        },
        "required": ["claim", "document"],
        "additionalProperties": False,
    }

    def __init__(self, checker: GroundingCheck | None = None) -> None:
        self._checker = checker or GroundingCheck()

    async def run(self, args: dict[str, Any]) -> str:
        claim = args.get("claim")
        document = args.get("document")
        if not isinstance(claim, str) or not claim.strip():
            raise ToolError("claim is required and must be a non-empty string")
        if not isinstance(document, str) or not document.strip():
            raise ToolError("document is required and must be a non-empty string")
        return self._checker.check(document=document, claim=claim).to_json()
