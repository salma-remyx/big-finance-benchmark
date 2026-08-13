"""Evidence-grounding / provenance scoring for finished agent runs.

Implements the standard information-retrieval provenance metrics that FinRank
(arXiv:2608.07400) introduces for SEC-filing question answering: recall of the
gold supporting passages and discrimination against curated hard negatives. A
plausible — even numerically correct — answer is only trustworthy when it is
grounded in the *right* disclosure (the intended entity, reporting period, and
filing section), not a confusable neighbour. These metrics surface that.

This is a clean-room reimplementation of the standard, uncopyrightable IR
measurements (Recall@K, pairwise ranking accuracy, provenance precision)
described by the paper. It does not copy FinRank's code or redistribute its
CC-BY-NC dataset.

The harness already carries the seam for this: ``DatasetItem.sources`` holds
the gold supporting passages, and a finished run's retrieved evidence lives in
``RunRecord.steps[*].tool_results[*].content`` (the EDGAR / fetch / web-search
tool outputs). Until now nothing connected the two. This module is that
connection — given a run and its dataset item, score how well the agent's
retrieved passages cover the gold evidence and whether they confuse it with
known hard negatives.

Mode 2 (adapted port): the paper's core measurement (provenance recall +
hard-negative discrimination) is kept at full fidelity; its learned embedder
for passage matching is replaced by a parameter-free token-overlap proxy, and
its standalone benchmark suite is replaced by consumption of this repo's
existing ``RunRecord`` / ``DatasetItem`` trace format. Attribution lives here
in the docstring, not in the filename.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from big_finance_harness.types import DatasetItem, RunRecord, StepRecord

# Tools whose ``tool_results`` carry retrieved evidence passages. The agent's
# EDGAR search is the primary financial-evidence source; fetch/web search are
# the secondary retrieval surfaces the same grounding question applies to.
RETRIEVAL_TOOL_NAMES = frozenset({"edgar_search", "fetch_url", "serpapi", "tavily", "web_search"})

# FinRank reports Recall@10 on the pooled evidence corpus.
_DEFAULT_K = 10

# A retrieved passage counts as covering a gold passage when it contains at
# least this fraction of the gold passage's content tokens. This token-overlap
# proxy stands in for the paper's learned-embedder match — lenient enough to
# absorb the excerpt boundaries a retriever returns, strict enough to reject an
# unrelated filing that merely shares boilerplate.
_DEFAULT_MATCH_THRESHOLD = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Small stoplist of the highest-frequency English function words so overlap is
# not dominated by boilerplate; domain terms (a CIK, "revenue", a figure) carry
# the discriminative signal.
_STOPWORDS = frozenset("a an the of to in and for on with at by from is are was were be".split())


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS)


def passage_overlap(retrieved: str, gold: str) -> float:
    """Token containment of the gold passage inside the retrieved passage.

    Containment (not symmetric Jaccard) is the right shape: a retriever returns
    an excerpt that should fully contain the gold fact, while the excerpt may
    carry surrounding context the gold passage omits. Returns 0.0 when the gold
    passage has no content tokens.
    """
    gold_tokens = _content_tokens(gold)
    if not gold_tokens:
        return 0.0
    retrieved_tokens = _content_tokens(retrieved)
    return len(gold_tokens & retrieved_tokens) / len(gold_tokens)


def matches_gold(retrieved: str, gold: str, threshold: float = _DEFAULT_MATCH_THRESHOLD) -> bool:
    """True when ``retrieved`` covers ``gold`` at or above ``threshold``."""
    return passage_overlap(retrieved, gold) >= threshold


def _step_retrieved_passages(step: StepRecord, tool_names: frozenset[str]) -> list[str]:
    # Map each tool_use_id to the tool name that issued it within this step, so
    # we only collect evidence-bearing results (and skip final_answer etc.).
    name_by_id = {tc.id: tc.name for tc in step.tool_calls}
    passages: list[str] = []
    for result in step.tool_results:
        name = name_by_id.get(result.tool_use_id)
        if name is None or name not in tool_names:
            continue
        if result.is_error or not result.content.strip():
            continue
        passages.append(result.content)
    return passages


def extract_retrieved_passages(
    run: RunRecord,
    tool_names: Iterable[str] | None = None,
) -> list[str]:
    """Return the agent's retrieved evidence passages in retrieval order.

    Order follows the order the agent issued the calls across steps — that
    ordering is the ranking Recall@K is measured over. By default only the
    retrieval/evidence tools are included; pass ``tool_names`` to narrow
    (e.g. ``{"edgar_search"}``) or widen.
    """
    names = frozenset(tool_names) if tool_names is not None else RETRIEVAL_TOOL_NAMES
    passages: list[str] = []
    for step in run.steps:
        passages.extend(_step_retrieved_passages(step, names))
    return passages


def recall_at_k(
    retrieved: list[str],
    gold: Iterable[str],
    k: int = _DEFAULT_K,
    threshold: float = _DEFAULT_MATCH_THRESHOLD,
) -> float:
    """Fraction of gold passages covered by the top-``k`` retrieved passages.

    The FinRank primary passage-retrieval metric (Recall@K). Returns 0.0 when
    there is no gold evidence to cover.
    """
    gold_list = [g for g in gold if g and g.strip()]
    if not gold_list:
        return 0.0
    top_k = retrieved[: max(k, 0)]
    covered = 0
    for g in gold_list:
        if any(matches_gold(passage, g, threshold) for passage in top_k):
            covered += 1
    return covered / len(gold_list)


def provenance_precision(
    retrieved: list[str],
    gold: Iterable[str],
    hard_negatives: Iterable[str] | None = None,
    threshold: float = _DEFAULT_MATCH_THRESHOLD,
) -> float:
    """Of retrieved passages that match a known gold OR hard-negative passage,
    the fraction that match gold.

    1.0 means every attributable retrieved passage is the *right* evidence; a
    retriever that latches onto confusable hard negatives scores low. Returns
    0.0 when no retrieved passage can be attributed to either set.
    """
    gold_list = [g for g in gold if g and g.strip()]
    negatives = [n for n in (hard_negatives or []) if n and n.strip()]
    relevant = 0
    gold_hits = 0
    for passage in retrieved:
        if any(matches_gold(passage, g, threshold) for g in gold_list):
            relevant += 1
            gold_hits += 1
        elif negatives and any(matches_gold(passage, n, threshold) for n in negatives):
            relevant += 1
    return gold_hits / relevant if relevant else 0.0


def pairwise_accuracy(
    retrieved: list[str],
    gold: Iterable[str],
    hard_negatives: Iterable[str] | None = None,
    threshold: float = _DEFAULT_MATCH_THRESHOLD,
) -> float:
    """FinRank's hard-negative pairwise task.

    For each (gold, hard-negative) pair, ask whether the retriever ranks the
    gold passage above the confusable negative. A passage not retrieved at all
    is ranked last (after everything retrieved). Returns the fraction of pairs
    ranked correctly, or 0.0 when no pairs are evaluable. The paper reports
    pairwise accuracy falling 13.0–20.5pp when random negatives are replaced
    with the curated hard negatives — this is that measurement.
    """
    gold_list = [g for g in gold if g and g.strip()]
    negatives = [n for n in (hard_negatives or []) if n and n.strip()]
    pairs = [(g, n) for g in gold_list for n in negatives]
    if not pairs:
        return 0.0
    not_found = len(retrieved)

    def rank(target: str) -> int:
        for index, passage in enumerate(retrieved):
            if matches_gold(passage, target, threshold):
                return index
        return not_found

    correct = sum(1 for g, n in pairs if rank(g) < rank(n))
    return correct / len(pairs)


@dataclass
class EvidenceGroundingScore:
    """Provenance scores for one (run, dataset item) pair."""

    question_id: str
    n_retrieved: int
    n_gold: int
    n_hard_negatives: int
    recall_at_k: float
    provenance_precision: float
    pairwise_accuracy: float
    # Human-readable diagnostics: which gold passages the top-k covered, and
    # which hard negatives the retriever wrongly surfaced.
    matched_gold: list[str] = field(default_factory=list)
    matched_hard_negatives: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "n_retrieved": self.n_retrieved,
            "n_gold": self.n_gold,
            "n_hard_negatives": self.n_hard_negatives,
            "recall_at_k": self.recall_at_k,
            "provenance_precision": self.provenance_precision,
            "pairwise_accuracy": self.pairwise_accuracy,
            "matched_gold": self.matched_gold,
            "matched_hard_negatives": self.matched_hard_negatives,
        }


def score_evidence_grounding(
    run: RunRecord,
    item: DatasetItem,
    hard_negatives: Iterable[str] | None = None,
    *,
    k: int = _DEFAULT_K,
    match_threshold: float = _DEFAULT_MATCH_THRESHOLD,
    tool_names: Iterable[str] | None = None,
) -> EvidenceGroundingScore:
    """Score one run's evidence grounding against its dataset item.

    Mirrors the ``grade()`` seam (``run`` + ``item``) so this can drop in as a
    future opt-in alongside the rubric grade. Raises ``ValueError`` on an id
    mismatch, the same guard ``grade()`` enforces.
    """
    if run.question_id != item.id:
        raise ValueError(f"run/item id mismatch: run={run.question_id} item={item.id}")

    retrieved = extract_retrieved_passages(run, tool_names)
    gold = [s for s in item.sources if s and s.strip()]
    negatives = [n for n in (hard_negatives or []) if n and n.strip()]

    top_k = retrieved[: max(k, 0)]
    matched_gold: list[str] = []
    for g in gold:
        if any(matches_gold(p, g, match_threshold) for p in top_k):
            matched_gold.append(g)
    matched_hard_negatives: list[str] = []
    for n in negatives:
        if any(matches_gold(p, n, match_threshold) for p in retrieved):
            matched_hard_negatives.append(n)

    return EvidenceGroundingScore(
        question_id=item.id,
        n_retrieved=len(retrieved),
        n_gold=len(gold),
        n_hard_negatives=len(negatives),
        recall_at_k=recall_at_k(retrieved, gold, k, match_threshold),
        provenance_precision=provenance_precision(retrieved, gold, negatives, match_threshold),
        pairwise_accuracy=pairwise_accuracy(retrieved, gold, negatives, match_threshold),
        matched_gold=matched_gold,
        matched_hard_negatives=matched_hard_negatives,
    )
