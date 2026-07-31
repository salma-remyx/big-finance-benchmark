"""Capability diagnosis from rubric grades — adapted from CRAFT.

CRAFT ("Clustering Rubrics to Diagnose Weak LLM Capabilities and Generate
Targeted Fine-Tuning Data") turns a rubric-based evaluation into a
model-specific diagnosis of *why* a model fails: every rubric criterion is
treated as a capability probe, the probes are clustered into a hierarchical
capability tree, the target model is scored at every node, and the
low-performing nodes are surfaced at the granularity where each failure is
clearest.

This module ports CRAFT's **diagnostic core** (extract → cluster → score →
select weak nodes). It is an adapted port (Mode 2): the paper's learned
components are replaced with parameter-free proxies so the diagnosis needs no
model calls at all, and the targeted fine-tuning-data-generation half of the
paper is dropped — this harness *measures* models, it does not train them.

Substitutions (kept honest for review):
  * capability-description extraction — paper: an LLM summarising each
    (prompt, rubric) pair; here: normalised rubric-line text. A BigFinanceBench
    rubric line is already an independently-verifiable analyst step, so it is
    its own capability probe; we just strip boilerplate tokens.
  * sentence embedding — paper: a learned embedder; here: a bag-of-words
    term-frequency vector with cosine similarity.
  * hierarchical clustering — paper: learned clustering; here: average-linkage
    agglomerative clustering, giving the same hierarchical capability tree and
    the same "across tree levels" granularity control.

Kept at full fidelity: the per-node model scoring (from ``GradedRubricLine``
``earned`` / ``points``) and the dynamic weak-node selection across levels.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from big_finance_harness.types import GradedRubricLine, GradedRun

# Generic English function words plus the boilerplate that opens most analyst
# rubric lines ("correctly ... the answer ..."). Stripping these lets the
# capability-specific terms (edgar, ratio, 10-k, ...) dominate the clustering.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at for with from by as is are was
    were be been being this that these those it its their his her our your we you
    they i he she them us him me my no not nor so than too very can will just do
    does did doing done have has had having into out up down over under again
    further once here there when where why how all any both each few more most
    other some such only own same correctly answer value number result total final
    s t d ll m re ve
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def capability_description(rubric_text: str) -> str:
    """Parameter-free proxy for CRAFT's LLM capability-description extractor.

    CRAFT prompts an LLM to summarise each (prompt, rubric criterion) pair into a
    short capability description. The harness's rubric lines are *already*
    independently-verifiable analyst steps, so a line doubles as its own
    capability probe; we expose its content tokens (boilerplate stripped) as the
    description that clustering keys off.
    """
    return " ".join(_tokenize(rubric_text))


def _term_freq(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    total = len(tokens)
    return {term: count / total for term, count in Counter(tokens).items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(weight * large.get(term, 0.0) for term, weight in small.items())
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class CapabilityProbe:
    """One rubric criterion from one question, viewed as a capability probe."""

    question_id: str
    rubric_text: str
    description: str
    points: int
    earned: bool


def probes_from_grades(graded_runs: Iterable[GradedRun]) -> list[CapabilityProbe]:
    """Turn graded runs into the per-rubric-criterion probes CRAFT reasons over.

    Operates on the harness's own ``GradedRun`` / ``GradedRubricLine`` contract,
    so it reads exactly what ``grader.grade`` writes.
    """
    probes: list[CapabilityProbe] = []
    for run in graded_runs:
        line: GradedRubricLine
        for line in run.rubric_lines:
            if line.points <= 0:
                continue
            probes.append(
                CapabilityProbe(
                    question_id=run.question_id,
                    rubric_text=line.text,
                    description=capability_description(line.text),
                    points=line.points,
                    earned=line.earned,
                )
            )
    return probes


@dataclass
class CapabilityNode:
    """A node in the hierarchical capability tree — a cluster of rubric criteria
    that share a capability, scored at the cluster level (CRAFT's "score the
    target model at every node")."""

    node_id: int
    member_indices: list[int]
    level: int
    score: float
    support: int
    n_questions: int
    keywords: list[str]
    example_rubric: str
    children: tuple[int, ...] = ()


@dataclass
class WeakCapability:
    """A surfaced weak capability: a failing, recurring cluster, reported at the
    most specific tree level where the failure is both clear and well-supported."""

    node_id: int
    score: float
    support: int
    n_questions: int
    level: int
    keywords: list[str]
    example_rubric: str
    rubric_lines: list[str]


@dataclass
class CapabilityDiagnosis:
    n_probes: int
    overall_score: float
    tree: list[CapabilityNode]
    weak_capabilities: list[WeakCapability]


def _score(possible: int, earned: int) -> float:
    return earned / possible if possible else 0.0


def _keywords(member_indices: list[int], tokens_by_probe: list[list[str]], k: int = 5) -> list[str]:
    df: Counter[str] = Counter()
    for i in member_indices:
        df.update(set(tokens_by_probe[i]))
    return [term for term, _ in df.most_common(k)]


def _representative(member_indices: list[int], probes: list[CapabilityProbe]) -> str:
    lines = [probes[i].rubric_text for i in member_indices if probes[i].rubric_text]
    return min(lines, key=len) if lines else ""


def _cluster_distance_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def diagnose_weak_capabilities(
    probes: list[CapabilityProbe],
    *,
    weak_threshold: float = 0.5,
    min_support: int = 2,
) -> CapabilityDiagnosis:
    """Run CRAFT's diagnostic core over a set of capability probes.

    Builds the hierarchical capability tree (average-linkage agglomerative
    clustering on bag-of-words cosine distance), scores the model at every node
    as the fraction of rubric points earned inside it, and selects the weakest
    nodes at the deepest level that still meets ``min_support`` — i.e. at the
    granularity where each failure is clearest. ``weak_threshold`` is the
    earned-points fraction below which a node counts as weak (default 0.5: the
    model fails the capability more often than it passes).
    """
    if not probes:
        return CapabilityDiagnosis(n_probes=0, overall_score=0.0, tree=[], weak_capabilities=[])

    tokens_by_probe = [_tokenize(p.rubric_text) for p in probes]
    vectors = [_term_freq(tokens) for tokens in tokens_by_probe]

    total_possible = sum(p.points for p in probes)
    total_earned = sum(p.points for p in probes if p.earned)
    overall = _score(total_possible, total_earned)

    # Leaves: one node per probe.
    nodes: dict[int, CapabilityNode] = {}
    # Mutable cluster state used only while merging.
    state: dict[int, dict] = {}
    next_id = 0
    for i, p in enumerate(probes):
        cid = next_id
        next_id += 1
        nodes[cid] = CapabilityNode(
            node_id=cid,
            member_indices=[i],
            level=0,
            score=_score(p.points, p.points if p.earned else 0),
            support=1,
            n_questions=1,
            keywords=_keywords([i], tokens_by_probe),
            example_rubric=p.rubric_text,
            children=(),
        )
        state[cid] = {
            "size": 1,
            "possible": p.points,
            "earned": p.points if p.earned else 0,
            "qids": {p.question_id},
            "level": 0,
        }

    # Initial pairwise distances = 1 - cosine similarity between singleton probes.
    leaf_ids = list(state)
    dist: dict[tuple[int, int], float] = {}
    for ai in range(len(leaf_ids)):
        for bi in range(ai + 1, len(leaf_ids)):
            a, b = leaf_ids[ai], leaf_ids[bi]
            sim = _cosine_similarity(vectors[a], vectors[b])
            dist[_cluster_distance_key(a, b)] = 1.0 - sim

    active = set(state)
    while len(active) > 1:
        # Closest surviving pair (average linkage is monotone, so greedy is correct).
        best_key = min(
            (k for k in dist if k[0] in active and k[1] in active),
            key=lambda k: dist[k],
        )
        a, b = best_key
        sa, sb = state[a], state[b]
        new_id = next_id
        next_id += 1
        members = nodes[a].member_indices + nodes[b].member_indices
        possible = sa["possible"] + sb["possible"]
        earned = sa["earned"] + sb["earned"]
        qids = sa["qids"] | sb["qids"]
        level = max(sa["level"], sb["level"]) + 1
        nodes[new_id] = CapabilityNode(
            node_id=new_id,
            member_indices=members,
            level=level,
            score=_score(possible, earned),
            support=len(members),
            n_questions=len(qids),
            keywords=_keywords(members, tokens_by_probe),
            example_rubric=_representative(members, probes),
            children=(a, b),
        )
        state[new_id] = {
            "size": sa["size"] + sb["size"],
            "possible": possible,
            "earned": earned,
            "qids": qids,
            "level": level,
        }

        # Lance-Williams update for average linkage: d(C, A∪B) is the size-weighted
        # mean of d(C, A) and d(C, B).
        size_sum = sa["size"] + sb["size"]
        for c in active:
            if c in (a, b):
                continue
            da = dist[_cluster_distance_key(a, c)]
            db = dist[_cluster_distance_key(b, c)]
            dist[_cluster_distance_key(new_id, c)] = (sa["size"] * da + sb["size"] * db) / size_sum
        for c in list(active):
            dist.pop(_cluster_distance_key(a, c), None)
            dist.pop(_cluster_distance_key(b, c), None)
        active.discard(a)
        active.discard(b)
        active.add(new_id)

    tree = list(nodes.values())

    # Dynamic weak-node selection: keep a node only if it is weak AND well
    # supported AND no strict descendant is already weak — that descendant is a
    # more specific statement of the same failure, so reporting the ancestor
    # would only blur it. This is CRAFT's "granularity where the failure is
    # clearest".
    weak_ids = {
        nd.node_id for nd in tree if nd.support >= min_support and nd.score < weak_threshold
    }
    children_map: dict[int, tuple[int, ...]] = {nd.node_id: nd.children for nd in tree}

    def has_weak_descendant(node_id: int) -> bool:
        stack = list(children_map.get(node_id, ()))
        while stack:
            x = stack.pop()
            if x in weak_ids:
                return True
            stack.extend(children_map.get(x, ()))
        return False

    selected = [nd for nd in tree if nd.node_id in weak_ids and not has_weak_descendant(nd.node_id)]

    weak: list[WeakCapability] = []
    for nd in selected:
        lines = sorted({probes[i].rubric_text for i in nd.member_indices})
        weak.append(
            WeakCapability(
                node_id=nd.node_id,
                score=nd.score,
                support=nd.support,
                n_questions=nd.n_questions,
                level=nd.level,
                keywords=nd.keywords,
                example_rubric=nd.example_rubric,
                rubric_lines=lines,
            )
        )
    # Weakest first; on ties, prefer the most specific (smallest) cluster.
    weak.sort(key=lambda w: (w.score, w.support))

    return CapabilityDiagnosis(
        n_probes=len(probes),
        overall_score=overall,
        tree=tree,
        weak_capabilities=weak,
    )


def format_report(diagnosis: CapabilityDiagnosis) -> str:
    """Human-readable summary of the diagnosis — what the model cannot do and at
    what granularity, in the spirit of CRAFT's capability report."""
    lines = [
        f"CRAFT capability diagnosis — {diagnosis.n_probes} rubric probes, "
        f"overall score {round(diagnosis.overall_score * 100):.0f}%, "
        f"{len(diagnosis.weak_capabilities)} weak capability cluster(s).",
    ]
    if not diagnosis.weak_capabilities:
        lines.append("No weak capabilities above the support threshold.")
        return "\n".join(lines)
    for i, w in enumerate(diagnosis.weak_capabilities, 1):
        lines.append(
            f"\n{i}. [{round(w.score * 100):.0f}% earned, {w.support} probes across "
            f"{w.n_questions} question(s), level {w.level}] "
            f"{', '.join(w.keywords) or '(no shared keywords)'}"
        )
        lines.append(f'   e.g. "{w.example_rubric}"')
    return "\n".join(lines)
