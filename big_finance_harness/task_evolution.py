"""Task evolution — synthesize deep-research items from simple QA.

Adapted (Mode 2) from "From Simple QA to Deep Research: A Verifiable Benchmark
Constructed through Iterative Task Evolution" (arXiv:2608.02163). The paper
iteratively transforms a simple question into a deep-research task through an
Explorer -> Formalizer -> Challenger pipeline, representing each task as a DAG of
atomic checkpoints and reducing that DAG to a structure-weighted rubric.

This module ports that pipeline at full fidelity while substituting the paper's
own LLM backend with this harness's native `ModelClient` — the same LiteLLM-backed
abstraction the ReAct agent and the judge already use. The DAG of checkpoints
reduces to the repo's existing `RubricLine` (text + points), so every evolved item
is a plain `DatasetItem` that drops into `scripts/run_eval_set.py` with **no
schema change** — exactly the data contract the benchmark already speaks.

Intentionally scoped out (auxiliary, not core):
  - The paper's standalone deep-research benchmark suite and its own eval harness.
    Grading of evolved items belongs to this repo's existing eval/grade pipeline.
  - The paper's 31-topic / 10-category taxonomy. This harness is a single
    financial-research domain; we evolve within it rather than importing the
    taxonomy.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from big_finance_harness.models.base import ModelClient, ThinkingLevel
from big_finance_harness.types import DatasetItem, Message, RubricLine, TextBlock

Phase = str  # one of "explore" | "formalize" | "challenge"

_EXPLORER_SYSTEM = (
    "You are the Explorer stage of a deep-research task constructor. Given a simple "
    "financial question, surface the related data points, entities, dates, and filings a "
    "thorough analyst would need to resolve a harder, multi-step version of it. Return "
    "ONLY the JSON object specified; no prose."
)

_FORMALIZER_SYSTEM = (
    "You are the Formalizer stage of a deep-research task constructor. Given a simple "
    "financial question and the Explorer's findings, compose a more complex multi-step "
    "research query and decompose it into an ORDERED list of atomic, independently "
    "verifiable checkpoints. Each checkpoint's depends_on lists the 0-based indices of "
    "the checkpoints whose results it builds on (earlier indices only). Return ONLY the "
    "JSON object specified; no prose."
)

_CHALLENGER_SYSTEM = (
    "You are the Challenger stage of a deep-research task constructor. Given a draft "
    "deep-research query and its checkpoint DAG, add verification constraints (e.g. cite "
    "the filing, show the arithmetic, round per a stated convention, cross-check a second "
    "source) and up-weight the checkpoints whose synthesis carries the most analytic load. "
    "Return ONLY the JSON object specified; no prose."
)


class Checkpoint(BaseModel):
    """One atomic, independently verifiable analyst step = one DAG node."""

    text: str
    base_points: int = 1
    depends_on: list[int] = Field(default_factory=list)


class CheckpointDAG(BaseModel):
    """A DAG of checkpoints reduced to a structure-weighted rubric.

    Structure weighting (the paper's "structure-weighted rubric") assigns each node
    points = base_points + depth, where depth is the longest dependency path from a
    root. Synthesis steps that sit on top of many inputs end up worth more than raw
    lookups — matching how the benchmark's own expert rubrics weight the final
    calculation above its constituent recordings.
    """

    checkpoints: list[Checkpoint]

    def _normalized_deps(self, i: int) -> list[int]:
        cp = self.checkpoints[i]
        return [
            d
            for d in cp.depends_on
            if isinstance(d, int) and 0 <= d < len(self.checkpoints) and d != i
        ]

    def depth(self, i: int, _seen: frozenset[int] | None = None) -> int:
        seen = _seen or frozenset()
        if i in seen:  # cycle guard — treat a back-edge as terminal
            return 0
        next_seen = seen | {i}
        deps = self._normalized_deps(i)
        if not deps:
            return 0
        return 1 + max(self.depth(d, next_seen) for d in deps)

    def to_rubric(self) -> list[RubricLine]:
        """Reduce the DAG to the repo's `RubricLine` schema, structure-weighted."""
        return [
            RubricLine(text=cp.text, points=max(1, cp.base_points + self.depth(i)))
            for i, cp in enumerate(self.checkpoints)
        ]


class EvolutionStep(BaseModel):
    """Provenance for one pipeline phase — the paper emphasizes traceable verification."""

    phase: Phase
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)


class EvolvedItem(BaseModel):
    """An evolved deep-research item plus the DAG/trace behind it.

    `item` is a first-class `DatasetItem` — load it through the same JSONL path the
    rest of the harness uses. `dag` and `evolution` are provenance only.
    """

    item: DatasetItem
    dag: CheckpointDAG
    source_id: str
    evolution: list[EvolutionStep] = Field(default_factory=list)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first balanced {...} object out of a model response.

    Models occasionally wrap JSON in ```json fences or add stray prose; this finds
    the first balanced object rather than requiring a clean top-level document.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


async def _phase(
    client: ModelClient,
    system: str,
    user: str,
    thinking: ThinkingLevel,
) -> str:
    """One LLM call for a pipeline phase. No tools — pure text/JSON generation."""
    response = await client.chat(
        system=system,
        messages=[Message(role="user", content=[TextBlock(text=user)])],
        tools=[],
        temperature=0.0,
        thinking=thinking,
        max_output_tokens=4096,
    )
    return response.text


def _dag_from_source(item: DatasetItem) -> CheckpointDAG:
    """Graceful fallback: chain the source rubric lines into a linear DAG."""
    checkpoints: list[Checkpoint] = []
    for i, line in enumerate(item.rubric):
        checkpoints.append(
            Checkpoint(
                text=line.text,
                base_points=max(1, line.points),
                depends_on=[i - 1] if i > 0 else [],
            )
        )
    return CheckpointDAG(checkpoints=checkpoints)


def _build_dag(
    formalized: dict[str, Any] | None, challenger: dict[str, Any] | None
) -> CheckpointDAG:
    """Assemble the final DAG from the Formalizer draft plus Challenger overrides."""
    raw = (formalized or {}).get("checkpoints")
    if not isinstance(raw, list) or not raw:
        # Formalizer gave us nothing usable — keep the core pipeline honest by
        # falling back to a structure-weighted version of the source rubric.
        return CheckpointDAG(checkpoints=[])
    overrides = (challenger or {}).get("weight_overrides")
    overrides = overrides if isinstance(overrides, dict) else {}
    checkpoints: list[Checkpoint] = []
    for idx, node in enumerate(raw):
        if not isinstance(node, dict):
            continue
        text = str(node.get("text", "")).strip()
        if not text:
            continue
        deps = node.get("depends_on", [])
        deps = [d for d in deps if isinstance(d, int)] if isinstance(deps, list) else []
        base = node.get("base_points", 1)
        try:
            base = int(base)
        except (TypeError, ValueError):
            base = 1
        # Challenger may up-weight a checkpoint by synthesis load.
        ov = overrides.get(str(idx)) or overrides.get(idx)
        if isinstance(ov, int):
            base = ov
        elif isinstance(ov, dict) and isinstance(ov.get("base_points"), int):
            base = ov["base_points"]
        checkpoints.append(Checkpoint(text=text, base_points=max(1, base), depends_on=deps))
    return CheckpointDAG(checkpoints=checkpoints)


def _user_prompt_for_item(item: DatasetItem) -> str:
    rubric = "\n".join(f"- [{r.points}pt] {r.text}" for r in item.rubric)
    return (
        f"Simple question:\n{item.query}\n\n"
        f"Reference answer:\n{item.reference_answer}\n\n"
        f"Existing rubric checkpoints:\n{rubric or '(none)'}\n\n"
        f"Sources: {', '.join(item.sources) or '(none)'}"
    )


async def evolve_item(
    *,
    item: DatasetItem,
    client: ModelClient,
    thinking: ThinkingLevel = "off",
) -> EvolvedItem:
    """Run the Explorer -> Formalizer -> Challenger pipeline on one simple item.

    Returns an `EvolvedItem` whose `.item` is a pipeline-compatible `DatasetItem`
    (new id, evolved query, structure-weighted DAG rubric) and whose `.dag` /
    `.evolution` carry the construction provenance. Degrades gracefully: if any
    phase returns unparseable JSON, the pipeline keeps the source's grounded
    reference answer and falls back to a structure-weighted version of the
    source rubric rather than raising.
    """
    base_prompt = _user_prompt_for_item(item)
    steps: list[EvolutionStep] = []

    # 1. Explore — surface related data a deeper version would need.
    exploration = _extract_json(
        await _phase(
            client,
            _EXPLORER_SYSTEM,
            base_prompt + '\n\nReturn JSON: {"related_points": [str, ...], "summary": str}',
            thinking,
        )
    )
    related = exploration.get("related_points", []) if exploration else []
    related = [str(p) for p in related if isinstance(p, str)] if isinstance(related, list) else []
    steps.append(
        EvolutionStep(
            phase="explore",
            summary=(exploration or {}).get("summary", "")
            if exploration
            else "(unparseable; no exploration)",
            detail={"related_points": related},
        )
    )

    # 2. Formalize — compose a harder query + decompose into a checkpoint DAG.
    explore_block = "\n".join(f"- {p}" for p in related) or "(none)"
    formalized = _extract_json(
        await _phase(
            client,
            _FORMALIZER_SYSTEM,
            base_prompt
            + f"\n\nExplorer's related points:\n{explore_block}\n\n"
            + 'Return JSON: {"evolved_query": str, "checkpoints": '
            '[{"text": str, "depends_on": [int], "base_points": int}]}',
            thinking,
        )
    )
    evolved_query = str((formalized or {}).get("evolved_query", "")).strip()
    steps.append(
        EvolutionStep(
            phase="formalize",
            summary=evolved_query or "(unparseable; reused source query)",
            detail={"checkpoint_drafts": len((formalized or {}).get("checkpoints", []) or [])},
        )
    )

    # 3. Challenge — add verification constraints + up-weight synthesis nodes.
    draft_query = evolved_query or item.query
    challenger = _extract_json(
        await _phase(
            client,
            _CHALLENGER_SYSTEM,
            f"Draft query:\n{draft_query}\n\nCheckpoints:\n"
            + "\n".join(
                f"- {c['text']}"
                for c in ((formalized or {}).get("checkpoints") or [])
                if isinstance(c, dict)
            ),
            thinking,
        )
    )
    constraints = (challenger or {}).get("constraints", [])
    constraints = (
        [str(c) for c in constraints if isinstance(c, str)] if isinstance(constraints, list) else []
    )
    steps.append(
        EvolutionStep(
            phase="challenge",
            summary="; ".join(constraints) if constraints else "(unparseable; no constraints)",
            detail={"constraints": constraints},
        )
    )

    # Assemble the final structure-weighted DAG.
    dag = _build_dag(formalized, challenger)
    if not dag.checkpoints:
        dag = _dag_from_source(item)

    # Compose the final deep-research query: evolved query + explicit constraints.
    final_query = evolved_query or item.query
    if constraints:
        final_query = final_query + " Constraints: " + "; ".join(constraints)
    reference_framing = str((challenger or {}).get("reference_framing", "")).strip()
    reference_answer = item.reference_answer
    if reference_framing:
        reference_answer = f"{item.reference_answer} ({reference_framing})"

    evolved = DatasetItem(
        # New id signals this is a synthesized deep-research item, not the source.
        id=f"{item.id}.evo",
        query=final_query,
        reference_answer=reference_answer,
        rubric=dag.to_rubric(),
        annotator_notes=(
            f"Evolved from {item.id} via Explorer-Formalizer-Challenger "
            "(arXiv:2608.02163). Rubric is structure-weighted by checkpoint DAG depth."
        ),
        sources=list(item.sources),
    )
    return EvolvedItem(item=evolved, dag=dag, source_id=item.id, evolution=steps)


__all__ = [
    "Checkpoint",
    "CheckpointDAG",
    "EvolvedItem",
    "EvolutionStep",
    "evolve_item",
]
