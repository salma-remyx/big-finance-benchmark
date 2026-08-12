"""Cross-trial final-answer consistency measurement.

Adapted from "What Current AI Benchmarks Leave Unmeasured: Modality, Search,
Citations, and Implications (for Safety Evaluations)" (arXiv:2608.06202). That
audit finds that repeated runs of the *same* prompt produce inconsistent
responses in up to 21% of prompts -- a behavioral dimension that single-run
accuracy hides. This module ports that one dimension onto the BigFinanceBench
harness, which already runs each (question, model) pair across ``trial_idx``
(3 trials by default; see the README methodology).

For every (question_id, model_label, judge) group with >= 2 graded trials we
ask whether the trials agreed, two ways:

  * text agreement         -- the normalized final-answer strings are identical
                               across trials (the paper's "response
                               consistency");
  * correctness agreement   -- the judge's ``final_answer_correct`` verdict is
                               identical across trials (a harness-native signal
                               the paper does not have: it works with one rater,
                               the harness has a rubric judge).

The headline number -- the fraction of groups whose trials DISAGREE on the
answer text -- is the direct analog of the paper's "X% of prompts produced
inconsistent responses across repeated runs".

This is a Mode 2 (adapted port): the core mechanism -- measure response
agreement across repeated runs of identical prompts and report an
inconsistency rate -- is kept at full fidelity. Substituted auxiliaries: the
paper's varying conditions (modality chat-UI vs API, web search on/off) have
no counterpart in this single-modality, fixed-tool harness, so the harness's
native ``trial_idx`` axis is the repeated-run condition instead; the paper's
response-text-similarity model is replaced by a parameter-free
normalize-and-compare heuristic; and BBQ/SafetyBench is replaced by
BigFinanceBench. The paper's other audited dimensions (modality, search
conditions, citation grounding, abstention) are intentionally out of scope --
there is no varying condition to measure on existing run data.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = [
    "ConsistencyRecord",
    "GroupConsistency",
    "ConsistencySummary",
    "normalize_final_answer",
    "compute_consistency",
    "read_grade_records",
]

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = ".,;:!?"
_CURRENCY = re.compile(r"[$£€]")
# A comma between a digit and a 3-digit group, e.g. the "," in "1,234".
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}\b)")


def normalize_final_answer(text: str | None) -> str:
    """Normalize a final answer for cross-trial equality comparison.

    Lowercases, collapses whitespace, strips surrounding quotes and trailing
    sentence punctuation, drops currency symbols, and removes thousands
    separators so that ``"$114.3 Billion."`` and ``"114.3 billion"`` compare
    equal. ``None`` / empty answers normalize to ``""`` -- a single "no
    answer" bucket that is distinct from any concrete answer.
    """
    if not text:
        return ""
    s = text.strip().lower()
    s = _WHITESPACE.sub(" ", s)
    s = _CURRENCY.sub("", s)
    # Thousands separators chain ("1,234,567"); repeat until stable.
    prev: str | None = None
    while prev != s:
        prev = s
        s = _THOUSANDS_COMMA.sub("", s)
    s = s.strip(" \"'`")
    s = s.rstrip(_TRAILING_PUNCT)
    return s.strip()


@dataclass(frozen=True)
class ConsistencyRecord:
    """One graded trial reduced to the fields consistency is measured on.

    A thin view over ``GradedRun`` so the metric stays decoupled from I/O and
    from pydantic: scripts build these from JSONL rows, tests build them from
    ``GradedRun``.
    """

    question_id: str
    model_label: str
    trial_idx: int
    judge: str
    final_answer: str | None
    final_answer_correct: bool

    @classmethod
    def from_grade_dict(cls, row: dict, model_label: str) -> "ConsistencyRecord | None":
        """Build a record from a parsed ``<label>.grades*.jsonl`` line.

        Returns ``None`` for rows without a ``question_id`` so the reader can
        skip them without raising (matching the lenient parsing in
        ``scripts/build_analysis_csv.py``).
        """
        qid = row.get("question_id")
        if not qid:
            return None
        return cls(
            question_id=str(qid),
            model_label=model_label,
            trial_idx=int(row.get("trial_idx", 0) or 0),
            judge=str(row.get("judge", "")),
            final_answer=row.get("final_answer"),
            final_answer_correct=bool(row.get("final_answer_correct")),
        )


@dataclass(frozen=True)
class GroupConsistency:
    """Agreement across the repeated trials of one (question, model, judge)."""

    question_id: str
    model_label: str
    judge: str
    n_trials: int
    text_consistent: bool  # all normalized final answers equal
    correctness_consistent: bool  # all final_answer_correct verdicts equal
    distinct_answers: int  # unique normalized answers across the trials
    answers: tuple[str, ...]  # normalized answers, in ascending trial order


@dataclass(frozen=True)
class ConsistencySummary:
    """Aggregate multi-run consistency over a set of graded trials."""

    n_groups: int
    n_text_inconsistent: int  # groups whose trials disagreed on answer text
    n_correctness_inconsistent: int
    text_inconsistency_rate: float  # paper's headline: n_text_inconsistent / n_groups
    correctness_inconsistency_rate: float
    per_group: tuple[GroupConsistency, ...]


def compute_consistency(records: Iterable[ConsistencyRecord]) -> ConsistencySummary:
    """Measure cross-trial agreement for every (question, model, judge) group.

    Groups with fewer than 2 trials are excluded from the summary denominator:
    a single trial trivially agrees with itself, and the paper reports the
    inconsistency rate only over prompts that were actually run more than once.
    Including single-trial groups would understate the rate.
    """
    groups: dict[tuple[str, str, str], list[ConsistencyRecord]] = defaultdict(list)
    for r in records:
        groups[(r.question_id, r.model_label, r.judge)].append(r)

    per: list[GroupConsistency] = []
    for (qid, model, judge), recs in sorted(groups.items()):
        recs = sorted(recs, key=lambda r: r.trial_idx)  # stable left-to-right by trial
        if len(recs) < 2:
            continue
        answers = tuple(normalize_final_answer(r.final_answer) for r in recs)
        corrects = [r.final_answer_correct for r in recs]
        distinct = len(set(answers))
        per.append(
            GroupConsistency(
                question_id=qid,
                model_label=model,
                judge=judge,
                n_trials=len(recs),
                text_consistent=distinct <= 1,
                correctness_consistent=len(set(corrects)) <= 1,
                distinct_answers=distinct,
                answers=answers,
            )
        )

    n = len(per)
    n_text_inc = sum(1 for g in per if not g.text_consistent)
    n_corr_inc = sum(1 for g in per if not g.correctness_consistent)
    return ConsistencySummary(
        n_groups=n,
        n_text_inconsistent=n_text_inc,
        n_correctness_inconsistent=n_corr_inc,
        text_inconsistency_rate=(n_text_inc / n) if n else 0.0,
        correctness_inconsistency_rate=(n_corr_inc / n) if n else 0.0,
        per_group=tuple(per),
    )


def read_grade_records(run_dir: str | Path) -> list[ConsistencyRecord]:
    """Load ``ConsistencyRecord`` from every ``<label>.grades*.jsonl`` in a run.

    Mirrors the grade-file parsing in ``scripts/build_analysis_csv.py``: the
    model label is parsed from the filename stem (``gpt55.grades.jsonl`` ->
    ``gpt55``; ``gpt55.grades.gemini.jsonl`` -> ``gpt55``), and malformed lines
    are skipped rather than failing the whole report.
    """
    run_dir = Path(run_dir)
    out: list[ConsistencyRecord] = []
    for path in sorted(set(run_dir.glob("*.grades*.jsonl"))):
        stem = path.stem  # "gpt55.grades" or "gpt55.grades.gemini"
        if ".grades." in stem:
            label = stem.split(".grades.", 1)[0]
        elif stem.endswith(".grades"):
            label = stem[: -len(".grades")]
        else:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = ConsistencyRecord.from_grade_dict(row, label)
            if rec is not None:
                out.append(rec)
    return out
