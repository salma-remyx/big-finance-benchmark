"""Tests for the CRAFT capability diagnosis.

These exercise the diagnosis through the harness's own grading contract:
``GradedRun`` / ``GradedRubricLine`` (from ``big_finance_harness.types``) are
exactly what ``grader.grade`` writes, so these tests prove the new analysis
layer integrates with the real evaluation output — not a self-test of dead code.
"""

from __future__ import annotations

from big_finance_harness.capability_diagnosis import (
    diagnose_weak_capabilities,
    format_report,
    probes_from_grades,
)
from big_finance_harness.types import GradedRubricLine, GradedRun

# EDGAR-retrieval criteria, worded tightly so they cluster together. Every one
# of them is FAILED by the model — the recurring weakness the diagnosis should
# surface.
_EDGAR_RUBRICS = [
    "Pulls the 10-K filing from EDGAR",
    "Fetches the 10-K filing from EDGAR",
    "Retrieves the 10-K filing from EDGAR",
]
# Computation criteria that the model PASSES — these must NOT be flagged.
_COMPUTATION_RUBRICS = [
    "Computes the net margin ratio",
    "Divides price by earnings",
    "Calculates the P/E ratio",
]


def _graded(qid: str, lines: list[tuple[str, int, bool]]) -> GradedRun:
    rubric_lines = [GradedRubricLine(text=t, points=p, earned=e) for t, p, e in lines]
    possible = sum(p for _, p, _ in lines)
    earned = sum(p for _, p, e in lines if e)
    return GradedRun(
        question_id=qid,
        trial_idx=0,
        model="anthropic:claude-test",
        judge="vertex:gemini-test",
        final_answer="$0",
        reference_answer="$0",
        final_answer_correct=False,
        rubric_lines=rubric_lines,
        rubric_points_earned=earned,
        rubric_points_possible=possible,
        rubric_lines_earned=sum(1 for _, _, e in lines if e),
        rubric_lines_possible=len(lines),
    )


def _mixed_runs() -> list[GradedRun]:
    """Three questions, each with a failed EDGAR criterion and a passed calc one."""
    runs: list[GradedRun] = []
    for i, (edgar, calc) in enumerate(zip(_EDGAR_RUBRICS, _COMPUTATION_RUBRICS, strict=True)):
        runs.append(_graded(f"bf-edgar-{i}", [(edgar, 1, False), (calc, 1, True)]))
    return runs


def test_surfaces_recurring_weak_capability_and_ignores_passed() -> None:
    runs = _mixed_runs()
    probes = probes_from_grades(runs)
    diagnosis = diagnose_weak_capabilities(probes)

    # A weak capability must be surfaced.
    assert diagnosis.weak_capabilities, "expected at least one weak capability"

    # It must be the EDGAR-retrieval weakness: some surfaced cluster is labelled
    # by EDGAR vocabulary and contains only FAILED rubric lines.
    edgar_weak = [w for w in diagnosis.weak_capabilities if "edgar" in w.keywords]
    assert edgar_weak, f"EDGAR weakness not surfaced; got {diagnosis.weak_capabilities}"
    for w in edgar_weak:
        assert w.score == 0.0
        assert all(any(line in r for r in _EDGAR_RUBRICS) for line in w.rubric_lines)

    # The computation capabilities all passed, so no weak cluster may reference
    # their vocabulary.
    all_keywords = {kw for w in diagnosis.weak_capabilities for kw in w.keywords}
    assert not (all_keywords & {"margin", "ratio", "earnings"}), all_keywords


def test_all_passing_yields_no_weak_capabilities() -> None:
    runs = [_graded("bf-ok-0", [(r, 1, True) for r in _EDGAR_RUBRICS])]
    diagnosis = diagnose_weak_capabilities(probes_from_grades(runs))
    assert diagnosis.weak_capabilities == []
    assert diagnosis.overall_score == 1.0


def test_empty_input_is_safe() -> None:
    diagnosis = diagnose_weak_capabilities([])
    assert diagnosis.n_probes == 0
    assert diagnosis.weak_capabilities == []
    # The report must render without raising on empty input.
    assert "0 rubric probes" in format_report(diagnosis)


def test_overall_score_matches_rubric_points() -> None:
    runs = _mixed_runs()  # 3 EDGAR (failed) + 3 calc (passed) = 3/6 points earned
    diagnosis = diagnose_weak_capabilities(probes_from_grades(runs))
    assert diagnosis.n_probes == 6
    assert diagnosis.overall_score == 0.5


def test_tree_is_hierarchical_with_levels() -> None:
    # CRAFT's "hierarchical capability tree" — every non-leaf node's level is
    # deeper than its children, so levels form a genuine tree.
    runs = _mixed_runs()
    diagnosis = diagnose_weak_capabilities(probes_from_grades(runs))
    assert len(diagnosis.tree) > len(probes_from_grades(runs))  # internal nodes present
    by_id = {nd.node_id: nd for nd in diagnosis.tree}
    for nd in diagnosis.tree:
        for child in nd.children:
            assert by_id[child].level < nd.level
