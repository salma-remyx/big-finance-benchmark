"""Task-level LLM router — admission-time bandit selection with per-task pinning.

Adapted from TRACE-ROUTER (arXiv:2607.22465), a task-level routing framework that:

  * assigns each task to a single backend **once at admission**,
  * **pins** every subsequent LLM call in that task's trajectory to that backend, and
  * updates its routing policy from the task's **delayed terminal reward**, jointly
    accounting for accuracy and latency rather than from per-call feedback.

Delivered as a drop-in `ModelClient`: anywhere the harness injects a
`client: ModelClient` (notably `run_question` in `big_finance_harness.agent`), a
`TaskRouterClient` can be substituted with no other code change. The router selects an
arm the first time it sees a question, delegates every later step of that question to
the same arm, and exposes `feedback()` so the caller can fold the grader's binary verdict
plus the trajectory latency back into the policy as the delayed terminal reward.

Adaptation choices (Mode 2 — adapted port):
  * Core mechanism kept at fidelity: a LinUCB contextual bandit (Li et al., 2010) — a
    standard instantiation of the paper's "contextual bandit" — with admission-time
    selection, per-task pinning, and a delayed accuracy+latency reward.
  * Auxiliary substitution: the paper's context representation is replaced by a
    parameter-free feature vector derived from the question text (length, numeric
    density, finance/EDGAR keyword signal). No learned embedding or explicit
    task-complexity estimator is introduced — the paper deliberately avoids the latter.
  * Cut: the paper's separate agentic-benchmark evaluation suite. This module is the
    routing capability; wiring it into a full benchmark sweep and reporting Pareto
    numbers belongs in a downstream PR.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from big_finance_harness.models.base import ModelClient, ThinkingLevel
from big_finance_harness.types import Message, ModelResponse, TextBlock, ToolSpec

_NUM_RE = re.compile(r"\b\d[\d,.․]*\b")
# Surface-level signal that a question leans on retrieval / computation the stronger
# backend is more likely to earn its cost on. These are proxies, not a complexity model.
_KEYWORDS = frozenset(
    "edgar sec 10-k 10q 10-q filing fiscal revenue ebitda ebit balance cash "
    "ratio calculate compute annual quarterly ticker shares debt margin".split()
)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve ``matrix @ x = vector`` via Gauss-Jordan with partial pivoting.

    The bandit's ridge regularization keeps ``matrix`` positive-definite, so a pivot is
    always available; the nudge below only guards against pathological float collapse.
    """
    n = len(matrix)
    aug = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[pivot][col] += 1e-9
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / pivot_val
            if factor:
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _rank_one_update(matrix: list[list[float]], x: list[float], coeff: float) -> None:
    """In-place ``matrix += coeff * (x outer x)``."""
    for i, xi in enumerate(x):
        if xi == 0.0:
            continue
        row = matrix[i]
        for j, xj in enumerate(x):
            row[j] += coeff * xi * xj


def _context_features(text: str) -> list[float]:
    """Parameter-free context vector for a question (the paper's context proxy)."""
    words = text.split()
    word_count = len(words)
    numeric_density = len(_NUM_RE.findall(text)) / max(word_count, 1)
    keyword_hits = sum(1 for w in words if w.lower().strip(".,?;:()") in _KEYWORDS)
    return [
        1.0,  # bias
        min(word_count / 80.0, 1.0),  # length proxy
        min(numeric_density * 5.0, 1.0),  # computation proxy
        min(keyword_hits / 3.0, 1.0),  # finance / EDGAR signal
        min(len(text) / 600.0, 1.0),  # char-length proxy
    ]


def _first_user_text(messages: list[Message]) -> str:
    """The task signature: the first user message, stable across a trajectory's steps."""
    for m in messages:
        if m.role != "user":
            continue
        return "".join(b.text for b in m.content if isinstance(b, TextBlock))
    return ""


@dataclass
class LinUCBBandit:
    """Disjoint LinUCB contextual bandit over a fixed set of arms.

    Each arm holds a ridge-regularized linear model of the reward given the context.
    Selection returns the arm with the highest upper-confidence bound on its predicted
    reward; updates fold an observed reward back into the arm that was selected.
    """

    arm_names: list[str]
    alpha: float = 1.0
    ridge_lambda: float = 1.0
    _a: dict[str, list[list[float]]] = field(default_factory=dict)
    _b: dict[str, list[float]] = field(default_factory=dict)
    _dim: int = 0

    def __post_init__(self) -> None:
        if not self.arm_names:
            raise ValueError("LinUCBBandit requires at least one arm")
        for name in self.arm_names:
            self._a.setdefault(name, [])
            self._b.setdefault(name, [])

    def _ensure(self, context: list[float]) -> None:
        # Initialize lazily on first observation so the feature dimension is fixed by
        # real data rather than guessed. Until then every arm is equally plausible.
        dim = len(context)
        if self._dim == 0:
            self._dim = dim
        elif self._dim != dim:
            raise ValueError(f"context dimension {dim} != initialized {self._dim}")
        for name in self.arm_names:
            if not self._a[name]:
                self._a[name] = [
                    [self.ridge_lambda if i == j else 0.0 for j in range(dim)] for i in range(dim)
                ]
                self._b[name] = [0.0] * dim

    def select(self, context: list[float]) -> str:
        self._ensure(context)
        best_name = self.arm_names[0]
        best_score = float("-inf")
        for name in self.arm_names:
            theta = _solve(self._a[name], self._b[name])
            z = _solve(self._a[name], context)
            mean = _dot(theta, context)
            variance = max(_dot(context, z), 0.0)
            score = mean + self.alpha * (variance**0.5)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name

    def update(self, arm: str, context: list[float], reward: float) -> None:
        if arm not in self._a:
            raise KeyError(f"unknown arm {arm!r}")
        self._ensure(context)
        _rank_one_update(self._a[arm], context, 1.0)
        b = self._b[arm]
        for i, xi in enumerate(context):
            b[i] += reward * xi

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_names": list(self.arm_names),
            "alpha": self.alpha,
            "ridge_lambda": self.ridge_lambda,
            "dim": self._dim,
            "a": {n: self._a[n] for n in self.arm_names},
            "b": {n: self._b[n] for n in self.arm_names},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinUCBBandit:
        bandit = cls(
            arm_names=list(data["arm_names"]),
            alpha=float(data.get("alpha", 1.0)),
            ridge_lambda=float(data.get("ridge_lambda", 1.0)),
        )
        bandit._dim = int(data.get("dim", 0))
        for name in bandit.arm_names:
            bandit._a[name] = [list(row) for row in data.get("a", {}).get(name, [])]
            bandit._b[name] = list(data.get("b", {}).get(name, []))
        return bandit


@dataclass(frozen=True)
class _Arm:
    name: str
    client: ModelClient


@dataclass
class _TaskStats:
    arm_name: str | None = None
    latency_seconds: float = 0.0
    cost_usd: float = 0.0


class TaskRouterClient(ModelClient):
    """A `ModelClient` that routes each task to one backend and pins it for the task.

    On the first call for a question the router selects an arm with its contextual
    bandit; every later call for that question is delegated to the same arm (per-task
    pinning). After the task is graded, call `feedback()` to fold the binary correctness
    verdict plus the trajectory latency into the bandit as the delayed terminal reward.
    """

    def __init__(
        self,
        arms: list[_Arm],
        *,
        alpha: float = 1.0,
        ridge_lambda: float = 1.0,
        accuracy_weight: float = 0.7,
        latency_weight: float = 0.3,
        latency_scale_s: float = 30.0,
        policy_path: str | Path | None = None,
        bandit: LinUCBBandit | None = None,
    ) -> None:
        if len(arms) < 2:
            raise ValueError("TaskRouterClient needs at least two arms to route between")
        if abs(accuracy_weight + latency_weight - 1.0) > 1e-9:
            raise ValueError("accuracy_weight + latency_weight must sum to 1.0")
        names = [a.name for a in arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        self._arms = arms
        self._arms_by_name = {a.name: a for a in arms}
        self.accuracy_weight = accuracy_weight
        self.latency_weight = latency_weight
        self.latency_scale_s = latency_scale_s
        self.policy_path = Path(policy_path) if policy_path else None
        self._bandit = bandit or LinUCBBandit(
            arm_names=names, alpha=alpha, ridge_lambda=ridge_lambda
        )
        if self._bandit.arm_names != names:
            raise ValueError("bandit arm names must match the supplied arms")
        self._pins: dict[str, str] = {}
        self._stats: dict[str, _TaskStats] = {}
        self._active_name: str | None = None
        if self.policy_path and self.policy_path.exists():
            self._load()

    @property
    def snapshot(self) -> str:
        if self._active_name is not None:
            return self._arms_by_name[self._active_name].client.snapshot
        return "router[" + "|".join(a.name for a in self._arms) + "]"

    @property
    def num_retries(self) -> int:
        if self._active_name is not None:
            return getattr(self._arms_by_name[self._active_name].client, "num_retries", 0)
        return 0

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float | None = None,
        thinking: ThinkingLevel = "off",
        max_output_tokens: int = 65536,
    ) -> ModelResponse:
        question = _first_user_text(messages)
        arm_name = self._pins.get(question)
        if arm_name is None:
            # Admission: select once, then pin every subsequent step of this task.
            arm_name = self._bandit.select(_context_features(question))
            self._pins[question] = arm_name
        arm = self._arms_by_name[arm_name]
        self._active_name = arm_name
        stats = self._stats.setdefault(question, _TaskStats())
        stats.arm_name = arm_name

        started = time.monotonic()
        response = await arm.client.chat(
            system=system,
            messages=messages,
            tools=tools,
            temperature=temperature,
            thinking=thinking,
            max_output_tokens=max_output_tokens,
        )
        stats.latency_seconds += time.monotonic() - started
        if response.cost_usd is not None:
            stats.cost_usd += response.cost_usd
        return response

    def reward(self, correct: bool, latency_seconds: float) -> float:
        """Scalar reward in [0, 1]: accuracy-dominant, penalized by latency."""
        latency_term = 1.0 - min(max(latency_seconds, 0.0) / self.latency_scale_s, 1.0)
        accuracy_term = 1.0 if correct else 0.0
        return self.accuracy_weight * accuracy_term + self.latency_weight * latency_term

    def feedback(
        self,
        question: str,
        correct: bool,
        *,
        latency_seconds: float | None = None,
    ) -> float:
        """Apply the delayed terminal reward for a routed task.

        ``question`` is the task text (the first user message the router pinned on).
        ``correct`` is the grader's binary verdict; ``latency_seconds`` defaults to the
        wallclock the router observed for the task but may be overridden (e.g. with
        ``RunRecord.total_wallclock_seconds``). Returns the reward that was applied.
        """
        stats = self._stats.get(question)
        if stats is None or stats.arm_name is None:
            raise ValueError(
                f"no routing decision recorded for question {question!r}; the router "
                "must serve the task before feedback is applied"
            )
        latency = stats.latency_seconds if latency_seconds is None else latency_seconds
        reward = self.reward(correct, latency)
        self._bandit.update(stats.arm_name, _context_features(question), reward)
        self._save()
        return reward

    def selected_arm_for(self, question: str) -> str | None:
        return self._pins.get(question)

    def _save(self) -> None:
        if not self.policy_path:
            return
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_path.write_text(json.dumps(self._bandit.to_dict()))

    def _load(self) -> None:
        if not self.policy_path:
            return
        try:
            data = json.loads(self.policy_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        candidate = LinUCBBandit.from_dict(data)
        # Only reuse a policy recorded for the same arm set; otherwise start fresh.
        if candidate.arm_names == [a.name for a in self._arms]:
            self._bandit = candidate


def make_router(
    model_ids: list[str],
    *,
    arm_names: list[str] | None = None,
    **kwargs: Any,
) -> TaskRouterClient:
    """Build a `TaskRouterClient` whose arms are `LiteLLMClient`s for each model id.

    ``arm_names`` optionally labels arms (defaults to the model id). Other kwargs forward
    to `TaskRouterClient` (bandit hyperparameters, ``policy_path``, ...).
    """
    from big_finance_harness.models.base import LiteLLMClient

    if arm_names is None:
        arm_names = list(model_ids)
    if len(arm_names) != len(model_ids):
        raise ValueError("arm_names must match model_ids one-to-one")
    arms = [_Arm(name, LiteLLMClient(mid)) for name, mid in zip(arm_names, model_ids, strict=True)]
    return TaskRouterClient(arms, **kwargs)
