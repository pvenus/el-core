

"""
Ontology-ish action layer for the emotion simulation.

Goal
----
Given:
- a preferred direction vector (where the agent would like to move next)
- agent context (vars, current/comfort vectors, turn, etc.)

Return:
- a ranked list of "actions" the agent is allowed to take

This module is designed so that:
- A "no-reasoner" mode works out of the box (fast, code-driven).
- A "reasoner" mode can coexist via an adapter interface.

Key idea
--------
Vector similarity proposes candidates (soft, continuous),
Ontology/constraints decide executability (hard, symbolic).

You can plug this into:
- RL action masking (allowed actions)
- A planner that selects among allowed actions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

# Reuse the same tiny vector helpers style as other modules
try:
    # src/test layout
    from sim_agent import normalize
except Exception:  # pragma: no cover
    # if you later move modules under a package, keep a local fallback
    def normalize(v: List[float], eps: float = 1e-12) -> List[float]:
        n = sum(x * x for x in v) ** 0.5
        if n < eps:
            return [0.0 for _ in v]
        return [x / n for x in v]


# -----------------------------
# Vector similarity
# -----------------------------

def cosine_similarity(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> float:
    """Cosine similarity in [-1, 1]. Works best if inputs are normalized."""
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na < eps or nb < eps:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


# -----------------------------
# Context + Action types
# -----------------------------

@dataclass
class AgentContext:
    """Inputs the ontology layer may use.

    You can extend this as needed (traits, tags, inventory, etc.).
    """
    vars: Dict[str, float] = field(default_factory=dict)
    turn: int = 0
    # optional: live vectors (if you want conditions like "dist > x")
    current_vec: Optional[List[float]] = None
    comfort_vec: Optional[List[float]] = None


@dataclass(frozen=True)
class Action:
    """An executable choice for the agent.

    effect_dir:
      - a *direction* vector describing where this action tends to move emotion
      - used for candidate ranking (soft)
      - normalized on construction

    precond:
      - symbolic constraints (hard). A reasoner can validate these.
      - In no-reasoner mode, we evaluate a limited safe subset.

    cost:
      - scalar costs to deduct from vars if/when applied (the simulation/RL env
        will decide how to apply; this module only *reports* the costs).
    """
    action_id: str
    name: str
    effect_dir: Tuple[float, ...]
    precond: Optional[str] = None
    cost: Dict[str, float] = field(default_factory=dict)
    tags: Tuple[str, ...] = ()

    @staticmethod
    def create(
        action_id: str,
        name: str,
        effect_dir: Sequence[float],
        precond: Optional[str] = None,
        cost: Optional[Dict[str, float]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> "Action":
        eff = normalize(list(effect_dir))
        return Action(
            action_id=str(action_id),
            name=str(name),
            effect_dir=tuple(float(x) for x in eff),
            precond=precond,
            cost=dict(cost or {}),
            tags=tuple(tags or ()),
        )


# -----------------------------
# Reasoner adapter (optional)
# -----------------------------

class Reasoner(Protocol):
    """Adapter interface so reasoner/no-reasoner modes can coexist.

    - If you have OWL/rdflib/owlready2, implement this interface.
    - If not, use NoReasoner (default).
    """
    def is_action_allowed(self, action: Action, ctx: AgentContext) -> bool: ...


class NoReasoner:
    """No-reasoner mode: evaluate precond via a small safe expression evaluator."""

    def __init__(self, evaluator: Optional[Callable[[Optional[str], AgentContext], bool]] = None):
        self._eval = evaluator or eval_precond_safe

    def is_action_allowed(self, action: Action, ctx: AgentContext) -> bool:
        return self._eval(action.precond, ctx)


# -----------------------------
# Safe precondition evaluation (no reasoner)
# -----------------------------

_ALLOWED_NAMES = {"and", "or", "not", "True", "False"}

def _ctx_value(ctx: AgentContext, dotted: str) -> Any:
    # Supports:
    # - "vars.energy"
    # - "vars.stamina"
    # - "turn"
    if dotted == "turn":
        return ctx.turn
    if dotted.startswith("vars."):
        key = dotted.split(".", 1)[1]
        return float(ctx.vars.get(key, 0.0))
    raise KeyError(dotted)


def eval_precond_safe(expr: Optional[str], ctx: AgentContext) -> bool:
    """Evaluate a limited boolean expression safely.

    Supported:
      - comparisons: <, <=, >, >=, ==, !=
      - boolean ops: and/or/not
      - literals: numbers, True/False
      - variables: vars.<name>, turn

    Examples:
      - "vars.energy >= 0.3 and vars.stamina >= 0.2"
      - "turn >= 3"
      - "not (vars.energy < 0.2)"
    """
    if expr is None or str(expr).strip() == "":
        return True

    import ast

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # constants
        if isinstance(node, ast.Constant):
            return node.value

        # names: True/False only
        if isinstance(node, ast.Name):
            if node.id in ("True", "False"):
                return node.id == "True"
            raise ValueError(f"Name not allowed: {node.id}")

        # vars.energy style: represented as Attribute(Attribute(Name('vars'), 'energy')) in AST
        if isinstance(node, ast.Attribute):
            # build dotted path
            parts: List[str] = []
            cur: ast.AST = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                dotted = ".".join(reversed(parts))
                return _ctx_value(ctx, dotted)
            raise ValueError("Unsupported attribute base")

        # boolean ops
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(bool(_eval(v)) for v in node.values)
            if isinstance(node.op, ast.Or):
                return any(bool(_eval(v)) for v in node.values)
            raise ValueError("Unsupported BoolOp")

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not bool(_eval(node.operand))

        # comparisons
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = _eval(comparator)
                ok: bool
                if isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                elif isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                else:
                    raise ValueError("Unsupported comparison operator")
                if not ok:
                    return False
                left = right
            return True

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    return bool(_eval(tree))


# -----------------------------
# Action catalog (ontology-ish store)
# -----------------------------

@dataclass
class ActionCatalog:
    """Stores actions and queries them by preferred direction.

    This is the "ontology-ish" part:
    - actions are entities
    - tags/preconditions are symbolic constraints
    - reasoner validates constraints (optional)
    """

    dim: int
    actions: List[Action] = field(default_factory=list)
    reasoner: Reasoner = field(default_factory=NoReasoner)

    def add(self, action: Action) -> None:
        if len(action.effect_dir) != self.dim:
            raise ValueError(f"action dim mismatch: expected {self.dim}, got {len(action.effect_dir)}")
        self.actions.append(action)

    def extend(self, actions: Iterable[Action]) -> None:
        for a in actions:
            self.add(a)

    def query_actions(
        self,
        preferred_dir: Sequence[float],
        ctx: AgentContext,
        top_k: int = 6,
        min_similarity: float = -1.0,
        require_allowed: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return ranked action candidates.

        preferred_dir:
          - direction vector representing "where the agent wants to go"
          - will be normalized before similarity

        min_similarity:
          - filter actions with cosine < threshold

        require_allowed:
          - if True, exclude actions failing preconditions
          - if False, return them but mark allowed=False
        """
        p = normalize(list(preferred_dir))
        scored: List[Tuple[float, Action, bool]] = []

        for a in self.actions:
            sim = cosine_similarity(p, a.effect_dir)
            if sim < float(min_similarity):
                continue
            allowed = bool(self.reasoner.is_action_allowed(a, ctx))
            if require_allowed and not allowed:
                continue
            scored.append((sim, a, allowed))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for sim, a, allowed in scored[: int(top_k)]:
            out.append(
                {
                    "action_id": a.action_id,
                    "name": a.name,
                    "similarity": float(sim),
                    "allowed": bool(allowed),
                    "precond": a.precond,
                    "cost": dict(a.cost),
                    "tags": list(a.tags),
                    "effect_dir": list(a.effect_dir),
                }
            )
        return out


# -----------------------------
# Example catalogs
# -----------------------------

def build_demo_catalog(dim: int, reasoner: Optional[Reasoner] = None) -> ActionCatalog:
    """A tiny starter set you can replace with OWL-backed definitions later."""
    cat = ActionCatalog(dim=dim, reasoner=reasoner or NoReasoner())

    cat.extend(
        [
            Action.create(
                "breath",
                "Breathing / Grounding",
                effect_dir=[-0.10, 0.25, -0.05, 0.10, 0.05, -0.05][:dim],
                precond="vars.energy >= 0.05",
                cost={"energy": 0.05},
                tags=["self_regulation"],
            ),
            Action.create(
                "reach_out",
                "Reach Out to Friend",
                effect_dir=[0.20, -0.05, 0.10, 0.05, 0.15, 0.00][:dim],
                precond="vars.energy >= 0.15 and vars.stamina >= 0.10",
                cost={"energy": 0.10, "stamina": 0.05},
                tags=["social"],
            ),
            Action.create(
                "avoid",
                "Avoid / Withdraw",
                effect_dir=[-0.05, -0.10, 0.15, 0.10, -0.20, 0.05][:dim],
                precond=None,
                cost={"energy": 0.02},
                tags=["avoidance"],
            ),
            Action.create(
                "vent",
                "Venting / Outburst",
                effect_dir=[0.25, 0.05, -0.15, 0.10, -0.05, 0.00][:dim],
                precond="vars.energy >= 0.30",
                cost={"energy": 0.20},
                tags=["impulsive"],
            ),
            Action.create(
                "plan",
                "Plan / Reframe",
                effect_dir=[0.05, 0.10, 0.10, -0.05, 0.15, 0.10][:dim],
                precond="vars.energy >= 0.20",
                cost={"energy": 0.10},
                tags=["cognitive"],
            ),
        ]
    )

    return cat


if __name__ == "__main__":
    # Quick smoke test
    dim = 6
    catalog = build_demo_catalog(dim)

    ctx = AgentContext(vars={"energy": 0.25, "stamina": 0.15}, turn=1)
    preferred = [0.10, 0.15, 0.05, 0.00, 0.20, 0.05]

    results = catalog.query_actions(preferred, ctx, top_k=5, min_similarity=-1.0)
    print("Preferred dir:", preferred)
    print("Top actions:")
    for r in results:
        print(f"- {r['action_id']:10s} sim={r['similarity']:.3f} allowed={r['allowed']} precond={r['precond']}")