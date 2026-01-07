"""Simple emotion-vector simulation with round-based action choices.

Goal
----
- An event moves the agent's emotion vector.
- The agent tries to move toward a comfort point (vector) by choosing from 3 actions per round.
- Ontology-like constraints decide which actions are feasible (allowed/blocked/cooldown/cost).
- Vector similarity to the comfort direction ranks feasible actions.

This file is self-contained (no LLM required) so you can test the full flow quickly.

Run:
  python sim_test.py

Optional:
  python sim_test.py --seed 7 --rounds 2 --event breakup
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Vector utilities
# -----------------------------

def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: List[float]) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: List[float], eps: float = 1e-12) -> List[float]:
    n = norm(a)
    if n < eps:
        return [0.0 for _ in a]
    return [x / n for x in a]


def add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def mul(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]


def cosine(a: List[float], b: List[float], eps: float = 1e-12) -> float:
    na = norm(a)
    nb = norm(b)
    if na < eps or nb < eps:
        return 0.0
    return dot(a, b) / (na * nb)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# -----------------------------
# Ontology-like rules (minimal)
# -----------------------------

@dataclass
class Action:
    name: str
    # Prototype movement direction (delta). Interpreted in emotion-space.
    delta: List[float]
    # Ontology-like meta
    allowed_when: Optional[str] = None
    blocked_by: Optional[str] = None
    cost: Dict[str, float] = field(default_factory=dict)
    cooldown_s: float = 0.0
    base_speed: float = 1.0


# A tiny safe evaluator for expressions like: "energy > 0.3 and stamina > 0.2"
# Supports variables from env, numbers, comparisons, and/or/not, parentheses.
import ast
import operator


def eval_condition(expr: Optional[str], env: Dict[str, float]) -> bool:
    if expr is None:
        return True

    OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    CMPS = {
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise ValueError("Unsupported constant")

        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise ValueError(f"Unknown variable: {node.id}")

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(bool(_eval(v)) for v in node.values)
            if isinstance(node.op, ast.Or):
                return any(bool(_eval(v)) for v in node.values)
            raise ValueError("Unsupported boolean op")

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not bool(_eval(node.operand))
            op_fn = OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError("Unsupported unary op")
            return op_fn(float(_eval(node.operand)))

        if isinstance(node, ast.BinOp):
            op_fn = OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError("Unsupported binary op")
            return op_fn(float(_eval(node.left)), float(_eval(node.right)))

        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = _eval(comp)
                cmp_fn = CMPS.get(type(op))
                if cmp_fn is None:
                    raise ValueError("Unsupported comparator")
                if not cmp_fn(float(left), float(right)):
                    return False
                left = right
            return True

        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    try:
        tree = ast.parse(expr, mode="eval")
        return bool(_eval(tree))
    except Exception:
        return False


@dataclass
class OntologyEngine:
    """Provides feasible actions given round + agent state, enforcing rules."""

    actions: Dict[str, Action]

    def feasible_actions(
        self,
        round_actions: List[str],
        agent_state: Dict[str, float],
        cooldowns: Dict[str, float],
        now_s: float,
    ) -> Tuple[List[Action], Dict[str, str]]:
        """Return feasible Action objects + debug reason per action name."""
        env = dict(agent_state)
        feasible: List[Action] = []
        reasons: Dict[str, str] = {}

        for name in round_actions:
            a = self.actions[name]

            allowed = eval_condition(a.allowed_when, env)
            blocked = (a.blocked_by is not None) and eval_condition(a.blocked_by, env)

            last = cooldowns.get(a.name, -1e9)
            remaining = max(0.0, (last + a.cooldown_s) - now_s)
            cooldown_ok = remaining <= 0.0

            if not allowed:
                reasons[name] = f"blocked: allowedWhen={a.allowed_when}"
                continue
            if blocked:
                reasons[name] = f"blocked: blockedBy={a.blocked_by}"
                continue
            if not cooldown_ok:
                reasons[name] = f"blocked: cooldown remaining {remaining:.1f}s"
                continue

            feasible.append(a)
            reasons[name] = "ok"

        return feasible, reasons


# -----------------------------
# Agent and simulation
# -----------------------------

@dataclass
class EmotionAgent:
    dim: int
    current_vec: List[float]
    comfort_vec: List[float]
    # runtime state (0..1)
    state: Dict[str, float] = field(
        default_factory=lambda: {
            "energy": 0.6,
            "stamina": 0.6,
            "self_control": 0.5,
            "threat": 0.3,
            "overwhelm": 0.2,
        }
    )
    cooldowns: Dict[str, float] = field(default_factory=dict)

    def to_comfort_dir(self) -> List[float]:
        return normalize(sub(self.comfort_vec, self.current_vec))

    def dist_to_comfort(self) -> float:
        # cosine distance-ish: 1 - cos (in [-1,1] -> [0,2])
        return 1.0 - cosine(self.current_vec, self.comfort_vec)

    def apply_cost(self, cost: Dict[str, float]) -> None:
        for k, dv in (cost or {}).items():
            if k in self.state:
                self.state[k] = clamp01(float(self.state[k]) + float(dv))

    def apply_action(self, action: Action, now_s: float, alpha: float) -> None:
        # Move toward action delta direction. alpha controls step size.
        step = mul(normalize(action.delta), alpha)
        self.current_vec = normalize(add(self.current_vec, step))

        self.apply_cost(action.cost)
        if action.cooldown_s > 0:
            self.cooldowns[action.name] = now_s


def event_delta(event_name: str, dim: int) -> List[float]:
    """A deterministic-ish event delta just for testing."""
    base = [0.0] * dim

    # Hand-crafted tiny examples (replace with embedding-derived delta later)
    presets = {
        "breakup": [0.25, -0.10, 0.30, 0.05, -0.20, 0.10],
        "praise": [-0.15, 0.25, -0.10, 0.30, 0.05, -0.05],
        "threat": [0.30, 0.05, 0.10, -0.20, 0.25, 0.15],
        "loss": [0.20, -0.05, 0.35, 0.10, -0.15, 0.05],
    }

    v = presets.get(event_name.lower())
    if v is None:
        rnd = random.Random(event_name)
        v = [(rnd.random() - 0.5) * 0.5 for _ in range(dim)]

    for i in range(min(dim, len(v))):
        base[i] = v[i]
    return base


def rank_actions_by_comfort(agent: EmotionAgent, actions: List[Action]) -> List[Tuple[float, Action]]:
    """Rank by alignment between action delta and comfort direction."""
    to_comfort = agent.to_comfort_dir()

    ranked: List[Tuple[float, Action]] = []
    for a in actions:
        align = cosine(normalize(a.delta), to_comfort)
        cost_mag = sum(abs(float(v)) for v in (a.cost or {}).values())
        score = align - 0.25 * cost_mag
        ranked.append((score, a))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


def choose_action(agent: EmotionAgent, ranked: List[Tuple[float, Action]]) -> Action:
    """Greedy chooser for MVP. Replace with RL policy later."""
    return ranked[0][1]


def simulate(
    agent: EmotionAgent,
    ontology: OntologyEngine,
    rounds: int,
    round_action_sets: Dict[int, List[str]],
    event_name: str,
    seed: int,
) -> None:
    random.seed(seed)
    now_s = time.time()

    print("=" * 72)
    print(f"Event: {event_name}")
    print(
        f"Start state: vec={['%.3f' % x for x in agent.current_vec]} | dist_to_comfort={agent.dist_to_comfort():.3f}"
    )
    print(f"Comfort vec: {['%.3f' % x for x in agent.comfort_vec]}")
    print(f"Agent vars: {agent.state}")

    # 1) Apply event disturbance first
    ev = event_delta(event_name, agent.dim)
    agent.current_vec = normalize(add(agent.current_vec, ev))

    # Simple event effect on threat/overwhelm for visibility
    if event_name.lower() in ("threat", "loss", "breakup"):
        agent.state["threat"] = clamp01(agent.state["threat"] + 0.2)
        agent.state["overwhelm"] = clamp01(agent.state["overwhelm"] + 0.1)
    else:
        agent.state["energy"] = clamp01(agent.state["energy"] + 0.1)

    print("\nAfter event disturbance:")
    print(
        f"  vec={['%.3f' % x for x in agent.current_vec]} | dist_to_comfort={agent.dist_to_comfort():.3f}"
    )
    print(f"  to_comfort_dir={['%.3f' % x for x in agent.to_comfort_dir()]}")
    print(f"  vars={agent.state}")

    # 2) Round decisions
    for r in range(1, rounds + 1):
        print("\n" + ("-" * 72))
        print(f"Round {r}")

        round_actions = round_action_sets.get(r)
        if not round_actions:
            raise ValueError(f"No action set defined for round {r}")

        feasible, reasons = ontology.feasible_actions(
            round_actions=round_actions,
            agent_state=agent.state,
            cooldowns=agent.cooldowns,
            now_s=now_s,
        )

        print("Candidates (3):")
        for name in round_actions:
            print(f"  - {name:12s} : {reasons.get(name, 'n/a')}")

        if not feasible:
            print("No feasible actions. Agent stays still.")
            continue

        ranked = rank_actions_by_comfort(agent, feasible)

        print("Feasible ranked:")
        for score, a in ranked:
            print(
                f"  * {a.name:12s} score={score:+.3f} align={cosine(normalize(a.delta), agent.to_comfort_dir()):+.3f} "
                f"cost={a.cost} allowedWhen={a.allowed_when} blockedBy={a.blocked_by}"
            )

        chosen = choose_action(agent, ranked)

        # Step size alpha: depends on stamina/energy and action speed
        confidence = clamp01((ranked[0][0] + 1.0) / 2.0)
        alpha = (
            0.35
            * (0.4 + 0.6 * agent.state.get("stamina", 0.5))
            * (0.5 + 0.5 * confidence)
            * chosen.base_speed
        )
        alpha = max(0.05, min(0.6, alpha))

        prev_dist = agent.dist_to_comfort()
        agent.apply_action(chosen, now_s=now_s, alpha=alpha)
        now_dist = agent.dist_to_comfort()

        print(f"\nChosen: {chosen.name} | alpha={alpha:.3f}")
        print(f"  dist_to_comfort: {prev_dist:.3f} -> {now_dist:.3f} (delta {prev_dist - now_dist:+.3f})")
        print(f"  vec={['%.3f' % x for x in agent.current_vec]}")
        print(f"  vars={agent.state}")

    print("\nDone.")
    print("=" * 72)


# -----------------------------
# Demo setup
# -----------------------------

def build_demo_actions(dim: int) -> Dict[str, Action]:
    """Define a small action library (ontology action catalog)."""

    return {
        # Round 1
        "Avoid": Action(
            name="Avoid",
            delta=normalize([0.20, 0.05, -0.15, -0.05, 0.10, -0.10][:dim]),
            allowed_when="threat > 0.2",
            blocked_by=None,
            cost={"stamina": -0.08, "energy": -0.05},
            cooldown_s=5,
            base_speed=1.1,
        ),
        "Confront": Action(
            name="Confront",
            delta=normalize([-0.10, 0.20, -0.05, 0.15, -0.10, 0.05][:dim]),
            allowed_when="energy > 0.35 and self_control > 0.3",
            blocked_by="overwhelm > 0.8",
            cost={"stamina": -0.12, "energy": -0.10},
            cooldown_s=8,
            base_speed=1.0,
        ),
        "Reframe": Action(
            name="Reframe",
            delta=normalize([-0.05, 0.05, -0.10, 0.20, 0.05, -0.05][:dim]),
            allowed_when="self_control > 0.25",
            blocked_by=None,
            cost={"stamina": -0.06},
            cooldown_s=4,
            base_speed=0.9,
        ),

        # Round 2
        "Withdraw": Action(
            name="Withdraw",
            delta=normalize([0.15, -0.05, 0.10, -0.10, 0.05, 0.10][:dim]),
            allowed_when=None,
            blocked_by=None,
            cost={"stamina": -0.05},
            cooldown_s=3,
            base_speed=0.9,
        ),
        "Express": Action(
            name="Express",
            delta=normalize([-0.05, 0.15, -0.05, 0.10, -0.05, 0.05][:dim]),
            allowed_when="energy > 0.25",
            blocked_by="threat > 0.85",
            cost={"stamina": -0.10, "self_control": -0.05},
            cooldown_s=7,
            base_speed=1.0,
        ),
        "Suppress": Action(
            name="Suppress",
            delta=normalize([0.10, -0.10, 0.05, -0.05, 0.15, -0.10][:dim]),
            allowed_when="self_control > 0.4",
            blocked_by=None,
            cost={"energy": -0.08, "stamina": -0.04},
            cooldown_s=6,
            base_speed=0.8,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--event",
        type=str,
        default="breakup",
        choices=["breakup", "praise", "threat", "loss"],
    )
    args = parser.parse_args()

    dim = 6

    # Initial emotion vector and comfort vector (unit vectors)
    current = normalize([0.10, 0.05, 0.15, 0.05, 0.10, 0.05])
    comfort = normalize([-0.10, 0.20, -0.05, 0.25, 0.05, -0.05])

    agent = EmotionAgent(dim=dim, current_vec=current, comfort_vec=comfort)

    actions = build_demo_actions(dim)
    ontology = OntologyEngine(actions=actions)

    # Round action sets (3 choices each)
    round_action_sets = {
        1: ["Avoid", "Confront", "Reframe"],
        2: ["Withdraw", "Express", "Suppress"],
    }

    simulate(
        agent=agent,
        ontology=ontology,
        rounds=args.rounds,
        round_action_sets=round_action_sets,
        event_name=args.event,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
