"""Minimal reinforcement-learning training loop for the sim agents.

Goal
----
Train a single shared policy that selects an `action_id` each turn to move an
agent closer to its comfort center.

Notes
-----
- This is intentionally lightweight (NumPy only) and does NOT require sim_onto.
- It uses a simple REINFORCE (policy gradient) with a linear softmax policy.
- Action IDs must exist in the demo scenario data (sim_scenario.build_demo_scenario).

How it works
------------
1) Build a demo scenario (rounds with choices that each carry an action payload)
2) At each step, the policy chooses an `action_id` from a fixed action set.
3) The env picks a matching scenario choice for the current round (by action_id)
   and applies its action to the SimAgent via add_event(...).
4) Reward is improvement in distance-to-comfort (dense shaping) plus a small
   success bonus when entering comfort.

This is a starting point you can later upgrade to PPO/DQN and multi-agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
from pathlib import Path

import numpy as np

# Local imports (these modules exist in the same src/test package in this project)
from sim_agent import SimAgent

from sim_scenario import ScenarioManager, ScenarioSession, build_demo_scenario


# --- ScenarioSession round_id compatibility helper ---
def _session_round_id(session: ScenarioSession) -> int:
    """Best-effort access to ScenarioSession's current round id (int).

    Different iterations may store this as `round_id`, `current_round_id`, or inside a `state` object.
    """
    # Direct attributes
    for name in ("round_id", "current_round_id", "round", "current_round"):
        if hasattr(session, name):
            v = getattr(session, name)
            try:
                return int(v)
            except Exception:
                # handle strings like 'r1'
                if isinstance(v, str):
                    digits = "".join(ch for ch in v if ch.isdigit())
                    if digits:
                        return int(digits)

    # Nested state
    st = getattr(session, "state", None)
    if st is not None:
        for name in ("round_id", "current_round_id", "round"):
            if hasattr(st, name):
                v = getattr(st, name)
                try:
                    return int(v)
                except Exception:
                    if isinstance(v, str):
                        digits = "".join(ch for ch in v if ch.isdigit())
                        if digits:
                            return int(digits)

    # Fallback: try private fields
    for name in ("_round_id", "_current_round_id"):
        if hasattr(session, name):
            try:
                return int(getattr(session, name))
            except Exception:
                pass

    raise AttributeError("ScenarioSession has no accessible round id attribute")


# -----------------------------
# Utilities
# -----------------------------

def l2(a: List[float], b: List[float]) -> float:
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if s <= 0:
        return np.ones_like(x) / float(len(x))
    return ex / s


# -----------------------------
# Environment
# -----------------------------

# --- SimAgent compatibility helpers ---
import inspect

def _get_current_vec(agent: SimAgent) -> List[float]:
    """Best-effort access to the agent's current vector."""
    if hasattr(agent, "get_current_vec"):
        return list(agent.get_current_vec())  # type: ignore[misc]
    st = getattr(agent, "state", None)
    for name in ("current_vec", "current", "pos", "position"):
        if st is not None and hasattr(st, name):
            return list(getattr(st, name))
    for name in ("current_vec", "current", "pos", "position"):
        if hasattr(agent, name):
            return list(getattr(agent, name))
    raise AttributeError("Could not find current vector on SimAgent/state")


def _set_current_vec(agent: SimAgent, vec: List[float]) -> None:
    """Best-effort setter for the agent's current vector."""
    if hasattr(agent, "set_current_vec"):
        agent.set_current_vec(list(vec))  # type: ignore[misc]
        return
    st = getattr(agent, "state", None)
    for name in ("current_vec", "current", "pos", "position"):
        if st is not None and hasattr(st, name):
            try:
                setattr(st, name, list(vec))
                return
            except Exception:
                pass
    for name in ("current_vec", "current", "pos", "position"):
        if hasattr(agent, name):
            try:
                setattr(agent, name, list(vec))
                return
            except Exception:
                pass
    raise AttributeError("Could not set current vector on SimAgent/state")


def _agent_step(agent: SimAgent, turn: int) -> None:
    """Advance one simulation turn.

    Tries common method names; if none exist, applies the per-turn event delta directly
    using `event_delta_at_turn` when available.
    """
    t = int(turn)
    for name in ("step", "advance", "tick", "run_turn", "apply_turn", "update"):
        if hasattr(agent, name):
            fn = getattr(agent, name)
            try:
                fn(t)
                return
            except TypeError:
                # Some variants take no args
                try:
                    fn()
                    return
                except Exception:
                    pass
            except Exception:
                pass

    # Fallback: apply event delta if available
    if hasattr(agent, "event_delta_at_turn"):
        delta = list(agent.event_delta_at_turn(t))  # type: ignore[misc]
        cur = _get_current_vec(agent)
        new = [x + d for x, d in zip(cur, delta)]
        _set_current_vec(agent, new)
        return

    raise AttributeError("SimAgent has no supported step/advance method and no event_delta_at_turn")

def _get_comfort_vec(agent: SimAgent) -> List[float]:
    """Best-effort access to the agent's comfort center vector."""
    # Preferred: agent.state.*
    st = getattr(agent, "state", None)
    for name in ("comfort_vec", "comfort_center", "comfort", "center", "comfort_point"):
        if st is not None and hasattr(st, name):
            v = getattr(st, name)
            return list(v)
    # Fallback: agent.*
    for name in ("comfort_vec", "comfort_center", "comfort", "center", "comfort_point"):
        if hasattr(agent, name):
            v = getattr(agent, name)
            return list(v)
    raise AttributeError("Could not find comfort vector on SimAgent/state")


def _get_vars(agent: SimAgent) -> Dict[str, float]:
    st = getattr(agent, "state", None)
    if st is not None and hasattr(st, "vars"):
        try:
            return dict(getattr(st, "vars"))
        except Exception:
            pass
    if hasattr(agent, "vars"):
        try:
            return dict(getattr(agent, "vars"))
        except Exception:
            pass
    return {}


def _make_agent(dim: int, comfort: List[float], radius: float, init: List[float]) -> SimAgent:
    """Construct SimAgent with best-effort compatibility across constructor variants."""
    # Try common keyword variants first
    candidates = [
        dict(dim=dim, comfort_center=comfort, radius=radius, current_vec=init),
        dict(dim=dim, comfort_vec=comfort, comfort_radius=radius, current_vec=init),
        dict(dim=dim, comfort=comfort, radius=radius, current_vec=init),
        dict(dim=dim, comfort=comfort, comfort_radius=radius, current_vec=init),
    ]
    for kw in candidates:
        try:
            return SimAgent(**kw)  # type: ignore[arg-type]
        except TypeError:
            continue

    # If keyword variants fail, try constructing with an explicit AgentState if available.
    try:
        sig = inspect.signature(SimAgent)
        if "state" in sig.parameters:
            try:
                from sim_agent import AgentState  # type: ignore

                # Try common AgentState keyword names.
                st = None
                try:
                    st = AgentState(
                        comfort_vec=list(comfort),
                        comfort_radius=float(radius),
                        current_vec=list(init),
                        vars={"energy": 0.25, "stamina": 0.15},
                    )
                except TypeError:
                    # Alternate names
                    st = AgentState(
                        comfort_center=list(comfort),
                        radius=float(radius),
                        current_vec=list(init),
                        vars={"energy": 0.25, "stamina": 0.15},
                    )

                try:
                    return SimAgent(dim=dim, state=st)  # type: ignore[arg-type]
                except TypeError:
                    # Some variants may also accept current_vec separately
                    return SimAgent(dim=dim, state=st, current_vec=list(init))  # type: ignore[arg-type]
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: construct the agent with dim only, then patch fields on agent.state/agent.
    try:
        ag = SimAgent(dim=dim)  # type: ignore[misc]
    except TypeError:
        ag = SimAgent(dim)  # type: ignore[misc]
    # Try to attach comfort/current if possible
    st = getattr(ag, "state", None)
    if st is not None:
        for name in ("comfort_vec", "comfort_center", "comfort"):
            if hasattr(st, name):
                setattr(st, name, list(comfort))
                break
        for name in ("comfort_radius", "radius"):
            if hasattr(st, name):
                setattr(st, name, float(radius))
                break
        for name in ("current_vec", "current"):
            if hasattr(st, name):
                setattr(st, name, list(init))
                break
    return ag


@dataclass
class StepInfo:
    round_id: int
    choice_id: str
    action_id: str
    reward: float
    dist_before: float
    dist_after: float


class SimEnv:
    """A tiny turn-based environment around SimAgent + demo Scenario.

    - One episode = traverse scenario rounds until no more rounds.
    - Each step consumes 1 simulation turn.
    - Policy acts in action_id space (fixed, small).
    """

    def __init__(self, dim: int = 6):
        self.dim = int(dim)

        # Build scenario + manager
        built = build_demo_scenario(self.dim)

        # Some project variants return a Scenario, others return a ScenarioManager.
        if isinstance(built, ScenarioManager):
            self._manager = built
            # Best-effort: recover the underlying scenario object
            if hasattr(built, "scenario"):
                self._scenario = built.scenario  # type: ignore[assignment]
            elif hasattr(built, "_scenario"):
                self._scenario = built._scenario  # type: ignore[assignment]
            else:
                # Keep a reference; rank_choices will fail if scenario cannot be recovered
                self._scenario = built  # type: ignore[assignment]
        else:
            self._scenario = built
            self._manager = ScenarioManager(self._scenario)

        # Fixed action_id space (keep small / stable)
        self.action_ids: List[str] = [
            "breath",
            "reach_out",
            "plan",
            "vent",
            "avoid",
        ]

        self.agent: SimAgent | None = None
        self.session: ScenarioSession | None = None
        self.turn: int = 0
        self._episode_done: bool = False

        # Episode horizon (number of scenario decisions). Prevents getting stuck in a round.
        self._episode_step: int = 0
        self._episode_horizon: int = 3

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(int(seed))

        # Agent starts at a random point near comfort center (small noise)
        comfort = [-0.1, 0.2, -0.05, 0.25, 0.05, -0.05]
        radius = 0.35

        init = [c + float(np.random.uniform(-0.15, 0.15)) for c in comfort]
        self.agent = _make_agent(self.dim, comfort, radius, init)

        # Start scenario session (demo uses numeric round ids starting at 1)
        self.session = ScenarioSession(manager=self._manager, init_round_id=1, ctx=self._make_ctx())

        self.turn = 0
        self._episode_step = 0

        # Best-effort: determine number of rounds in the demo scenario.
        horizon = None
        sc = getattr(self._manager, "scenario", None)
        if sc is None:
            sc = getattr(self._manager, "_scenario", None)
        if sc is None:
            sc = getattr(self, "_scenario", None)

        # Common shapes: scenario.rounds (dict), scenario._rounds (dict), or list-like
        for name in ("rounds", "_rounds"):
            if sc is not None and hasattr(sc, name):
                try:
                    v = getattr(sc, name)
                    horizon = len(v)
                    break
                except Exception:
                    pass

        # Fallback to 3 (our demo scenario)
        self._episode_horizon = int(horizon) if horizon and int(horizon) > 0 else 3

        self._episode_done = False
        return self._observe()

    def _make_ctx(self):
        # SelectionContext signature may differ across iterations of this project.
        # We try a few common variants.
        from sim_scenario import SelectionContext

        assert self.agent is not None
        turn = int(self.turn)
        current_vec = list(_get_current_vec(self.agent))
        comfort_vec = list(_get_comfort_vec(self.agent))

        # Try common keyword forms
        for kw in (
            {"turn": turn, "current_vec": current_vec, "comfort_vec": comfort_vec, "tags": []},
            {"turn": turn, "current_vec": current_vec, "comfort_vec": comfort_vec},
            {"turn": turn, "current_vec": current_vec, "comfort_vec": comfort_vec, "memory_tags": []},
            {"turn": turn, "current_vec": current_vec, "comfort_vec": comfort_vec, "pref_tags": []},
        ):
            try:
                return SelectionContext(**kw)  # type: ignore[arg-type]
            except TypeError:
                continue

        # Try positional variants
        try:
            return SelectionContext(turn, current_vec, comfort_vec, [])  # type: ignore[misc]
        except TypeError:
            pass
        try:
            return SelectionContext(turn, current_vec, comfort_vec)  # type: ignore[misc]
        except TypeError:
            pass

        # Last resort: construct empty then patch attributes if it supports it
        ctx = SelectionContext()  # type: ignore[call-arg]
        for name, value in (
            ("turn", turn),
            ("current_vec", current_vec),
            ("comfort_vec", comfort_vec),
            ("tags", []),
            ("memory_tags", []),
            ("pref_tags", []),
        ):
            if hasattr(ctx, name):
                try:
                    setattr(ctx, name, value)
                except Exception:
                    pass
        return ctx

    def _observe(self) -> np.ndarray:
        assert self.agent is not None

        cur = _get_current_vec(self.agent)
        comfort = _get_comfort_vec(self.agent)
        delta = [c - x for x, c in zip(cur, comfort)]
        dist = l2(cur, comfort)

        # Optional agent vars (if missing, treat as 0)
        v = _get_vars(self.agent)
        energy = float(v.get("energy", 0.0))
        stamina = float(v.get("stamina", 0.0))

        obs = np.array(list(cur) + list(delta) + [dist, energy, stamina], dtype=np.float32)
        return obs

    @property
    def obs_dim(self) -> int:
        # cur(dim) + delta(dim) + dist + energy + stamina
        return self.dim + self.dim + 3

    @property
    def n_actions(self) -> int:
        return len(self.action_ids)

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, StepInfo]:
        if self._episode_done:
            raise RuntimeError("Episode is done; call reset().")

        assert self.agent is not None
        assert self.session is not None

        action_index = int(action_index)
        action_index = max(0, min(self.n_actions - 1, action_index))
        desired_action_id = self.action_ids[action_index]
        used_fallback = False

        # Update context for this turn
        self.session.ctx = self._make_ctx()

        # Get choices for current round from the scenario
        round_id = _session_round_id(self.session)
        ranked = self._manager.rank_choices(round_id, self.session.ctx)

        # Pick a choice that matches desired_action_id; fallback to first choice if none
        chosen = None
        for c in ranked:
            a = c.get("action") or {}
            if str(a.get("action_id", "")) == desired_action_id:
                chosen = c
                break
        if chosen is None:
            chosen = ranked[0]
            used_fallback = True

        choice_id = str(chosen.get("choice_id"))
        action = chosen.get("action") or {}
        action_id = str(action.get("action_id", ""))
        if used_fallback:
            # Helpful for debugging: the policy asked for desired_action_id, but this round didn't have it.
            action_id = f"{action_id} (fallback_for={desired_action_id})"

        # Apply action as an event on this turn
        duration = int(action.get("duration", 1))
        magnitude = float(action.get("magnitude", 0.25))
        direction = list(action.get("embed_vec") or [0.0] * self.dim)

        # Compute reward shaping: distance improvement
        dist_before = l2(_get_current_vec(self.agent), _get_comfort_vec(self.agent))

        self.agent.add_event(
            start_turn=self.turn,
            duration=duration,
            direction=direction,
            magnitude=magnitude,
        )

        # Advance one turn of dynamics
        _agent_step(self.agent, self.turn)
        self.turn += 1

        dist_after = l2(_get_current_vec(self.agent), _get_comfort_vec(self.agent))

        # Dense shaping: get closer => positive
        reward = float(dist_before - dist_after)

        # Success bonus if enter comfort
        # PATCH: guard against missing is_in_comfort method
        if hasattr(self.agent, "is_in_comfort"):
            in_after = bool(self.agent.is_in_comfort())  # type: ignore[misc]
        else:
            # Fallback: consider inside comfort if distance is within radius when available
            in_after = False
            st = getattr(self.agent, "state", None)
            radius = None
            if st is not None and hasattr(st, "comfort_radius"):
                radius = float(getattr(st, "comfort_radius"))
            elif st is not None and hasattr(st, "radius"):
                radius = float(getattr(st, "radius"))
            if radius is not None:
                in_after = dist_after <= radius
        if in_after:
            reward += 0.05

        # Advance scenario session (may vary by implementation)
        _ = self.session.step(choice_id)

        # One decision per call to env.step
        self._episode_step += 1

        # Terminate after the scenario horizon to avoid getting stuck in a round.
        if self._episode_step >= self._episode_horizon:
            self._episode_done = True

        obs = self._observe()
        info = StepInfo(
            round_id=round_id,
            choice_id=choice_id,
            action_id=action_id,
            reward=reward,
            dist_before=dist_before,
            dist_after=dist_after,
        )
        return obs, reward, self._episode_done, info


# -----------------------------
# Policy (REINFORCE)
# -----------------------------


def save_policy_npz(policy: "LinearSoftmaxPolicy", path: str | Path) -> str:
    """Save policy parameters to a NumPy .npz file.

    Returns the absolute path of the saved file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(p), W=policy.W, b=policy.b)
    return str(p.resolve())


def load_policy_npz(path: str | Path, obs_dim: int, n_actions: int) -> "LinearSoftmaxPolicy":
    """Load policy parameters from a NumPy .npz file."""
    p = Path(path)
    data = np.load(str(p))
    policy = LinearSoftmaxPolicy(obs_dim=obs_dim, n_actions=n_actions, seed=0)
    policy.W = data["W"].astype(np.float32)
    policy.b = data["b"].astype(np.float32)
    return policy


class LinearSoftmaxPolicy:
    """A tiny linear policy: logits = W @ obs + b."""

    def __init__(self, obs_dim: int, n_actions: int, seed: int = 0):
        rng = np.random.default_rng(int(seed))
        self.W = rng.normal(0.0, 0.05, size=(n_actions, obs_dim)).astype(np.float32)
        self.b = np.zeros((n_actions,), dtype=np.float32)

    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        logits = (self.W @ obs) + self.b
        return softmax(logits.astype(np.float64)).astype(np.float32)

    def sample_action(self, obs: np.ndarray) -> Tuple[int, float]:
        p = self.action_probs(obs)
        a = int(np.random.choice(len(p), p=p))
        logp = float(np.log(max(1e-12, p[a])))
        return a, logp

    def update(self, grads_W: np.ndarray, grads_b: np.ndarray, lr: float) -> None:
        self.W += lr * grads_W
        self.b += lr * grads_b


def train_reinforce(
    episodes: int = 500,
    gamma: float = 0.98,
    lr: float = 0.02,
    seed: int = 0,
    save_path: str | Path | None = "data/models/policy_reinforce.npz",
) -> LinearSoftmaxPolicy:
    env = SimEnv(dim=6)
    policy = LinearSoftmaxPolicy(env.obs_dim, env.n_actions, seed=seed)

    np.random.seed(int(seed))

    for ep in range(1, int(episodes) + 1):
        obs = env.reset(seed=seed + ep)

        # Debug: cap steps to avoid infinite loops if ScenarioSession never ends
        max_steps = 200

        traj_obs: List[np.ndarray] = []
        traj_act: List[int] = []
        traj_logp: List[float] = []
        traj_rew: List[float] = []
        traj_info: List[StepInfo] = []

        done = False
        step_i = 0
        while not done:
            a, logp = policy.sample_action(obs)
            next_obs, r, done, info = env.step(a)

            step_i += 1

            # Episodes are short (3 steps). Log each step so round progression is visible.
            if step_i <= 10 or (step_i % 25 == 0):
                print(
                    f"[EP {ep:4d} | STEP {step_i:3d}] a={env.action_ids[a]} r={float(r):+.4f} "
                    f"round={info.round_id} choice={info.choice_id} action_id={info.action_id} "
                    f"dist {info.dist_before:.4f}->{info.dist_after:.4f} done={done}"
                )

            # Safety: break if episode runs too long (likely scenario not terminating)
            if step_i >= max_steps:
                print(
                    f"[EP {ep:4d}] WARNING: hit max_steps={max_steps}; forcing done=True "
                    f"(last round={info.round_id}, choice={info.choice_id})"
                )
                done = True

            # NaN/inf guard
            if not np.isfinite(next_obs).all():
                print(f"[EP {ep:4d}] ERROR: non-finite obs detected; forcing done=True")
                done = True

            traj_obs.append(obs)
            traj_act.append(a)
            traj_logp.append(logp)
            traj_rew.append(float(r))
            traj_info.append(info)

            obs = next_obs

        # Compute returns
        G = 0.0
        returns: List[float] = []
        for r in reversed(traj_rew):
            G = float(r) + float(gamma) * G
            returns.append(G)
        returns.reverse()

        # Normalize returns for stability
        ret = np.array(returns, dtype=np.float32)
        ret = (ret - ret.mean()) / (ret.std() + 1e-6)

        # Policy gradient accumulation
        grads_W = np.zeros_like(policy.W)
        grads_b = np.zeros_like(policy.b)

        for t, (o, a, R) in enumerate(zip(traj_obs, traj_act, ret)):
            p = policy.action_probs(o)
            # grad log pi(a|o) for softmax-linear:
            # dlogpi/dW = (onehot(a) - p)[:,None] * o[None,:]
            one = np.zeros_like(p)
            one[a] = 1.0
            diff = (one - p).astype(np.float32)  # shape (A,)
            grads_W += (R * diff[:, None] * o[None, :]).astype(np.float32)
            grads_b += (R * diff).astype(np.float32)

        policy.update(grads_W, grads_b, lr=float(lr))

        # Episode summary
        ep_return = float(sum(traj_rew))
        last = traj_info[-1]
        if ep % 25 == 0 or ep == 1:
            print(
                f"[EP {ep:4d}] return={ep_return:+.4f} steps={len(traj_rew)} "
                f"last_dist={last.dist_after:.4f} last_action={env.action_ids[traj_act[-1]]} "
                f"(choice={last.choice_id}, action_id={last.action_id})"
            )

    # Save trained policy (optional)
    if save_path is not None:
        saved_to = save_policy_npz(policy, save_path)
        print(f"[SAVE] policy -> {saved_to}")

    return policy


if __name__ == "__main__":
    # Minimal demo run
    _ = train_reinforce(episodes=300, gamma=0.98, lr=0.02, seed=0)