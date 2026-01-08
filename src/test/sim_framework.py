"""Turn-based simulation runner built on top of `SimAgent`.

Design goals
------------
- Use `SimAgent` as a pure state container:
  - comfort zone (comfort_vec + comfort_radius)
  - scalar vars
  - event history (multi-turn events)
- The simulation runner owns the *live* emotion position (`current_vec`).
- Each turn:
  1) Aggregate active event deltas for this turn.
  2) Apply the delta to `current_vec` (simple additive dynamics).
  3) Optionally spawn random events when the agent is "in comfort".

This module is intentionally simple (MVP). You can later plug:
- ontology/reasoner for action feasibility
- RL policy for action selection
- more realistic dynamics (friction, decay, non-linearities)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Sequence

from sim_agent import SimAgent, AgentState, add, scale, normalize

# Scenario (tag-based) selection layer
from src.test.scenario.builder import build_demo_scenario
from src.test.scenario.manager import ScenarioManager
from src.test.scenario.dto.selection_ctx import SelectionContext
from sim_scenario import ScenarioSession


# -----------------------------
# Distance metrics
# -----------------------------

def l2_distance(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


# -----------------------------
# Random event generator (hook)
# -----------------------------

RandomEventFn = Callable[[int, int], Tuple[List[float], float]]
"""(dim, turn) -> (direction, magnitude). Duration is handled by caller."""


SharedActionFn = Callable[[int, str, SimAgent], Dict[str, object]]
"""(round_idx, chooser_id, chooser_agent) -> scenario selection result dict.

Expected keys (minimum):
- choice_id: str
- display_text: str
- action: dict with keys: action_id, embed_text, duration, magnitude, embed_vec
"""


def default_random_event(dim: int, turn: int) -> Tuple[List[float], float]:
    """A tiny deterministic-ish random event generator for testing.

    - Produces a direction vector based on turn index.
    - Produces a small magnitude.

    Replace this later with:
    - embedding(event_text) -> direction
    - sampled scenario -> direction
    """
    # Simple pseudo-random but reproducible pattern
    vals = []
    for i in range(dim):
        x = ((turn * 31 + i * 17) % 100) / 100.0
        vals.append((x - 0.5) * 2.0)  # [-1, 1]
    direction = normalize(vals)
    magnitude = 0.15
    return direction, magnitude


# -----------------------------
# Simulation runner
# -----------------------------

@dataclass
class TurnResult:
    turn: int
    current_vec_before: List[float]
    current_vec_after: List[float]
    event_delta: List[float]
    in_comfort_before: bool
    in_comfort_after: bool
    spawned_event: Optional[Dict[str, object]] = None


@dataclass
class TurnSimulation:
    """Runs a turn loop for one agent.

    Notes:
    - `SimAgent` stores comfort-zone + vars + events.
    - This class stores live emotion position: `current_vec`.
    - Comfort queries are delegated to SimAgent (distance/in-comfort/direction).
    """

    agent: SimAgent

    # dynamics
    distance_fn: Callable[[List[float], List[float]], float] = l2_distance
    apply_event_scale: float = 1.0

    # comfort-driven random events
    spawn_when_in_comfort: bool = False
    spawn_probability: float = 0.25
    spawn_duration: int = 1
    random_event_fn: RandomEventFn = default_random_event

    # internal
    turn: int = 0
    history: List[TurnResult] = field(default_factory=list)

    def __post_init__(self):
        if not (0.0 <= float(self.spawn_probability) <= 1.0):
            raise ValueError("spawn_probability must be in [0,1]")
        if int(self.spawn_duration) < 1:
            raise ValueError("spawn_duration must be >= 1")


    # ---------
    # Turn step
    # ---------

    def step(self) -> TurnResult:
        """Advance one turn.

        Turn pipeline:
        1) turn += 1
        2) aggregate active event delta for this turn
        3) apply delta to current position
        4) optionally spawn a random event if in comfort
        """
        self.turn += 1
        t = self.turn

        before = self.agent.get_current_vec()
        in_before = self.agent.is_in_comfort(distance_fn=self.distance_fn)

        spawned: Optional[Dict[str, object]] = None

        # 1) optionally spawn an event when already calm/comfortable
        if self.spawn_when_in_comfort and in_before:
            # If no event is currently active, always spawn exactly one event
            if len(self.agent.events_active_at_turn(t)) == 0:
                direction, magnitude = self.random_event_fn(self.agent.dim, t)
                self.agent.add_event(
                    start_turn=t,
                    duration=int(self.spawn_duration),
                    direction=direction,
                    magnitude=float(magnitude),
                )
                spawned = {
                    "start_turn": t,
                    "duration": int(self.spawn_duration),
                    "direction": list(direction),
                    "magnitude": float(magnitude),
                }

        # 2) aggregate event delta (force) for this turn
        delta = self.agent.event_delta_at_turn(t)
        if self.apply_event_scale != 1.0:
            delta = scale(delta, float(self.apply_event_scale))

        # 3) apply to live position (simple additive dynamics)
        after = add(before, delta)
        self.agent.set_current_vec(after)

        in_after = self.agent.is_in_comfort(distance_fn=self.distance_fn)

        result = TurnResult(
            turn=t,
            current_vec_before=before,
            current_vec_after=after,
            event_delta=list(delta),
            in_comfort_before=in_before,
            in_comfort_after=in_after,
            spawned_event=spawned,
        )
        self.history.append(result)
        return result

    def run(self, turns: int) -> List[TurnResult]:
        n = int(turns)
        if n < 0:
            raise ValueError("turns must be >= 0")
        out: List[TurnResult] = []
        for _ in range(n):
            out.append(self.step())
        return out


# -----------------------------
# Dual-agent alternating runner
# -----------------------------

@dataclass
class DualTurnResult:
    round: int
    chooser: str
    chosen_action: Dict[str, object]
    a: TurnResult
    b: TurnResult


@dataclass
class DualTurnSimulation:
    """Runs a shared round loop for two agents.

    - Each round, one agent is the *chooser* (alternates A/B).
    - The chooser selects an action vector.
    - That action triggers the SAME event on both agents.
    - Then both agents advance one turn.

    Notes:
    - Each agent keeps its own turn counter (their turns progress in lockstep here).
    - This class does not do any reasoner/RL; it just wires the mechanics.
    """

    sim_a: TurnSimulation
    sim_b: TurnSimulation
    scenario: ScenarioSession
    shared_action_fn: SharedActionFn

    round: int = 0
    history: List[DualTurnResult] = field(default_factory=list)

    def _apply_shared_event(self, direction: Sequence[float], magnitude: float, duration: int) -> None:
        """Add the same event to both agents, starting on their next turn."""
        # Start on next turn for each sim (since step() increments first).
        start_a = self.sim_a.turn + 1
        start_b = self.sim_b.turn + 1
        dur = int(duration)
        if dur < 1:
            raise ValueError("duration must be >= 1")

        self.sim_a.agent.add_event(
            start_turn=start_a,
            duration=dur,
            direction=list(direction),
            magnitude=float(magnitude),
        )
        self.sim_b.agent.add_event(
            start_turn=start_b,
            duration=dur,
            direction=list(direction),
            magnitude=float(magnitude),
        )

    def step(self) -> DualTurnResult:
        """Advance one shared round (both agents step once)."""
        self.round += 1
        r = self.round

        chooser = "A" if (r % 2 == 1) else "B"
        chooser_agent = self.sim_a.agent if chooser == "A" else self.sim_b.agent
        sel = self.shared_action_fn(r, chooser, chooser_agent)

        action = dict(sel.get("action") or {})
        if not action:
            raise ValueError("Scenario selection did not provide an 'action' payload")

        direction = action.get("embed_vec")
        magnitude = action.get("magnitude")
        duration = action.get("duration")

        if direction is None or magnitude is None or duration is None:
            raise ValueError("Action payload must include embed_vec, magnitude, duration")

        # Apply to both agents
        self._apply_shared_event(direction=direction, magnitude=float(magnitude), duration=int(duration))

        chosen_action = {
            "choice_id": str(sel.get("choice_id", "")),
            "display_text": str(sel.get("display_text", "")),
            "onto_action_id": sel.get("onto_action_id", ""),
            "onto_score": sel.get("onto_score", 0),
            "action": {
                "action_id": str(action.get("action_id", "")),
                "embed_text": str(action.get("embed_text", "")),
                "duration": int(duration),
                "magnitude": float(magnitude),
                "embed_vec": list(action.get("embed_vec") or []),
            },
            # convenience
            "direction_unit": list(normalize(list(direction))),
        }

        # Step both sims
        a_res = self.sim_a.step()
        b_res = self.sim_b.step()

        out = DualTurnResult(
            round=r,
            chooser=chooser,
            chosen_action=chosen_action,
            a=a_res,
            b=b_res,
        )
        self.history.append(out)
        return out

    def run(self, rounds: int) -> List[DualTurnResult]:
        n = int(rounds)
        if n < 0:
            raise ValueError("rounds must be >= 0")
        out: List[DualTurnResult] = []
        for _ in range(n):
            out.append(self.step())
        return out


# -----------------------------
# Minimal demo
# -----------------------------

if __name__ == "__main__":
    # Build two agents with identical comfort zones but independent states
    agent_a = SimAgent(
        dim=6,
        state=AgentState(
            comfort_vec=[-0.10, 0.20, -0.05, 0.25, 0.05, -0.05],
            comfort_radius=0.35,
            current_vec=[-0.10, 0.20, -0.05, 0.25, 0.05, -0.05],
            vars={"stamina": 0.6, "energy": 0.6, "tags": ["comforting", "grounding"]},
        ),
    )

    agent_b = SimAgent(
        dim=6,
        state=AgentState(
            comfort_vec=[-0.10, 0.20, -0.05, 0.25, 0.05, -0.05],
            comfort_radius=0.35,
            current_vec=[0.15, -0.05, 0.10, 0.05, -0.10, 0.20],
            vars={"stamina": 0.6, "energy": 0.6, "tags": ["vent", "direct"]},
        ),
    )

    sim_a = TurnSimulation(agent=agent_a, spawn_when_in_comfort=False)
    sim_b = TurnSimulation(agent=agent_b, spawn_when_in_comfort=False)

    def _log(msg: str) -> None:
        print(msg)

    def _fmt_tags(tags: List[str]) -> str:
        if not tags:
            return "[]"
        return "[" + ", ".join(map(str, tags)) + "]"

    # Build demo scenario (3 rounds, tag-based choices, each with an `action` payload)
    manager: ScenarioManager = build_demo_scenario(dim=agent_a.dim)

    # Ensure each scenario choice action includes an action_id (demo default).
    try:
        sc = getattr(manager, "scenario", None)
        rounds_obj = getattr(sc, "rounds", None)
        if isinstance(rounds_obj, dict):
            for _rid, rnd in rounds_obj.items():
                choices = rnd.get("choices", []) if isinstance(rnd, dict) else []
                for ch in choices:
                    if isinstance(ch, dict):
                        act = ch.get("action")
                        if isinstance(act, dict) and "action_id" not in act:
                            act["action_id"] = str(ch.get("choice_id", ""))
    except Exception:
        pass
    # ScenarioSession requires an initial round id and an initial selection context
    def _make_ctx(turn: int, current_vec: List[float], comfort_vec: List[float], tags: List[str]) -> SelectionContext:
        """Create SelectionContext with required core args.

        Your SelectionContext requires (turn, current_vec, comfort_vec). Tag preferences are attached if supported.
        """
        t = int(turn)
        cv = list(current_vec)
        cz = list(comfort_vec)

        # Try common keyword names for the required fields
        kw_variants = [
            {"turn": t, "current_vec": cv, "comfort_vec": cz},
            {"t": t, "current": cv, "comfort": cz},
        ]

        # First try: required args as keywords
        for base in kw_variants:
            try:
                ctx = SelectionContext(**base)  # type: ignore[arg-type]
                break
            except TypeError:
                ctx = None  # type: ignore[assignment]
        else:
            ctx = None  # type: ignore[assignment]

        # Second try: required args as positional
        if ctx is None:
            ctx = SelectionContext(t, cv, cz)  # type: ignore[call-arg]

        # Attach tags if the context supports it (best-effort)
        for attr in ("memory_tags", "tags", "preferred_tags", "pref_tags", "context_tags"):
            if hasattr(ctx, attr):
                try:
                    setattr(ctx, attr, list(tags))
                    break
                except Exception:
                    pass

        return ctx

    scenario = ScenarioSession(
        manager=manager,
        init_round_id=1,
        ctx=_make_ctx(
            turn=0,
            current_vec=agent_a.get_current_vec(),
            comfort_vec=agent_a.state.comfort_vec,
            tags=[],
        ),
    )

    def _preferred_dir_to_comfort(agent: SimAgent) -> List[float]:
        """Unit direction that would move the agent toward its comfort center."""
        cur = agent.get_current_vec()
        comfort = agent.state.comfort_vec
        raw = [c - x for x, c in zip(cur, comfort)]
        if all(abs(v) < 1e-12 for v in raw):
            return [0.0] * agent.dim
        return normalize(raw)


    def _get_onto_catalog(dim: int):
        """Build a demo ontology action catalog (best-effort)."""
        try:
            import sim_onto  # type: ignore
        except Exception as e:
            _log(f"[ONTO] import sim_onto FAILED: {e}")
            return None

        if not hasattr(sim_onto, "build_demo_catalog"):
            _log("[ONTO] sim_onto.build_demo_catalog not found")
            return None

        try:
            return sim_onto.build_demo_catalog(dim=int(dim))
        except Exception as e:
            _log(f"[ONTO] build_demo_catalog FAILED: {e}")
            return None


    def _build_onto_ctx(agent: SimAgent, turn: int):
        """Create sim_onto.AgentContext(vars=..., turn=...) (best-effort)."""
        try:
            import sim_onto  # type: ignore
        except Exception:
            return None

        if not hasattr(sim_onto, "AgentContext"):
            _log("[ONTO] sim_onto.AgentContext not found")
            return None

        AgentContext = getattr(sim_onto, "AgentContext")
        try:
            return AgentContext(vars=dict(agent.state.vars), turn=int(turn))
        except Exception as e:
            _log(f"[ONTO] AgentContext init FAILED: {e}")
            return None


    def _query_allowed_action_scores(agent: SimAgent, turn: int, top_k: int = 5) -> Dict[str, float]:
        """Return allowed action_id -> similarity for allowed actions."""
        catalog = _get_onto_catalog(agent.dim)
        if catalog is None or not hasattr(catalog, "query_actions"):
            _log("[ONTO] catalog.query_actions not available")
            return {}

        ctx_onto = _build_onto_ctx(agent, turn)
        if ctx_onto is None:
            return {}

        preferred = _preferred_dir_to_comfort(agent)
        # If preferred direction is the zero vector (common at turn 0 when current==comfort),
        # use a deterministic pseudo-random direction so ontology can break ties.
        if all(abs(v) < 1e-12 for v in preferred):
            # Reuse the existing deterministic random generator for reproducibility.
            fallback_dir, _ = default_random_event(agent.dim, int(turn))
            preferred = list(fallback_dir)
            _log(f"[ONTO] preferred_dir was zero; using fallback_dir for tie-break at turn={turn}")
        try:
            results = catalog.query_actions(preferred, ctx_onto, top_k=int(top_k), min_similarity=-1.0)
        except Exception as e:
            _log(f"[ONTO] query_actions FAILED: {e}")
            return {}

        _log(f"[ONTO] preferred_dir={ [round(x, 4) for x in preferred] }")

        allowed: Dict[str, float] = {}
        for r in results:
            try:
                aid = str(r.get("action_id", ""))
                sim = float(r.get("similarity", 0.0))
                allowed_flag = bool(r.get("allowed", False))
                precond = str(r.get("precond", ""))
            except Exception:
                continue

            _log(f"[ONTO] - action_id={aid} sim={sim:.3f} allowed={allowed_flag} precond={precond}")
            if aid and allowed_flag:
                allowed[aid] = sim

        return allowed


    def _action_score(action_id: str, allowed_scores: Dict[str, float]) -> float:
        """Higher is better. Use ontology cosine similarity directly.

        - Missing action_id: very low
        - Not allowed: low
        - Allowed: similarity ([-1,1])
        """
        if not action_id:
            return -10_000.0
        if action_id not in allowed_scores:
            return -1_000.0
        return float(allowed_scores[action_id])
    def choose_from_scenario(round_idx: int, chooser_id: str, chooser_agent: SimAgent) -> Dict[str, object]:
        """Pick one scenario choice for the current round.

        Uses the chooser agent's tag preferences (stored in AgentState.vars['tags'])
        to build a SelectionContext, then picks the top-ranked option.
        """
        pref_tags = chooser_agent.state.vars.get("tags", [])
        if not isinstance(pref_tags, list):
            pref_tags = []

        _log(f"\n[SELECT] round={round_idx} chooser={chooser_id} pref_tags={_fmt_tags(pref_tags)}")

        # Build a SelectionContext using the chooser agent's current position and comfort center.
        sim_turn = sim_a.turn if chooser_id == "A" else sim_b.turn
        ctx = _make_ctx(
            turn=sim_turn,
            current_vec=chooser_agent.get_current_vec(),
            comfort_vec=chooser_agent.state.comfort_vec,
            tags=pref_tags,
        )

        # Keep session context in sync (ScenarioSession API expects ctx at construction)
        try:
            scenario.ctx = ctx  # type: ignore[attr-defined]
        except Exception:
            pass

        # rank_choices signature may be (round_id, ctx) depending on sim_scenario implementation
        round_id = None
        for attr in ("round_id", "current_round_id", "cur_round_id", "current_round", "round"):
            if hasattr(scenario, attr):
                try:
                    round_id = getattr(scenario, attr)
                    break
                except Exception:
                    pass
        if round_id is None:
            round_id = round_idx

        # The demo scenario may have fewer rounds than we run. Clamp/cycle round_id safely.
        max_round_id: Optional[int] = None
        try:
            sc = getattr(manager, "scenario", None)
            rounds_obj = getattr(sc, "rounds", None)
            if isinstance(rounds_obj, dict) and rounds_obj:
                keys = [int(k) for k in rounds_obj.keys()]
                max_round_id = max(keys)
        except Exception:
            max_round_id = None

        rid = int(round_id)
        if max_round_id is not None and max_round_id >= 1:
            rid = ((rid - 1) % max_round_id) + 1

        _log(f"[SELECT] scenario_round_id={rid} sim_turn={sim_turn}")

        try:
            ranked = manager.rank_choices(rid, ctx)  # type: ignore[misc]
        except KeyError:
            ranked = manager.rank_choices(1, ctx)  # type: ignore[misc]
        except TypeError:
            ranked = manager.rank_choices(ctx)  # type: ignore[call-arg]

        if not ranked:
            raise ValueError("No ranked choices returned for current round")

        # Use ontology action catalog to rank feasible actions (action_id), then re-rank scenario choices.
        allowed_scores = _query_allowed_action_scores(chooser_agent, sim_turn, top_k=5)
        _log(f"[SELECT] allowed_action_ids={list(allowed_scores.keys())}")

        cand_meta: List[Dict[str, object]] = []
        for idx, c in enumerate(ranked):
            act = c.get("action") or {}
            if not isinstance(act, dict):
                act = {}

            action_id = str(act.get("action_id", ""))
            embed_text = str(act.get("embed_text", ""))
            score = _action_score(action_id, allowed_scores)

            disp = str(c.get("display_text", ""))
            disp_short = (disp[:70] + "...") if len(disp) > 70 else disp

            _log(f"[CAND {idx}] id={c.get('choice_id')} action_id={action_id} score={score:.3f} | {disp_short}")
            _log(f"          embed_text='{embed_text}'")

            cand_meta.append({"choice": c, "action_id": action_id, "score": score})

        ranked_choices = [m["choice"] for m in cand_meta]

        best = ranked_choices[0]
        best_score = float(cand_meta[0]["score"])
        best_action_id = str(cand_meta[0]["action_id"])

        for m in cand_meta[1:]:
            sc_ = float(m["score"])
            if sc_ > best_score:
                best_score = sc_
                best = m["choice"]
                best_action_id = str(m["action_id"])

        # Attach selection telemetry (non-breaking)
        try:
            if isinstance(best, dict):
                best = dict(best)
                best["onto_action_id"] = best_action_id
                best["onto_score"] = float(best_score)
        except Exception:
            pass

        ranked = [best] + [x for x in ranked_choices if x is not best]
        _log(
            f"[SELECT] picked_by_ontology_action id={(best.get('choice_id') if isinstance(best, dict) else None)} action_id={best_action_id} score={best_score:.3f}"
        )

        # Pick the top candidate and (optionally) advance scenario state.
        top = ranked[0]
        chosen_id = str(top.get("choice_id"))

        used_method: Optional[str] = None
        record: Dict[str, object] = {}

        for meth in ("select", "choose", "pick", "apply", "advance", "step", "commit", "take"):
            if hasattr(scenario, meth):
                fn = getattr(scenario, meth)
                try:
                    out = fn(chosen_id)
                    used_method = meth
                except TypeError:
                    try:
                        out = fn(int(chosen_id))
                        used_method = meth
                    except Exception:
                        continue
                except Exception:
                    continue

                if isinstance(out, dict):
                    record = out
                break

        if not record:
            record = dict(top)

        if used_method is None:
            _log("[SELECT] scenario_session: no-advance-method (using ranked choice only)")
        else:
            _log(f"[SELECT] scenario_session: advanced_via={used_method}")

        return {
            "choice_id": str(record.get("choice_id", chosen_id)),
            "display_text": str(record.get("display_text", top.get("display_text", ""))),
            "action": record.get("action", top.get("action", {})),
            "onto_action_id": record.get("onto_action_id", top.get("onto_action_id", "")),
            "onto_score": record.get("onto_score", top.get("onto_score", 0)),
        }

    dual = DualTurnSimulation(sim_a=sim_a, sim_b=sim_b, scenario=scenario, shared_action_fn=choose_from_scenario)
    results = dual.run(3)

    print("Comfort center:", agent_a.state.comfort_vec)
    print("Comfort radius:", agent_a.state.comfort_radius)

    for r in results:
        print("=" * 72)
        print(f"Round {r.round} | chooser={r.chooser} | choice={r.chosen_action.get('choice_id')}\n  {r.chosen_action.get('display_text')}")
        act = r.chosen_action.get("action", {})
        print(f"  ontology: action_id={r.chosen_action.get('onto_action_id')} score={r.chosen_action.get('onto_score')}")
        print(f"  action: action_id={act.get('action_id')} duration={act.get('duration')} magnitude={act.get('magnitude')} embed_text={act.get('embed_text')}")
        print("A: delta:", [round(x, 4) for x in r.a.event_delta], "pos:", [round(x, 4) for x in r.a.current_vec_after])
        print("B: delta:", [round(x, 4) for x in r.b.event_delta], "pos:", [round(x, 4) for x in r.b.current_vec_after])
        print(
            f"A in_comfort: {r.a.in_comfort_before}->{r.a.in_comfort_after} | dist={sim_a.distance_fn(r.a.current_vec_after, agent_a.state.comfort_vec):.4f}"
        )
        print(
            f"B in_comfort: {r.b.in_comfort_before}->{r.b.in_comfort_after} | dist={sim_b.distance_fn(r.b.current_vec_after, agent_b.state.comfort_vec):.4f}"
        )
