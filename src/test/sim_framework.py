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
from typing import Callable, Dict, List, Optional, Tuple

from sim_agent import SimAgent, AgentState, add, scale, normalize


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
# Minimal demo
# -----------------------------

if __name__ == "__main__":
    # Build an agent with a comfort-zone definition (center position + radius)
    agent = SimAgent(
        dim=6,
        state=AgentState(
            comfort_vec=[-0.10, 0.20, -0.05, 0.25, 0.05, -0.05],
            comfort_radius=0.35,
            current_vec=[-0.10, 0.20, -0.05, 0.25, 0.05, -0.05],
            vars={"stamina": 0.6, "energy": 0.6},
        ),
    )

    # Add a 2-turn event starting at turn 1


    sim = TurnSimulation(
        agent=agent,
        spawn_when_in_comfort=True,
        spawn_probability=0.30,
        spawn_duration=1,
    )

    results = sim.run(5)

    print("Comfort center:", agent.state.comfort_vec)
    print("Comfort radius:", agent.state.comfort_radius)
    print("Start current:", results[0].current_vec_before)

    for r in results:
        print("-" * 72)
        print(f"Turn {r.turn}")
        print("  delta:", [round(x, 4) for x in r.event_delta])
        print("  pos:", [round(x, 4) for x in r.current_vec_after])
        print(f"  in_comfort: {r.in_comfort_before} -> {r.in_comfort_after} | dist={sim.distance_fn(r.current_vec_after, agent.state.comfort_vec):.4f}")
        if r.spawned_event:
            print("  spawned_event:", r.spawned_event)
