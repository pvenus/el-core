"""Independent agent data model for emotion simulation.

What this module provides (and ONLY provides):
- Agent state storage (comfort zone center+radius, live current position, scalar vars).
- Event history storage (multi-turn events).
- Simple helper queries (distance-to-comfort, in-comfort check, direction-to-comfort).

This file intentionally does NOT:
- choose actions
- apply ontology rules
- run a time loop (that's the simulation's job)

Event format:
- start_turn: int (when the event starts, inclusive)
- duration: int (how many turns it stays active; >= 1)
- direction: a unit direction vector (normalized on insert)
- magnitude: float (event strength)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# -----------------------------
# Vector helpers (minimal)
# -----------------------------

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: List[float]) -> float:
    return (_dot(a, a) ** 0.5)


def normalize(v: List[float], eps: float = 1e-12) -> List[float]:
    """Return a unit vector. If norm is ~0, returns a zero vector."""
    n = _norm(v)
    if n < eps:
        return [0.0 for _ in v]
    return [x / n for x in v]


def add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def l2_distance(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


# -----------------------------
# Core data types
# -----------------------------

@dataclass(frozen=True)
class EventRecord:
    """A single external event that can persist for multiple turns."""

    start_turn: int
    duration: int
    direction: List[float]  # unit vector
    magnitude: float        # scalar strength

    def __post_init__(self):
        object.__setattr__(self, "start_turn", int(self.start_turn))

        d = int(self.duration)
        if d < 1:
            raise ValueError("duration must be >= 1")
        object.__setattr__(self, "duration", d)

        object.__setattr__(self, "direction", normalize(list(self.direction)))
        object.__setattr__(self, "magnitude", float(self.magnitude))

    def is_active(self, turn: int) -> bool:
        t = int(turn)
        return self.start_turn <= t < (self.start_turn + self.duration)

    def remaining_turns(self, turn: int) -> int:
        """Remaining active turns including `turn` if active; else 0."""
        t = int(turn)
        if not self.is_active(t):
            return 0
        end = self.start_turn + self.duration
        return max(0, end - t)


@dataclass
class AgentState:
    """Agent runtime state.

    Your definition:
    - comfort_vec is NOT a direction; it's the *center position* of a comfort region.
    - comfort_radius defines the region size around comfort_vec.
    - current_vec is the agent's live emotion-space position.

    All vectors are stored on raw scale (do NOT normalize here).
    """

    comfort_vec: List[float]
    comfort_radius: float = 0.25
    current_vec: List[float] = field(default_factory=list)
    vars: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.comfort_vec = list(self.comfort_vec)
        self.comfort_radius = float(self.comfort_radius)
        if self.comfort_radius < 0:
            raise ValueError("comfort_radius must be >= 0")

        self.current_vec = list(self.current_vec)


DistanceFn = Callable[[List[float], List[float]], float]


@dataclass
class SimAgent:
    """Independent agent storing ONLY:
    - comfort zone (comfort_vec + comfort_radius)
    - live current position (current_vec)
    - scalar vars
    - event history

    No time loop or decision logic.
    """

    dim: int
    state: AgentState
    events: List[EventRecord] = field(default_factory=list)

    def __post_init__(self):
        if len(self.state.comfort_vec) != self.dim:
            raise ValueError(f"comfort_vec dim mismatch: expected {self.dim}, got {len(self.state.comfort_vec)}")
        if len(self.state.current_vec) != self.dim:
            raise ValueError(
                f"current_vec dim mismatch: expected {self.dim}, got {len(self.state.current_vec)}. "
                f"Set AgentState.current_vec when creating the agent."
            )

    # ---- events ----

    def add_event(self, start_turn: int, duration: int, direction: List[float], magnitude: float) -> None:
        """Append a (possibly multi-turn) event to history."""
        if len(direction) != self.dim:
            raise ValueError(f"direction dim mismatch: expected {self.dim}, got {len(direction)}")
        self.events.append(
            EventRecord(
                start_turn=start_turn,
                duration=duration,
                direction=direction,
                magnitude=magnitude,
            )
        )

    def events_active_at_turn(self, turn: int) -> List[EventRecord]:
        t = int(turn)
        return [e for e in self.events if e.is_active(t)]

    def last_event_turn(self) -> Optional[int]:
        if not self.events:
            return None
        return max((e.start_turn + e.duration - 1) for e in self.events)

    def event_delta_at_turn(self, turn: int) -> List[float]:
        """Sum(direction*magnitude) for events active on the given turn."""
        delta = [0.0] * self.dim
        for e in self.events_active_at_turn(turn):
            delta = add(delta, scale(e.direction, e.magnitude))
        return delta

    # ---- position ----

    def get_current_vec(self) -> List[float]:
        return list(self.state.current_vec)

    def set_current_vec(self, v: List[float]) -> None:
        if len(v) != self.dim:
            raise ValueError(f"current_vec dim mismatch: expected {self.dim}, got {len(v)}")
        self.state.current_vec = list(v)

    # ---- comfort helpers (agent-side) ----

    def distance_to_comfort(
        self,
        pos: Optional[List[float]] = None,
        distance_fn: DistanceFn = l2_distance,
    ) -> float:
        """Distance from (pos or current_vec) to comfort center."""
        if pos is None:
            pos = self.get_current_vec()
        return float(distance_fn(pos, self.state.comfort_vec))

    def is_in_comfort(
        self,
        pos: Optional[List[float]] = None,
        distance_fn: DistanceFn = l2_distance,
    ) -> bool:
        return self.distance_to_comfort(pos, distance_fn) <= float(self.state.comfort_radius)

    def direction_to_comfort(self, pos: Optional[List[float]] = None) -> List[float]:
        """Unit direction FROM (pos or current_vec) TO comfort center."""
        if pos is None:
            pos = self.get_current_vec()
        return normalize(sub(self.state.comfort_vec, pos))

    # ---- vars ----

    def set_var(self, key: str, value: float) -> None:
        self.state.vars[str(key)] = float(value)

    def get_var(self, key: str, default: float = 0.0) -> float:
        return float(self.state.vars.get(str(key), default))

    # ---- debug ----

    def snapshot(self) -> Dict[str, object]:
        return {
            "dim": self.dim,
            "comfort_vec": list(self.state.comfort_vec),
            "comfort_radius": float(self.state.comfort_radius),
            "current_vec": list(self.state.current_vec),
            "vars": dict(self.state.vars),
            "events": [
                {
                    "start_turn": e.start_turn,
                    "duration": e.duration,
                    "direction": list(e.direction),
                    "magnitude": e.magnitude,
                }
                for e in self.events
            ],
        }


if __name__ == "__main__":
    agent = SimAgent(
        dim=6,
        state=AgentState(
            comfort_vec=[-0.10, 0.20, -0.05, 0.25, 0.05, -0.05],
            comfort_radius=0.35,
            current_vec=[-0.10, 0.20, -0.05, 0.25, 0.05, -0.05],
            vars={"stamina": 0.6, "energy": 0.6},
        ),
    )

    agent.add_event(start_turn=1, duration=2, direction=[0.25, -0.10, 0.30, 0.05, -0.20, 0.10], magnitude=0.8)

    print("Turn=1 delta:", agent.event_delta_at_turn(1))
    print("In comfort?", agent.is_in_comfort())
    print("Dir to comfort:", agent.direction_to_comfort())
    print("Snapshot:", agent.snapshot())