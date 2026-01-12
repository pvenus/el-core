from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.simulation.dto.agent import AgentSpec, AgentState
from src.simulation.dto.impact import Impact
from src.simulation.dto.vector_space import VectorSpaceSpec


@dataclass
class SimAgent:
    """
    런타임 객체
    - spec(불변): comfort_vec, radius
    - state(가변): current_vec, step_idx
    """
    space: VectorSpaceSpec
    spec: AgentSpec
    state: AgentState
    active_impacts: list[Impact] = field(default_factory=list)

    # 보류(캐시/디버그)
    runtime_cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.space.validate_vec(self.spec.comfort_vec, "spec.comfort_vec")
        self.space.validate_vec(self.state.current_vec, "state.current_vec")
        if self.spec.radius < 0:
            raise ValueError("spec.radius must be >= 0")

    @property
    def agent_id(self) -> str:
        return self.spec.agent_id

    def distance_to_comfort(self) -> float:
        return float(np.linalg.norm(self.state.current_vec - self.spec.comfort_vec))

    def is_in_comfort(self) -> bool:
        return self.distance_to_comfort() <= float(self.spec.radius)

    def inject_impacts(self, impacts: list[Impact]) -> None:
        if not impacts:
            return
        for imp in impacts:
            if not isinstance(imp.direction, np.ndarray):
                raise TypeError("Impact.direction must be np.ndarray")
            self.active_impacts.append(imp)

    def tick_and_expire_impacts(self) -> list[Impact]:
        expired: list[Impact] = []
        for imp in self.active_impacts:
            imp.tick()
            if imp.expired:
                expired.append(imp)
        if expired:
            self.active_impacts = [i for i in self.active_impacts if not i.expired]
        return expired
