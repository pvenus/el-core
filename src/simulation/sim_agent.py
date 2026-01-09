# sim_engine/sim_agent.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from .sim_vector import l2_distance
from .sim_types import Vector, Vars

from .dto.agent import AgentSpec, AgentState
from .dto.impact import Impact


@dataclass
class SimAgent:
    """
    런타임 Agent:
    - spec(불변) + state(가변)
    - active_impacts 스택 소유
    - comfort 관측/판정 함수 제공 (전이 규칙 적용은 금지)
    """
    spec: AgentSpec
    state: AgentState
    active_impacts: List[Impact] = None

    def __post_init__(self) -> None:
        self.spec.validate()
        self.state.validate(self.spec)
        if self.active_impacts is None:
            self.active_impacts = []
        else:
            for imp in self.active_impacts:
                imp.validate(self.spec.dim)

    @classmethod
    def create(
        cls,
        dim: int = 8,
        comfort_radius: float = 1.0,
        spec_vars: Optional[Vars] = None,
        spec_meta: Optional[Dict] = None,
        init_vec: Optional[Vector] = None,
        init_vars: Optional[Vars] = None,
    ) -> "SimAgent":
        comfort_vec = [0.0] * dim
        spec = AgentSpec(
            dim=dim,
            comfort_vec=comfort_vec,
            comfort_radius=float(comfort_radius),
            vars=dict(spec_vars or {}),
            meta=dict(spec_meta or {}),
        )
        state = AgentState(
            turn=0,
            current_vec=list(init_vec) if init_vec is not None else [0.0] * dim,
            vars=dict(init_vars) if init_vars is not None else dict(spec.vars),
        )
        return cls(spec=spec, state=state)

    @property
    def dim(self) -> int:
        return self.spec.dim

    # ---- comfort helpers (observation only) ----
    def distance_to_comfort(self, state: Optional[AgentState] = None) -> float:
        s = state or self.state
        return l2_distance(s.current_vec, self.spec.comfort_vec)

    def in_comfort(self, state: Optional[AgentState] = None) -> bool:
        return self.distance_to_comfort(state) <= float(self.spec.comfort_radius)

    # ---- impact stack ops ----
    def add_impacts(self, impacts: List[Impact]) -> None:
        for imp in impacts:
            imp.validate(self.dim)
        self.active_impacts.extend(impacts)

    def snapshot_active_impacts(self) -> List[Impact]:
        # 이번 턴 적용 대상 스냅샷(얕은 복사)
        return list(self.active_impacts)

    def tick_impacts(self) -> None:
        """
        duration 1 감소 & 만료 제거.
        - "이번 턴 적용 후" 호출해야 의미가 맞음.
        """
        next_list: List[Impact] = []
        for imp in self.active_impacts:
            if imp.duration > 1:
                next_list.append(replace(imp, duration=imp.duration - 1))
        self.active_impacts = next_list

    def set_state(self, new_state: AgentState) -> None:
        new_state.validate(self.spec)
        self.state = new_state
