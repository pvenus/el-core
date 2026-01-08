# sim_engine/sim_agent.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from .dto.agent import AgentSpec, AgentState

Vector = List[float]
Vars = Dict[str, float]


@dataclass
class SimAgent:
    """
    런타임 에이전트 객체.
    - spec(불변) + state(가변) 보유
    - 시뮬레이션 전이(dynamics)는 외부에서 수행하고,
      SimAgent는 보관/초기화/교체 등 최소 유틸만 제공한다.
    """
    spec: AgentSpec
    state: AgentState

    def __post_init__(self) -> None:
        self.spec.validate()
        self.state.validate(self.spec)

    @property
    def dim(self) -> int:
        return self.spec.dim

    def snapshot(self) -> Dict:
        return {
            "turn": int(self.state.turn),
            "current_vec": list(self.state.current_vec),
            "vars": dict(self.state.vars),
            "comfort_vec": list(self.spec.comfort_vec),
            "comfort_radius": float(self.spec.comfort_radius),
            "spec_vars": dict(self.spec.vars),
            "spec_meta": dict(self.spec.meta),
        }

    def reset(self, init_vec: Optional[Vector] = None, init_vars: Optional[Vars] = None) -> None:
        """
        turn=0으로 초기화.
        - init_vars를 안 주면 spec.vars를 기본값으로 사용(추천 기본 정책)
        """
        v = list(init_vec) if init_vec is not None else [0.0] * self.dim
        if init_vars is None:
            vs = dict(self.spec.vars)  # spec vars -> state vars 기본 복사
        else:
            vs = dict(init_vars)

        self.state = AgentState(turn=0, current_vec=v, vars=vs)
        self.state.validate(self.spec)

    def set_state(self, new_state: AgentState) -> None:
        new_state.validate(self.spec)
        self.state = new_state

    def advance_turn(self) -> None:
        self.state = replace(self.state, turn=self.state.turn + 1)
