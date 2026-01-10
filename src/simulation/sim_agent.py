from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dto.agent import AgentSpec, AgentState
from .dto.impact import Impact
from .dto.vector_space import VectorSpaceSpec


@dataclass
class SimAgent:
    """
    런타임 객체.
    - spec(불변) + state(가변 SSOT) 보유
    - active_impacts는 상태에 영향을 주는 핵심 런타임 요소라서 여기에서 관리(실질적으로 state의 일부 성격)
    - state에 넣지 않는 '가변이지만 상태가 아닌 값'은 runtime_cache 같은 곳에 둔다.
    """
    space: VectorSpaceSpec
    spec: AgentSpec
    state: AgentState
    active_impacts: list[Impact] = field(default_factory=list)

    # 가변이지만 시뮬레이션 의미에 영향 X (캐시/디버그/메모이제이션 등) → 여기 둬도 OK
    runtime_cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.space.validate_vec(self.spec.base_vec, "spec.base_vec")
        self.space.validate_vec(self.spec.comfort_vec, "spec.comfort_vec")
        self.space.validate_vec(self.state.current_vec, "state.current_vec")

    @property
    def agent_id(self) -> str:
        return self.spec.agent_id

    def distance_to_comfort(self) -> float:
        return float(np.linalg.norm(self.state.current_vec - self.spec.comfort_vec))

    def in_comfort(self, threshold: float = 1e-9) -> bool:
        return self.distance_to_comfort() <= threshold

    def inject_impacts(self, impacts: list[Impact]) -> None:
        if not impacts:
            return
        for imp in impacts:
            # direction dim 검증은 dynamics에서 하도록 두되,
            # 여기서는 최소한 ndarray인지 정도는 체크해도 좋음
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
