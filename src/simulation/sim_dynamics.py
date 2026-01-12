from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.simulation.dto.impact import Impact
from src.simulation.dto.vector_space import VectorSpaceSpec
from src.simulation.sim_agent import SimAgent


@dataclass
class SimDynamics:
    """
    Impact 기반 전이만 적용.
    - 노이즈/클램프/랜덤/감쇠 등 Impact 외 공식은 아직 적용하지 않음.
    - 대신 확장 훅 자리는 마련해 둠.
    """
    space: VectorSpaceSpec
    pre_update_hooks: list[Callable[[SimAgent], None]] = field(default_factory=list)
    post_update_hooks: list[Callable[[SimAgent], None]] = field(default_factory=list)

    def apply(self, agent: SimAgent) -> None:
        # hooks (자리만)
        for h in self.pre_update_hooks:
            h(agent)

        # validate
        self.space.validate_vec(agent.state.current_vec, "agent.state.current_vec")

        # 1) 벡터 합산
        delta = self.space.zeros()
        for imp in agent.active_impacts:
            self.space.validate_vec(imp.direction, "impact.direction")
            delta = delta + imp.delta_vec()

        agent.state.current_vec = agent.state.current_vec + delta

        # 2) vars 합산
        if agent.active_impacts:
            if agent.state.vars is None:
                agent.state.vars = {}
            for imp in agent.active_impacts:
                for k, dv in imp.delta_vars.items():
                    agent.state.vars[k] = float(agent.state.vars.get(k, 0.0) + float(dv))

        # hooks (자리만)
        for h in self.post_update_hooks:
            h(agent)
