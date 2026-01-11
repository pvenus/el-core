from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dto.step_io import StepInput, StepResult, AgentStepResult
from .sim_agent import SimAgent
from .sim_dynamics import SimDynamics
from src.ontology.pipeline import pipeline_quick_test


@dataclass
class StepRunner:
    """
    '한 스텝' 실행만 담당.
    - 흐름(히스토리/step_idx 증가 등)은 Manager가 담당.
    """

    dynamics: SimDynamics

    def run(self, agents: dict[str, SimAgent], step_input: StepInput, step_idx: int) -> StepResult:
        results: list[AgentStepResult] = []

        picked = pipeline_quick_test(agents['0'].state.current_vec)
        impacts = []
        impacts.append(picked.impact)
        for agent_id, agent in agents.items():
            print(f"inject impact {agent_id} {picked.impact}")
            agent.inject_impacts(impacts)

        # 1) impacts 주입
        #for agent_id, imps in step_input.impacts_by_agent.items():
        #    if agent_id not in agents:
        #        raise KeyError(f"Unknown agent_id in StepInput: {agent_id}")
        #    agents[agent_id].inject_impacts(imps)

        # 2) 각 agent 업데이트
        for agent_id, agent in agents.items():
            injected = step_input.impacts_by_agent.get(agent_id, [])

            before_vec = agent.state.current_vec.copy()
            before_vars = dict(agent.state.vars) if agent.state.vars is not None else {}

            # 전이
            self.dynamics.apply(agent)

            # step index 반영(상태 SSOT)
            agent.state.step_idx = step_idx

            # 만료 처리(정책: 스텝 끝에서 tick)
            expired = agent.tick_and_expire_impacts()

            after_vec = agent.state.current_vec.copy()
            after_vars = dict(agent.state.vars) if agent.state.vars is not None else {}

            delta_vec = after_vec - before_vec
            delta_vars: dict[str, float] = {}
            keys = set(before_vars.keys()) | set(after_vars.keys())
            for k in keys:
                delta_vars[k] = float(after_vars.get(k, 0.0) - before_vars.get(k, 0.0))

            metrics = {
                "distance_to_comfort": agent.distance_to_comfort(),
                "in_comfort": agent.in_comfort(),
                "delta_vec_norm": float(np.linalg.norm(delta_vec)),
            }

            results.append(
                AgentStepResult(
                    agent_id=agent_id,
                    before_vec=before_vec,
                    after_vec=after_vec,
                    delta_vec=delta_vec,
                    before_vars=before_vars,
                    after_vars=after_vars,
                    injected_impacts=list(impacts),
                    expired_impacts=list(expired),
                    active_impacts_after=list(agent.active_impacts),
                    metrics=metrics,
                )
            )

        return StepResult(step_idx=step_idx, per_agent=results, metadata=dict(step_input.metadata))
