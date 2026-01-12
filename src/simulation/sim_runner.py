from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.dto.step_io import StepInput, StepResult
from src.simulation.sim_agent import SimAgent
from src.simulation.sim_dynamics import SimDynamics

from src.ontology.pipeline import pipeline_quick_test


@dataclass
class StepRunner:
    dynamics: SimDynamics

    def run(self, agents: dict[str, SimAgent], step_input: StepInput, step_idx: int) -> StepResult:
        agent_id = step_input.agent_id
        if agent_id not in agents:
            raise KeyError(f"Unknown agent_id in StepInput: {agent_id}")

        picked = pipeline_quick_test(agents[agent_id].state.current_vec)
        impacts = []
        impacts.append(picked.impact)
        for agent_id, agent in agents.items():
            print(f"inject impact {agent_id} {picked.impact}")
            agent.inject_impacts(impacts)

        injected = list(step_input.impacts)
        agent.inject_impacts(injected)

        before_vec = agent.state.current_vec.copy()

        # apply dynamics on this single agent
        self.dynamics.apply(agent)
        agent.state.step_idx = step_idx

        expired = agent.tick_and_expire_impacts()

        after_vec = agent.state.current_vec.copy()
        delta_vec = after_vec - before_vec

        metrics = {
            "distance_to_comfort": agent.distance_to_comfort(),
            "in_comfort": agent.is_in_comfort(),
            "delta_vec_norm": float(np.linalg.norm(delta_vec)),
            "active_impacts_after": len(agent.active_impacts),
        }

        return StepResult(
            step_idx=step_idx,
            agent_id=agent_id,
            before_vec=before_vec,
            after_vec=after_vec,
            delta_vec=delta_vec,
            injected_impacts=injected,
            expired_impacts=list(expired),
            active_impacts_after=list(agent.active_impacts),
            metrics=metrics,
            metadata=dict(step_input.metadata),
        )
