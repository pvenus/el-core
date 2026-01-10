from __future__ import annotations

from dataclasses import dataclass, field

from .dto.vector_space import VectorSpaceSpec
from .dto.step_io import StepInput, StepResult
from .sim_agent import SimAgent
from .sim_runner import StepRunner
from .sim_dynamics import SimDynamics


@dataclass
class SimulationManager:
    space_spec: VectorSpaceSpec
    agents: list[SimAgent]
    runner: StepRunner | None = None

    step_idx: int = 0
    history: list[StepResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._agents: dict[str, SimAgent] = {a.agent_id: a for a in self.agents}
        if self.runner is None:
            self.runner = StepRunner(dynamics=SimDynamics(space=self.space_spec))

    def step(self, step_input: StepInput) -> StepResult:
        self.step_idx += 1
        result = self.runner.run(self._agents, step_input, self.step_idx)
        self.history.append(result)
        return result

    def snapshot(self) -> dict:
        return {
            aid: {
                "vec": a.state.current_vec.copy(),
                "vars": dict(a.state.vars),
                "active_impacts": len(a.active_impacts),
            }
            for aid, a in self._agents.items()
        }


def main() -> None:
    from .sim_builders import build_demo_manager, build_demo_step_input

    mgr = build_demo_manager(dim=8, n_agents=2)
    agent_ids = [a.agent_id for a in mgr.agents]

    step_input = build_demo_step_input(agent_ids, dim=mgr.space_spec.dim)
    result = mgr.step(step_input)

    print("step:", result.step_idx)
    for r in result.per_agent:
        print(r.agent_id, r.metrics)


if __name__ == "__main__":
    main()
