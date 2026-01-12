from __future__ import annotations

from dataclasses import dataclass, field

from src.simulation.dto.agent import AgentSpec, AgentState
from src.simulation.dto.vector_space import VectorSpaceSpec
from src.simulation.dto.step_io import StepInput, StepResult
from src.simulation.sim_agent import SimAgent
from src.simulation.sim_runner import StepRunner
from src.simulation.sim_dynamics import SimDynamics
from src.simulation.sim_protocols import AgentSpecSource


@dataclass
class SimulationManager:
    space_spec: VectorSpaceSpec
    agents: list[SimAgent]
    runner: StepRunner | None = None

    step_idx: int = 0
    history: list[StepResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._agents: dict[str, SimAgent] = {a.agent_id: a for a in self.agents}
        if len(self._agents) != len(self.agents):
            raise ValueError("Duplicate agent_id detected")

        if self.runner is None:
            self.runner = StepRunner(dynamics=SimDynamics(space=self.space_spec))

    @staticmethod
    def create(
        space_spec: VectorSpaceSpec,
        agent_spec_source: AgentSpecSource,
        *,
        agent_count: int | None = None,
    ) -> "SimulationManager":
        specs: list[AgentSpec] = agent_spec_source.build_agent_specs(space_spec)

        if agent_count is not None:
            if agent_count <= 0:
                raise ValueError("agent_count must be > 0")
            if len(specs) < agent_count:
                raise ValueError(f"agent_count={agent_count} but specs={len(specs)}")
            specs = specs[:agent_count]

        agents: list[SimAgent] = []
        for spec in specs:
            # 형 정책: 초기 current_vec = comfort_vec
            state = AgentState(current_vec=spec.comfort_vec.copy(), step_idx=0, vars=None)
            agents.append(SimAgent(space=space_spec, spec=spec, state=state))

        return SimulationManager(space_spec=space_spec, agents=agents)

    def step(self, step_input: StepInput) -> StepResult:
        self.step_idx += 1
        assert self.runner is not None
        result = self.runner.run(self._agents, step_input, self.step_idx)
        self.history.append(result)
        return result

    def snapshot(self) -> dict:
        out = {}
        for aid, a in self._agents.items():
            out[aid] = {
                "name": a.spec.name,
                "radius": float(a.spec.radius),
                "comfort_vec": a.spec.comfort_vec.copy(),
                "current_vec": a.state.current_vec.copy(),
                "distance_to_comfort": a.distance_to_comfort(),
                "is_in_comfort": a.is_in_comfort(),
                "active_impacts": len(a.active_impacts),
            }
        return out


def main() -> None:
    """
    실행:
      PYTHONPATH=src python -m simulation.sim_manager
    """
    from src.simulation.sim_builders import DemoSpaceBuilder, DemoAgentSpecSource, build_demo_step_input_json
    from src.simulation.dto.step_io import StepInput

    space = DemoSpaceBuilder().build()
    spec_source = DemoAgentSpecSource()

    mgr = SimulationManager.create(space_spec=space, agent_spec_source=spec_source, agent_count=1)

    # ✅ demo stepInput json(하드코딩) -> StepInput 변환
    step_json = build_demo_step_input_json(space_dim=space.dim, agent_id=mgr.agents[0].agent_id)
    step_input = StepInput.from_dict(step_json)

    out = mgr.step(step_input)
    print("step:", out.step_idx)
    for r in out.per_agent:
        print(r.agent_id, "distance", r.metrics.get("distance_to_comfort"), "in", r.metrics.get("in_comfort"))


if __name__ == "__main__":
    main()
