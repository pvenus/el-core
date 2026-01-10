from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .dto.vector_space import AxisSpec, VectorSpaceSpec
from .dto.agent import AgentSpec, AgentState
from .dto.impact import Impact
from .dto.step_io import StepInput
from .sim_agent import SimAgent
from .sim_dynamics import SimDynamics
from .sim_runner import StepRunner
from .sim_manager import SimulationManager


@dataclass
class DemoSpaceBuilder:
    dim: int = 27
    seed: int = 42

    def build(self) -> VectorSpaceSpec:
        rng = np.random.default_rng(self.seed)
        axes: list[AxisSpec] = []

        for i in range(self.dim):
            axes.append(
                AxisSpec(
                    key=f"axis_{i:02d}",
                    tags=[f"tag{i % 3}"],
                    proto_vec=rng.normal(size=(self.dim,)).astype(float),
                )
            )

        return VectorSpaceSpec(dim=self.dim, axes=axes)


@dataclass
class DemoAgentBuilder:
    n_agents: int = 2
    seed: int = 123

    def build(self, space: VectorSpaceSpec) -> list[SimAgent]:
        rng = np.random.default_rng(self.seed)
        agents: list[SimAgent] = []

        for i in range(self.n_agents):
            base = rng.normal(size=(space.dim,))
            comfort = rng.normal(size=(space.dim,))

            spec = AgentSpec(
                agent_id=f"agent_{i+1}",
                name=f"Agent {i+1}",
                base_vec=base,
                comfort_vec=comfort,
                vars_base={"hp": 100.0},
            )

            state = AgentState(
                current_vec=base.copy(),
                vars=dict(spec.vars_base),
                step_idx=0,
            )

            agents.append(SimAgent(space=space, spec=spec, state=state))

        return agents


def build_demo_manager(dim: int = 12, n_agents: int = 2, seed: int = 42) -> SimulationManager:
    space = DemoSpaceBuilder(dim=dim, seed=seed).build()
    agents = DemoAgentBuilder(n_agents=n_agents, seed=seed + 1).build(space)

    dynamics = SimDynamics(space=space)
    runner = StepRunner(dynamics=dynamics)

    return SimulationManager(space_spec=space, agents=agents, runner=runner)


def build_demo_step_input(agent_ids: list[str], dim: int, seed: int = 7) -> StepInput:
    rng = np.random.default_rng(seed)

    impacts: dict[str, list[Impact]] = {}
    for aid in agent_ids:
        impacts[aid] = [
            Impact(
                direction=rng.normal(size=(dim,)),
                magnitude=float(rng.uniform(0.1, 1.0)),
                duration=1,
                delta_vars={"mood": float(rng.uniform(-0.5, 0.5))},
            )
        ]

    return StepInput(impacts_by_agent=impacts, metadata={"demo": True})
