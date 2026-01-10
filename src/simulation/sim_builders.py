from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dto.vector_space import AxisSpec, VectorSpaceSpec
from .dto.agent import AgentSpec
from .dto.impact import Impact
from .dto.step_io import StepInput
from .sim_protocols import AgentSpecSource


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
class DemoAgentSpecSource(AgentSpecSource):
    """
    - AgentSpecSource 구현체
    """
    n_agents: int = 2
    seed: int = 123

    def build_agent_specs(self, space: VectorSpaceSpec) -> list[AgentSpec]:
        rng = np.random.default_rng(self.seed)
        specs: list[AgentSpec] = []
        for i in range(self.n_agents):
            base = rng.normal(size=(space.dim,)).astype(float)
            comfort = rng.normal(size=(space.dim,)).astype(float)
            specs.append(
                AgentSpec(
                    agent_id=f"agent_{i+1}",
                    name=f"Agent {i+1}",
                    base_vec=base,
                    comfort_vec=comfort,
                    vars={"hp": 100.0},
                    meta={"demo": True},
                )
            )
        return specs


def build_demo_step_input(agent_ids: list[str], dim: int, seed: int = 7) -> StepInput:
    rng = np.random.default_rng(seed)
    impacts_by_agent: dict[str, list[Impact]] = {}

    for aid in agent_ids:
        impacts_by_agent[aid] = [
            Impact(
                direction=rng.normal(size=(dim,)).astype(float),
                magnitude=float(rng.uniform(0.1, 1.0)),
                duration=1,
                delta_vars={"mood": float(rng.uniform(-0.5, 0.5))},
                profile={"src": "demo"},
            )
        ]

    return StepInput(impacts_by_agent=impacts_by_agent, metadata={"demo": True})
