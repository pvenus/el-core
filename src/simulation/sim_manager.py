from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .dto.agent import AgentSpec, AgentState
from .dto.vector_space import VectorSpaceSpec
from .dto.step_io import StepInput, StepResult
from .sim_agent import SimAgent
from .sim_runner import StepRunner
from .sim_dynamics import SimDynamics
from .sim_protocols import AgentSpecSource


VarsTemplateMode = Literal["merge", "replace"]


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
        seed: int | None = None,
        agent_count: int | None = None,
        vars_template: dict[str, float] | None = None,
        vars_template_mode: VarsTemplateMode = "merge",
    ) -> "SimulationManager":
        """
        외부에서는 create만 호출하면 조립이 끝나게.
        - seed: spec 선택/샘플링 재현용(셔플+트림)
        - agent_count: source가 반환한 spec 중 N개만 사용
        - vars_template: 초기 vars 템플릿 오버라이드
        - vars_template_mode:
            - "merge": base(spec.vars)에 vars_template 덮어쓰기
            - "replace": vars_template로 완전 교체
        """
        specs = agent_spec_source.build_agent_specs(space_spec)

        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(specs)

        if agent_count is not None:
            if agent_count <= 0:
                raise ValueError("agent_count must be > 0")
            if len(specs) < agent_count:
                raise ValueError(f"agent_count={agent_count} but specs={len(specs)}")
            specs = specs[:agent_count]

        agents: list[SimAgent] = []
        for spec in specs:
            # 초기 state 규칙(형이 확정한 규칙)
            current_vec = spec.base_vec.copy()

            if vars_template_mode == "replace":
                base_vars = dict(vars_template or {})
            else:
                base_vars = dict(spec.vars)
                if vars_template:
                    base_vars.update({k: float(v) for k, v in vars_template.items()})

            state = AgentState(
                current_vec=current_vec,
                vars=base_vars,
                step_idx=0,
            )
            agents.append(SimAgent(space=space_spec, spec=spec, state=state))

        return SimulationManager(space_spec=space_spec, agents=agents)

    def step(self, step_input: StepInput) -> StepResult:
        self.step_idx += 1
        assert self.runner is not None
        result = self.runner.run(self._agents, step_input, self.step_idx)
        self.history.append(result)
        return result

    def snapshot(self) -> dict:
        return {
            aid: {
                "vec": a.state.current_vec.copy(),
                "vars": dict(a.state.vars),
                "active_impacts": len(a.active_impacts),
                "distance_to_comfort": a.distance_to_comfort(),
            }
            for aid, a in self._agents.items()
        }


def main() -> None:

    from .sim_builders import DemoSpaceBuilder, DemoAgentSpecSource, build_demo_step_input

    space = DemoSpaceBuilder(dim=8, seed=42).build()
    spec_source = DemoAgentSpecSource(n_agents=4, seed=7)

    mgr = SimulationManager.create(
        space_spec=space,
        agent_spec_source=spec_source,
        seed=999,                 # spec 선택 순서 재현
        agent_count=2,            # 4명 중 2명만
        vars_template={"hp": 80}, # 초기 vars override
        vars_template_mode="merge",
    )

    agent_ids = [a.agent_id for a in mgr.agents]
    step_in = build_demo_step_input(agent_ids=agent_ids, dim=space.dim, seed=3)

    out = mgr.step(step_in)
    print("step:", out.step_idx)
    for r in out.per_agent:
        print(r.agent_id, "delta_norm", round(float(r.metrics["delta_vec_norm"]), 4), "delta_vars", r.delta_vars)


if __name__ == "__main__":
    main()
