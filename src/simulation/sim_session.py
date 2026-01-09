# sim_engine/sim_session.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from simulation.dto.vector_space import VectorSpaceSpec
from simulation.sim_space_provider import SimSpaceProvider
from simulation.sim_agent_factory import SimAgentFactory
from simulation.sim_agent import SimAgent
from simulation.sim_runner import SimRunner
from simulation.dto.turn import TurnInput, TurnResult


@dataclass
class SimSession:
    """
    실행 컨텍스트 컨테이너.
    - space_spec은 provider에서 주입(또는 UI 텍스트로 생성) 가능
    - runner는 step만 담당
    - session은 history/reset 같은 편의 담당
    """
    provider: SimSpaceProvider
    agent_factory: SimAgentFactory = field(default_factory=SimAgentFactory)

    space: Optional[VectorSpaceSpec] = None
    agent: Optional[SimAgent] = None
    runner: Optional[SimRunner] = None
    history: List[TurnResult] = field(default_factory=list)

    def ensure_ready(self) -> None:
        """
        space/agent/runner가 없으면 provider로 로드해서 준비.
        """
        if self.space is None:
            self.space = self.provider.load()
        self.space.validate()

        if self.agent is None:
            self.agent = self.agent_factory.create_from_space(self.space)

        if self.runner is None:
            self.runner = SimRunner(self.agent)

    def set_space(self, space: VectorSpaceSpec, reset: bool = True) -> None:
        """
        외부에서 space_spec 주입하는 구간.
        reset=True면 agent/runner/history까지 같이 리셋(권장)
        """
        space.validate()
        self.space = space
        self.provider.save(space)

        if reset:
            self.reset()

    def reset(self) -> None:
        """
        현재 space 기준으로 agent/runner/history 초기화.
        """
        if self.space is None:
            self.space = self.provider.load()
        self.space.validate()

        self.agent = self.agent_factory.create_from_space(self.space)
        self.runner = SimRunner(self.agent)
        self.history = []

    def step(self, turn_input: TurnInput) -> TurnResult:
        self.ensure_ready()
        assert self.runner is not None

        result = self.runner.step_with_input(turn_input)
        self.history.append(result)
        return result

    def run_n(self, n: int, turn_input: TurnInput) -> List[TurnResult]:
        """
        동일 입력을 n번 반복 적용(테스트/데모용)
        """
        out: List[TurnResult] = []
        for _ in range(int(n)):
            out.append(self.step(turn_input))
        return out
