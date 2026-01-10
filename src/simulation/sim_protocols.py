from __future__ import annotations

from typing import Protocol, runtime_checkable

from .dto.vector_space import VectorSpaceSpec
from .sim_agent import SimAgent


@runtime_checkable
class SpaceSource(Protocol):
    """
    외부에서 VectorSpaceSpec을 제공하는 인터페이스
    """
    def load_space(self) -> VectorSpaceSpec:
        ...


@runtime_checkable
class AgentSource(Protocol):
    """
    외부에서 에이전트를 생성/제공하는 인터페이스
    """
    def build_agents(self, space: VectorSpaceSpec) -> list[SimAgent]:
        ...
