from __future__ import annotations

from typing import Protocol, runtime_checkable

from .dto.vector_space import VectorSpaceSpec
from .dto.agent import AgentSpec


@runtime_checkable
class SpaceSource(Protocol):
    """
    외부에서 관리되는 VectorSpaceSpec을 로드/제공하는 인터페이스
    (현 단계에서는 필수 적용 대상은 아니지만, 확장 고려해서 유지)
    """
    def load_space(self) -> VectorSpaceSpec:
        ...


@runtime_checkable
class AgentSpecSource(Protocol):
    """
    외부에서 '완성된 AgentSpec 목록'을 제공하는 인터페이스.
    - Manager는 이를 받아 내부에서 SimAgent(spec,state)를 생성/관리한다.
    - InitialStateSource는 보류.
    """
    def build_agent_specs(self, space: VectorSpaceSpec) -> list[AgentSpec]:
        ...
