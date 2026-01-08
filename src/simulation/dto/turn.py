# sim_engine/dto/turn.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agent_state import AgentState
from .impact import Impact

from ..sim_types import Meta

@dataclass(frozen=True)
class TurnInput:
    """
    엔진에 들어오는 입력.
    - impacts: 외부에서 준 임팩트(벡터) 리스트
    - meta: 디버그/출처 등 부가정보
    """
    impacts: List[Impact] = field(default_factory=list)
    meta: Meta = field(default_factory=dict)


@dataclass(frozen=True)
class TurnResult:
    """
    엔진이 내보내는 결과.
    - before/after: 상태 스냅샷
    - applied_impacts: 이번 턴에 실제 적용된 임팩트(스케일 반영 전 원본 그대로 저장)
    - applied_delta: 임팩트들을 합쳐 실제 적용된 총 델타 벡터
    - metrics: 거리/노름/클램프 등 관측치
    """
    turn: int
    before: AgentState
    after: AgentState

    applied_impacts: List[Impact] = field(default_factory=list)
    applied_delta: List[float] = field(default_factory=list)

    in_comfort: bool = False
    distance_to_comfort: float = 0.0

    metrics: Meta = field(default_factory=dict)
    meta: Meta = field(default_factory=dict)
