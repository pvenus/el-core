# sim_engine/dto/turn.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agent import AgentState
from .impact import Impact

from ..sim_types import Meta, Vector, Vars

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
class Transition:
    """
    dynamics 결과 묶음 (runner가 TurnResult를 조립하기 위한 최소 재료)

    - next_state: 전이 결과 상태 (turn은 dynamics에서 증가시키지 않음)
    - sum_delta_vec: 활성 impacts 합산으로 만들어진 총 delta 벡터
    - sum_delta_vars: vars 변화량 합산
    - metrics: 전이 관련 순수 관측치(선택)
    """
    next_state: AgentState
    sum_delta_vec: Vector
    sum_delta_vars: Vars = field(default_factory=dict)
    metrics: Meta = field(default_factory=dict)

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

    impacts_new: List[Impact] = field(default_factory=list)  # 이번 턴에 새로 들어온 것
    impacts_applied: List[Impact] = field(default_factory=list)  # 이번 턴 적용된(활성 스택 전체) 스냅샷
    impacts_remaining: List[Impact] = field(default_factory=list)  # tick 후 남은 것

    applied_delta_vec: List[float] = field(default_factory=list)
    applied_delta_vars: Dict[str, float] = field(default_factory=dict)

    in_comfort: bool = False
    distance_to_comfort: float = 0.0

    metrics: Meta = field(default_factory=dict)
    meta: Meta = field(default_factory=dict)

