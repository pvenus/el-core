# sim_engine/dto/agent.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

Vector = List[float]
Vars = Dict[str, float]
Meta = Dict[str, Any]


@dataclass(frozen=True)
class AgentSpec:
    """
    불변 스펙(설정값, 정체성)

    - comfort_vec / comfort_radius: 불변 확정
    - vars: 스펙 레벨 스칼라(기본값/성향 파라미터/상한 하한 등 가능)
    - meta: 태그/설명/성향/설정 등
    """
    dim: int
    comfort_vec: Vector
    comfort_radius: float = 1.0
    vars: Vars = field(default_factory=dict)
    meta: Meta = field(default_factory=dict)

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("AgentSpec.dim must be > 0")
        if len(self.comfort_vec) != self.dim:
            raise ValueError(f"comfort_vec dim mismatch: {len(self.comfort_vec)} != {self.dim}")
        if self.comfort_radius < 0:
            raise ValueError("comfort_radius must be >= 0")


@dataclass(frozen=True)
class AgentState:
    """
    가변 상태 스냅샷(턴 진행에 따라 변경)

    - vars는 state에 포함(스냅샷 자립성 확보)
    """
    turn: int
    current_vec: Vector
    vars: Vars = field(default_factory=dict)

    def validate(self, spec: AgentSpec) -> None:
        if self.turn < 0:
            raise ValueError("AgentState.turn must be >= 0")
        if len(self.current_vec) != spec.dim:
            raise ValueError(f"current_vec dim mismatch: {len(self.current_vec)} != {spec.dim}")
