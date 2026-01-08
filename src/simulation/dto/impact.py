# sim_engine/dto/impact.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.helper.vector import normalize
from ..sim_types import Vector, Vars, Meta


@dataclass(frozen=True)
class Impact:
    """
    외부/내부 어디서든 생성될 수 있는 임팩트 단위 (A안 핵심 DTO)

    - direction: 방향 벡터 (내부에서 normalize될 수 있음)
    - magnitude: 크기 (스칼라)
    - duration: 작용 턴 수 (>=1). 매 턴 1씩 감소
    - delta_vars: energy/stamina 같은 state vars 변화량
    - profile: action_id 같은 프로필/태그/설명 등
    """
    direction: Vector
    magnitude: float = 1.0
    duration: int = 1

    delta_vars: Vars = field(default_factory=dict)
    profile: Meta = field(default_factory=dict)

    def validate(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if len(self.direction) != dim:
            raise ValueError(f"direction dim mismatch: {len(self.direction)} != {dim}")
        if self.duration <= 0:
            raise ValueError("duration must be >= 1")

    def dir_unit(self) -> Vector:
        """방향 벡터를 단위 벡터로 변환 (0벡터면 0벡터 반환)."""
        return normalize(self.direction)

    def delta_vec(self) -> Vector:
        """이번 턴에 적용될 벡터 델타 = unit(direction) * magnitude"""
        u = self.dir_unit()
        m = float(self.magnitude)
        return [x * m for x in u]
