from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AgentSpec:
    """
    - comfort_vec: 중심(기본 위치)
    - radius: comfort 범위 거리
    """
    agent_id: str
    name: str
    comfort_vec: np.ndarray
    radius: float

    # vars, meta는 일단 무시(보류)
    vars: dict[str, float] | None = None
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "comfort_vec": self.comfort_vec.tolist(),
            "radius": float(self.radius),
        }

    @staticmethod
    def from_dict(d: dict[str, Any], *, dim: int) -> "AgentSpec":
        agent_id = str(d["agent_id"])
        name = str(d.get("name") or agent_id)

        comfort = np.array(d["comfort_vec"], dtype=float)
        if comfort.shape != (dim,):
            raise ValueError(f"AgentSpec.comfort_vec shape must be (dim,) got={comfort.shape}")

        radius = float(d.get("radius", 0.0))
        if radius < 0:
            raise ValueError("AgentSpec.radius must be >= 0")

        return AgentSpec(
            agent_id=agent_id,
            name=name,
            comfort_vec=comfort,
            radius=radius,
        )


@dataclass
class AgentState:
    """
    - current_vec: 현재 위치 벡터
    """
    current_vec: np.ndarray
    step_idx: int = 0

    # vars 보류
    vars: dict[str, float] | None = None
