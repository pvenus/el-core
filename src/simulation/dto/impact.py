from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _safe_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=float)
    return v / n


@dataclass
class Impact:
    """
    direction: 방향 벡터 (dim,)
    magnitude: 크기 (스칼라)
    duration: 남은 스텝 수 (>=1이면 적용 중, tick 후 0이 되면 만료 처리)
    delta_vars: vars에 더해지는 변화량
    profile: 메타데이터(디버그/태그/출처 등)
    """
    direction: np.ndarray
    magnitude: float
    duration: int = 1
    delta_vars: dict[str, float] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)

    def delta_vec(self) -> np.ndarray:
        return _safe_normalize(self.direction) * float(self.magnitude)

    def tick(self) -> None:
        self.duration -= 1

    @property
    def expired(self) -> bool:
        return self.duration <= 0

    @staticmethod
    def from_dict(d: dict) -> "Impact":
        """
        UI/외부 입력을 Impact로 변환.
        기대 형태:
          {
            "direction": [..] or np.ndarray,
            "magnitude": 0.5,
            "duration": 2,                 # optional
            "delta_vars": {"hp": -1},      # optional
            "profile": {...}              # optional
          }
        """
        if "direction" not in d or "magnitude" not in d:
            raise ValueError("Impact.from_dict requires 'direction' and 'magnitude'")

        direction = d["direction"]
        if isinstance(direction, np.ndarray):
            vec = direction.astype(float)
        else:
            vec = np.array(direction, dtype=float)

        return Impact(
            direction=vec,
            magnitude=float(d["magnitude"]),
            duration=int(d.get("duration", 1)),
            delta_vars={k: float(v) for k, v in (d.get("delta_vars") or {}).items()},
            profile=dict(d.get("profile") or {}),
        )

    def to_dict(self) -> dict:
        return {
            "direction": self.direction.tolist(),
            "magnitude": float(self.magnitude),
            "duration": int(self.duration),
            "delta_vars": dict(self.delta_vars),
            "profile": dict(self.profile),
        }
