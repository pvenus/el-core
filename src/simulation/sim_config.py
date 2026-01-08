# sim_engine/sim_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SimulationConfig:
    """
    시뮬레이션 전이 설정값.

    A안에서 의미 있는 것만 유지:
    - damping: 감쇠(0~1)
    - noise_std: 가우시안 노이즈 표준편차
    - normalize_vec: 적용 후 정규화 여부
    - clamp_norm: 벡터 노름 최대값 제한
    """
    damping: Optional[float] = None
    noise_std: float = 0.0
    normalize_vec: bool = False
    clamp_norm: Optional[float] = None
