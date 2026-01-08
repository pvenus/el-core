# sim_engine/sim_builder.py
from __future__ import annotations

from typing import Dict, List, Optional, Any

from .sim_agent import SimAgent
from .dto.impact import Impact

from .sim_types import Vector, Vars, Meta


def make_demo_agent(
    dim: int = 8,
    comfort_radius: float = 1.0,
    spec_vars: Optional[Vars] = None,
    spec_meta: Optional[Meta] = None,
    init_vec: Optional[Vector] = None,
    init_vars: Optional[Vars] = None,
) -> SimAgent:
    """
    데모/테스트용 에이전트 생성(조립 함수).
    내부적으로는 SimAgent.create()를 사용한다.
    """
    return SimAgent.create(
        dim=dim,
        comfort_radius=comfort_radius,
        spec_vars=spec_vars,
        spec_meta=spec_meta,
        init_vec=init_vec,
        init_vars=init_vars,
    )


def make_demo_impacts(
    dim: int = 8,
    preset: str = "burst",
) -> List[Impact]:
    """
    데모/테스트용 Impact 프리셋 생성.

    preset:
      - "burst": 1턴 강한 임팩트 1개
      - "buff":  3턴 지속 임팩트 1개
      - "dot":   5턴 지속(약하게) 임팩트 1개
      - "mix":   burst + buff + dot 조합
    """
    def dir_at(i: int) -> Vector:
        v = [0.0] * dim
        if dim > 0:
            v[i % dim] = 1.0
        return v

    if preset == "burst":
        return [
            Impact(
                direction=dir_at(0),
                magnitude=1.0,
                duration=1,
                delta_vars={"energy": -1.0},
                profile={"action_id": "burst_1", "tag": "demo"},
            )
        ]

    if preset == "buff":
        return [
            Impact(
                direction=dir_at(1),
                magnitude=0.35,
                duration=3,
                delta_vars={"stamina": -0.2},
                profile={"action_id": "buff_3t", "tag": "demo"},
            )
        ]

    if preset == "dot":
        return [
            Impact(
                direction=dir_at(2),
                magnitude=0.15,
                duration=5,
                delta_vars={"energy": -0.1},
                profile={"action_id": "dot_5t", "tag": "demo"},
            )
        ]

    if preset == "mix":
        return (
            make_demo_impacts(dim=dim, preset="burst")
            + make_demo_impacts(dim=dim, preset="buff")
            + make_demo_impacts(dim=dim, preset="dot")
        )

    raise ValueError(f"Unknown preset: {preset}")
