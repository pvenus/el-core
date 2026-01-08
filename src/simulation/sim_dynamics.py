# sim_engine/sim_dynamics.py
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional
import random

from .sim_config import SimulationConfig
from .dto import AgentState

from src.helper.vector import add, scale, l2_distance, normalize, clamp_norm

from .sim_types import Vector, Vars

def _apply_vars_delta(prev_vars: Vars, delta_vars: Optional[Vars], cfg: SimulationConfig) -> Vars:
    if not delta_vars:
        return dict(prev_vars)

    out = dict(prev_vars)
    for k, dv in delta_vars.items():
        out[k] = float(out.get(k, 0.0)) + float(dv)
    return out


def dynamics_step(
    prev: AgentState,
    delta_vec: Vector,
    cfg: SimulationConfig,
    rng: Optional[random.Random] = None,
    delta_vars: Optional[Vars] = None,
) -> AgentState:
    """
    순수 전이 커널.
    - 입력: 이전 상태 + (이번 턴에 적용할) 총 델타 벡터
    - 출력: 다음 상태
    """
    if rng is None:
        rng = random.Random()

    if len(prev.current_vec) != len(delta_vec):
        raise ValueError(f"delta_vec dim mismatch: {len(prev.current_vec)} != {len(delta_vec)}")

    # 1) delta 적용
    next_vec = add(prev.current_vec, delta_vec)

    # 2) damping (0~1). 1이면 유지, 0이면 완전 소멸 (현재 구현은 "감쇠 적용" 스타일)
    if cfg.damping is not None:
        d = float(cfg.damping)
        next_vec = scale(next_vec, max(0.0, min(1.0, d)))

    # 3) noise (옵션)
    if cfg.noise_std and cfg.noise_std > 0:
        std = float(cfg.noise_std)
        noise = [(rng.gauss(0.0, std)) for _ in next_vec]
        next_vec = add(next_vec, noise)

    # 4) normalize (옵션)
    if cfg.normalize_vec:
        next_vec = normalize(next_vec)

    # 5) clamp norm (옵션)
    if cfg.clamp_norm is not None:
        next_vec = clamp_norm(next_vec, float(cfg.clamp_norm))

    # 6) vars update (옵션)
    next_vars = _apply_vars_delta(prev.vars or {}, delta_vars, cfg)

    return replace(prev, current_vec=next_vec, vars=next_vars)


def distance_to_comfort(state: AgentState) -> float:
    return l2_distance(state.current_vec, state.comfort_vec)
