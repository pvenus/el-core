# sim_engine/sim_dynamics.py
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

from .sim_vector import add
from .dto.agent import AgentSpec, AgentState
from .dto.impact import Impact
from .dto.turn import Transition

from .sim_types import Vector, Vars

def _sum_impacts(spec: AgentSpec, impacts: List[Impact]) -> Tuple[Vector, Vars]:
    """
    활성 impacts를 합산하여 (sum_delta_vec, sum_delta_vars)를 만든다.
    - clamp/normalize/damping/noise 등 "impact 외 규칙"은 적용하지 않음 (형 요구)
    """
    dim = spec.dim
    sum_vec = [0.0 for _ in range(dim)]
    sum_vars: Vars = {}

    for imp in impacts:
        imp.validate(dim)

        dv = imp.delta_vec()   # unit(direction) * magnitude
        sum_vec = add(sum_vec, dv)

        if imp.delta_vars:
            for k, v in imp.delta_vars.items():
                sum_vars[k] = float(sum_vars.get(k, 0.0)) + float(v)

    return sum_vec, sum_vars


def apply(spec: AgentSpec, prev: AgentState, impacts: List[Impact]) -> Transition:
    """
    dynamics = physics (순수 전이)

    - runner가 관리하는 impacts(활성 스택 스냅샷)를 받아
    - delta 합산 -> state 전이 적용
    - Transition 반환
    """
    prev.validate(spec)

    delta_vec, delta_vars = _sum_impacts(spec, impacts)

    if len(prev.current_vec) != len(delta_vec):
        raise ValueError(f"delta_vec dim mismatch: {len(prev.current_vec)} != {len(delta_vec)}")

    next_vec = add(prev.current_vec, delta_vec)

    next_vars = dict(prev.vars)
    for k, v in delta_vars.items():
        next_vars[k] = float(next_vars.get(k, 0.0)) + float(v)

    next_state = replace(prev, current_vec=next_vec, vars=next_vars)

    # metrics는 "순수 관측치"만 (원하면 이후 확장)
    metrics = {
        "n_impacts": len(impacts),
    }

    return Transition(
        next_state=next_state,
        sum_delta_vec=delta_vec,
        sum_delta_vars=delta_vars,
        metrics=metrics,
    )
