# sim_engine/sim_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol
import random

from .sim_agent import SimAgent
from .sim_config import SimulationConfig
from .sim_dynamics import dynamics_step, distance_to_comfort
from .dto import Impact, TurnInput, TurnResult, AgentState


class TurnHook(Protocol):
    """
    외부와의 커뮤니케이션 포인트(선택).
    DTO에 ports를 두지 않기 위해 runner에 둔다.
    """
    def on_turn(self, result: TurnResult) -> None: ...


def _sum_impacts(impacts: List[Impact]) -> List[float]:
    if not impacts:
        return []
    dim = len(impacts[0].vec)
    out = [0.0 for _ in range(dim)]
    for imp in impacts:
        if len(imp.vec) != dim:
            raise ValueError(f"Impact dim mismatch: {len(imp.vec)} != {dim}")
        v = imp.scaled_vec()
        out = [a + b for a, b in zip(out, v)]
    return out


def _merge_vars_delta(impacts: List[Impact]) -> Optional[Dict[str, float]]:
    merged: Dict[str, float] = {}
    has_any = False
    for imp in impacts:
        dv = imp.get_vars_delta()
        if not dv:
            continue
        has_any = True
        for k, v in dv.items():
            merged[k] = float(merged.get(k, 0.0)) + float(v)
    return merged if has_any else None


@dataclass
class TurnSimulation:
    """
    A안 Turn runner.
    - 외부에서 impacts(벡터)만 주면 상태를 업데이트하고 TurnResult를 반환한다.
    """
    agent: SimAgent
    cfg: SimulationConfig
    seed: Optional[int] = None
    hook: Optional[TurnHook] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._turn = 0

    @property
    def turn(self) -> int:
        return self._turn

    def step(self, turn_input: TurnInput) -> TurnResult:
        before_state: AgentState = self.agent.state
        impacts = list(turn_input.impacts or [])

        # delta 합산
        applied_delta = _sum_impacts(impacts)
        if not applied_delta:
            # impacts가 없으면 "0 델타"로 취급
            applied_delta = [0.0 for _ in before_state.current_vec]

        # vars_delta 합산(옵션)
        vars_delta = _merge_vars_delta(impacts)

        # dynamics
        after_state = dynamics_step(
            prev=before_state,
            delta_vec=applied_delta,
            cfg=self.cfg,
            rng=self._rng,
            delta_vars=vars_delta,
        )

        # update agent
        self.agent.state = after_state

        dist = distance_to_comfort(after_state)
        in_comfort = dist <= float(after_state.comfort_radius)

        result = TurnResult(
            turn=self._turn,
            before=before_state,
            after=after_state,
            applied_impacts=impacts,
            applied_delta=applied_delta,
            in_comfort=in_comfort,
            distance_to_comfort=float(dist),
            metrics={
                "delta_norm": float(sum(x * x for x in applied_delta) ** 0.5),
                "vars_delta": dict(vars_delta) if vars_delta else {},
            },
            meta=dict(turn_input.meta) if turn_input.meta else {},
        )

        self._turn += 1

        if self.hook is not None:
            self.hook.on_turn(result)

        return result

    def run_n(self, inputs: List[TurnInput]) -> List[TurnResult]:
        out: List[TurnResult] = []
        for ti in inputs:
            out.append(self.step(ti))
        return out
