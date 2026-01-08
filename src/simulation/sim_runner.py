# sim_engine/sim_runner.py
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

from .sim_agent import SimAgent
from .sim_dynamics import apply as dynamics_apply
from .dto.impact import Impact
from .dto.turn import TurnInput, TurnResult

from .sim_types import Meta

class SimRunner:
    """
    runner = 오케스트레이션
    - turn 관리
    - new impacts 추가
    - dynamics 호출
    - TurnResult 조립
    - agent.state 업데이트
    - impacts tick/expire 처리(스택 owner는 agent라 agent가 수행)
    """

    def __init__(self, agent: SimAgent):
        self.agent = agent

    def step_with_input(self, turn_input: TurnInput) -> TurnResult:
        return self.step(impacts_new=turn_input.impacts, meta=turn_input.meta)

    def step(self, impacts_new: Optional[List[Impact]] = None, meta: Optional[Meta] = None) -> TurnResult:
        meta = dict(meta or {})

        before = self.agent.state
        turn = before.turn

        impacts_new = list(impacts_new or [])
        if impacts_new:
            self.agent.add_impacts(impacts_new)

        impacts_applied = self.agent.snapshot_active_impacts()

        transition = dynamics_apply(self.agent.spec, before, impacts_applied)

        # turn 증가는 runner에서 책임
        after = replace(transition.next_state, turn=turn + 1)

        # agent 업데이트
        self.agent.set_state(after)

        # stack tick/expire
        self.agent.tick_impacts()
        impacts_remaining = self.agent.snapshot_active_impacts()

        dist = self.agent.distance_to_comfort(after)
        in_c = self.agent.in_comfort(after)

        return TurnResult(
            turn=turn,
            before=before,
            after=after,
            impacts_new=impacts_new,
            impacts_applied=impacts_applied,
            impacts_remaining=impacts_remaining,
            applied_delta_vec=transition.sum_delta_vec,
            applied_delta_vars=dict(transition.sum_delta_vars),
            distance_to_comfort=float(dist),
            in_comfort=bool(in_c),
            metrics=dict(transition.metrics),
            meta=meta,
        )
