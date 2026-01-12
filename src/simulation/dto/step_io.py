from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.simulation.dto.impact import Impact


@dataclass
class StepInput:
    """
    단일 에이전트 타겟 입력
    """
    agent_id: str
    impacts: list[Impact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "StepInput":
        """
        기대 형태:
        {
          "agent_id": "agent_01",
          "impacts": [ {ImpactDict}, ... ],
          "metadata": {...}
        }
        """
        agent_id = str(d["agent_id"])
        impacts_raw = d.get("impacts") or []
        impacts = [Impact.from_dict(x) if isinstance(x, dict) else x for x in impacts_raw]
        return StepInput(
            agent_id=agent_id,
            impacts=impacts,
            metadata=dict(d.get("metadata") or {}),
        )

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "impacts": [i.to_dict() for i in self.impacts],
            "metadata": dict(self.metadata),
        }


@dataclass
class StepResult:
    """
    단일 에이전트 1-step 결과
    (기존 AgentStepResult 제거)
    """
    step_idx: int
    agent_id: str

    before_vec: np.ndarray
    after_vec: np.ndarray
    delta_vec: np.ndarray

    injected_impacts: list[Impact]
    expired_impacts: list[Impact]
    active_impacts_after: list[Impact]

    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step_idx": int(self.step_idx),
            "agent_id": self.agent_id,
            "before_vec": self.before_vec.tolist(),
            "after_vec": self.after_vec.tolist(),
            "delta_vec": self.delta_vec.tolist(),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
            "injected_impacts": [i.to_dict() for i in self.injected_impacts],
            "expired_impacts": [i.to_dict() for i in self.expired_impacts],
            "active_impacts_after": [i.to_dict() for i in self.active_impacts_after],
        }
