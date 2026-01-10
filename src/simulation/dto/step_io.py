from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .impact import Impact


@dataclass
class StepInput:
    impacts_by_agent: dict[str, list[Impact]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "StepInput":
        """
        기대 형태:
          {
            "impacts_by_agent": {
              "agent_1": [ {ImpactDict}, {ImpactDict} ],
              "agent_2": [ ... ]
            },
            "metadata": {...}
          }
        """
        iba = d.get("impacts_by_agent") or {}
        parsed: dict[str, list[Impact]] = {}
        for aid, impacts in iba.items():
            parsed[aid] = [Impact.from_dict(x) if isinstance(x, dict) else x for x in impacts]
        return StepInput(
            impacts_by_agent=parsed,
            metadata=dict(d.get("metadata") or {}),
        )

    def to_dict(self) -> dict:
        return {
            "impacts_by_agent": {
                aid: [imp.to_dict() for imp in imps]
                for aid, imps in self.impacts_by_agent.items()
            },
            "metadata": dict(self.metadata),
        }


@dataclass
class AgentStepResult:
    agent_id: str
    before_vec: np.ndarray
    after_vec: np.ndarray
    delta_vec: np.ndarray

    before_vars: dict[str, float]
    after_vars: dict[str, float]
    delta_vars: dict[str, float]

    injected_impacts: list[Impact]
    expired_impacts: list[Impact]
    active_impacts_after: list[Impact]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    step_idx: int
    per_agent: list[AgentStepResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def by_agent_id(self) -> dict[str, AgentStepResult]:
        return {r.agent_id: r for r in self.per_agent}

    def to_dict(self) -> dict:
        return {
            "step_idx": int(self.step_idx),
            "metadata": dict(self.metadata),
            "per_agent": [
                {
                    "agent_id": r.agent_id,
                    "before_vec": r.before_vec.tolist(),
                    "after_vec": r.after_vec.tolist(),
                    "delta_vec": r.delta_vec.tolist(),
                    "before_vars": dict(r.before_vars),
                    "after_vars": dict(r.after_vars),
                    "delta_vars": dict(r.delta_vars),
                    "metrics": dict(r.metrics),
                    "injected_impacts": [i.to_dict() for i in r.injected_impacts],
                    "expired_impacts": [i.to_dict() for i in r.expired_impacts],
                    "active_impacts_after": [i.to_dict() for i in r.active_impacts_after],
                }
                for r in self.per_agent
            ],
        }
