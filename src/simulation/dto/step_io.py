from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .impact import Impact


@dataclass
class StepInput:
    """
    한 스텝에 필요한 입력 묶음
    """
    impacts_by_agent: dict[str, list[Impact]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStepResult:
    agent_id: str
    before_vec: np.ndarray
    after_vec: np.ndarray
    before_vars: dict[str, float]
    after_vars: dict[str, float]
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
