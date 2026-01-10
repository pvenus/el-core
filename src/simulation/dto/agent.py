from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AgentSpec:
    """
    불변 스펙(외부에서 정의/주입되는 값)
    - vars: 초기 vars 템플릿
    """
    agent_id: str
    name: str
    base_vec: np.ndarray
    comfort_vec: np.ndarray
    vars: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """
    가변 상태(SSOT)
    """
    current_vec: np.ndarray
    vars: dict[str, float] = field(default_factory=dict)
    step_idx: int = 0
