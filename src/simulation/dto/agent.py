from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AgentSpec:
    """
    불변 스펙(SSOT, 외부에서 정의/주입되는 값)
    """
    agent_id: str
    name: str
    base_vec: np.ndarray
    comfort_vec: np.ndarray
    vars_base: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """
    가변 상태(SSOT, 재현/직렬화/테스트 기준)
    - "시뮬레이션 의미에 영향 주는 값"만 둔다.
    """
    current_vec: np.ndarray
    vars: dict[str, float] = field(default_factory=dict)
    step_idx: int = 0
