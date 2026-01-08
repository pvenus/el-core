# sim_engine/dto/impact.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..sim_types import Vector, Meta

@dataclass(frozen=True)
class Impact:
    """
    외부에서 들어오는 '임팩트' 단위.
    - Event/Action 구분 없음 (A안 정책)
    - vec + scale + meta 만 유지
    """
    vec: Vector
    scale: float = 1.0
    meta: Meta = field(default_factory=dict)

    def scaled_vec(self) -> Vector:
        s = float(self.scale)
        return [x * s for x in self.vec]

    def get_vars_delta(self) -> Optional[Dict[str, float]]:
        """
        vars_delta는 meta로 전달할 수 있다. (선택)
        meta["vars_delta"] = {"energy": -1.0, ...}
        """
        v = self.meta.get("vars_delta")
        if isinstance(v, dict):
            out: Dict[str, float] = {}
            for k, val in v.items():
                try:
                    out[str(k)] = float(val)
                except Exception:
                    continue
            return out
        return None
