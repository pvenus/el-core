# sim_engine/types.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence, TypeAlias

Vector: TypeAlias = List[float]
VecSeq: TypeAlias = Sequence[float]

Vars: TypeAlias = Dict[str, float]
Meta: TypeAlias = Dict[str, Any]
