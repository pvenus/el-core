from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class SessionState:
    """Mutable session state so dialogue can continue regardless of which choice was taken."""

    round_id: int
    turn: int
    current_vec: List[float]
    comfort_vec: List[float]
    memory_tags: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

