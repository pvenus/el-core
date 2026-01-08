from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass
class SelectionContext:
    """Inputs for selecting a choice."""

    turn: int
    current_vec: Sequence[float]
    comfort_vec: Sequence[float]
    allowed_tags: Optional[Sequence[str]] = None

