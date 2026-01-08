from dataclasses import dataclass, field
from .choice import Choice
from typing import List

@dataclass
class RoundSpec:
    """A single round containing multiple agent reply choices."""

    round_id: int
    choices: List[Choice] = field(default_factory=list)

    def add_choice(self, choice: Choice) -> None:
        self.choices.append(choice)
