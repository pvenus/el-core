from dataclasses import dataclass, field
from src.scenario.dto.choice_artifact import ChoiceArtifact
from typing import List

@dataclass
class RoundSpec:
    """A single round containing multiple agent reply choices."""

    round_id: int
    choices: List[ChoiceArtifact] = field(default_factory=list)

    def add_choice(self, choice: ChoiceArtifact) -> None:
        self.choices.append(choice)
