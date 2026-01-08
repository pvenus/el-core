from dataclasses import dataclass, field
from typing import List
from .round_spec import RoundSpec

@dataclass
class Scenario:
    """A full scenario: ordered rounds."""

    scenario_id: str
    title: str = ""
    dim: int = 6
    rounds: List[RoundSpec] = field(default_factory=list)

    def get_round(self, round_id: int) -> RoundSpec:
        for r in self.rounds:
            if r.round_id == int(round_id):
                return r
        raise KeyError(f"round_id not found: {round_id}")

    def add_round(self, round_spec: RoundSpec) -> None:
        self.rounds.append(round_spec)
        self.rounds.sort(key=lambda x: x.round_id)

