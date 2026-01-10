from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from src.simulation.dto.impact import Impact

@dataclass
class ChoiceArtifact:
    """Output of the maker pipeline (transport-friendly)."""

    choice_id: str
    round_id: int
    display_text: str
    embed_text: str
    concepts: List[Dict[str, Any]]
    embed_vec: List[float]
    # optional / derived (must come after non-default fields)
    ontology_best: Optional[Dict[str, Any]] = None
    impact: Optional[Impact] = None

    def to_choice_payload(self) -> Dict[str, Any]:
        """Return a payload compatible with Choice.create(...)

        NOTE: This returns only data. Actual object construction lives elsewhere.
        """
        return {
            "choice_id": self.choice_id,
            "display_text": self.display_text,
            "concepts": list(self.concepts),
            "ontology_best": (dict(self.ontology_best) if isinstance(self.ontology_best, dict) else None),
            "impact": (self.impact.to_dict() if isinstance(self.impact, Impact) else None),
            "embed_text": self.embed_text,
            "embed_vec": list(self.embed_vec),
        }