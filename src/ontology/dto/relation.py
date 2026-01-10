from dataclasses import dataclass

@dataclass
class Relation:
    source_id: str
    target_id: str
    type: str           # SIMILAR / TRANSITION / OPPOSE / REGULATE / CO_OCCUR ...
    weight: float
    evidence: str       # why

