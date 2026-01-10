from dataclasses import dataclass, field
from typing import List, Dict
from .vector import Vector

@dataclass
class Concept:
    """
    하위 감정/단어 노드.
    - embedding: 원본 임베딩
    - anchor_scores: '앵커 축 좌표' (핵심!)
    """
    id: str
    label: str
    category_name: str
    embedding: Vector

    anchor_scores: Dict[str, float] = field(default_factory=dict)  # anchor_id -> similarity
    relations: List["Relation"] = field(default_factory=list)
