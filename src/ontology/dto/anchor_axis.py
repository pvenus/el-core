from dataclasses import dataclass
from .vector import Vector

@dataclass
class AnchorAxis:
    """
    '진짜 앵커 축'
    예: emotion 카테고리의 7대 감정 (joy/anger/...)
    """
    id: str
    label: str
    category_name: str
    embedding: Vector

