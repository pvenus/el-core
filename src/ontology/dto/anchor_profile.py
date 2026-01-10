from dataclasses import dataclass
from typing import List

@dataclass
class AnchorProfile:
    """
    Concept를 AnchorAxis 공간에서 표현한 좌표 벡터.
    (기계적으로는 anchor_scores의 정렬된 벡터 버전)
    """
    category_name: str
    anchor_ids: List[str]
    values: List[float]

    def as_dense(self) -> List[float]:
        return self.values