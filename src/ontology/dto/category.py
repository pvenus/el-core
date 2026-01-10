from dataclasses import dataclass, field
from typing import Dict
from .anchor_axis import AnchorAxis

@dataclass
class Category:
    """예: emotion, regulation (그냥 분류/도메인 라벨)"""
    name: str
    anchors: Dict[str, "AnchorAxis"] = field(default_factory=dict)  # anchor_id -> axis

    def add_anchor(self, axis: "AnchorAxis"):
        if axis.category_name != self.name:
            raise ValueError("AnchorAxis.category_name must match Category.name")
        self.anchors[axis.id] = axis
