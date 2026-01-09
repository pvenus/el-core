# sim_engine/dto/vector_space.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..sim_types import Meta, Vector


@dataclass(frozen=True)
class AxisSpec:
    """
    axes=dim 1:1 강제.
    key는 고유키(예: Joy, Anger).
    prototype vec는 나중에 채울 수 있도록 Optional.
    """
    key: str
    tags: List[str] = field(default_factory=list)

    # prototype slot (merged)
    proto_vec: Optional[Vector] = None
    proto_meta: Meta = field(default_factory=dict)

    def validate(self, dim: int) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("AxisSpec.key must be a non-empty string")
        if self.proto_vec is not None and len(self.proto_vec) != dim:
            raise ValueError(f"AxisSpec.proto_vec dim mismatch: {len(self.proto_vec)} != {dim}")


@dataclass(frozen=True)
class VectorSpaceSpec:
    """
    감정/개념 등 어떤 벡터 공간이든 쓸 수 있는 통합 스펙.
    UI 라벨/태그 + 프로토타입(옵션)을 한 파일/객체로 관리한다.
    """
    space_id: str
    dim: int
    axes: List[AxisSpec]
    meta: Meta = field(default_factory=dict)

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("VectorSpaceSpec.dim must be > 0")
        if len(self.axes) != self.dim:
            raise ValueError(f"VectorSpaceSpec.axes must be 1:1 with dim (len={len(self.axes)} dim={self.dim})")

        keys = [a.key for a in self.axes]
        if len(set(keys)) != len(keys):
            dup = [k for k in set(keys) if keys.count(k) > 1]
            raise ValueError(f"AxisSpec.key must be unique. duplicated={dup}")

        for a in self.axes:
            a.validate(self.dim)

    @staticmethod
    def from_axis_keys(
        axis_keys: List[str],
        space_id: str = "ui_demo",
        meta: Optional[Meta] = None,
    ) -> "VectorSpaceSpec":
        """
        방식 1: UI 텍스트 줄 수 = dim
        - axis_keys 길이를 dim으로 사용
        - key 중복/공백 제거는 여기서 정리
        """
        cleaned: List[str] = []
        for k in axis_keys:
            kk = (k or "").strip()
            if kk:
                cleaned.append(kk)

        if not cleaned:
            raise ValueError("axis_keys is empty after cleaning")

        spec = VectorSpaceSpec(
            space_id=space_id,
            dim=len(cleaned),
            axes=[AxisSpec(key=k) for k in cleaned],
            meta=dict(meta or {}),
        )
        spec.validate()
        return spec

    @staticmethod
    def from_text_lines(
        text: str,
        space_id: str = "ui_demo",
        meta: Optional[Meta] = None,
    ) -> "VectorSpaceSpec":
        """
        Streamlit 텍스트 입력(줄바꿈) → axis_keys로 파싱
        """
        lines = [ln.strip() for ln in (text or "").splitlines()]
        return VectorSpaceSpec.from_axis_keys(lines, space_id=space_id, meta=meta)

    # ---- JSON helpers (외부 주입 대비) ----
    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "space_id": self.space_id,
            "dim": self.dim,
            "axes": [
                {
                    "key": a.key,
                    "tags": list(a.tags),
                    "proto_vec": list(a.proto_vec) if a.proto_vec is not None else None,
                    "proto_meta": dict(a.proto_meta),
                }
                for a in self.axes
            ],
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VectorSpaceSpec":
        axes_raw = d.get("axes") or []
        axes: List[AxisSpec] = []
        for a in axes_raw:
            axes.append(
                AxisSpec(
                    key=str(a.get("key", "")).strip(),
                    tags=list(a.get("tags") or []),
                    proto_vec=list(a["proto_vec"]) if a.get("proto_vec") is not None else None,
                    proto_meta=dict(a.get("proto_meta") or {}),
                )
            )

        spec = VectorSpaceSpec(
            space_id=str(d.get("space_id", "")).strip() or "space",
            dim=int(d.get("dim", len(axes))),
            axes=axes,
            meta=dict(d.get("meta") or {}),
        )
        spec.validate()
        return spec
