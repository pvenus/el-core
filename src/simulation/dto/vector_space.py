from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AxisSpec:
    key: str
    tags: list[str]
    proto_vec: np.ndarray  # (dim,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "tags": list(self.tags),
            "proto_vec": self.proto_vec.tolist(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any], *, dim: int) -> "AxisSpec":
        key = str(d["key"])
        tags = list(d.get("tags") or [])
        proto = d.get("proto_vec")
        if proto is None:
            raise ValueError("AxisSpec.from_dict requires 'proto_vec'")
        proto_vec = np.array(proto, dtype=float)
        if proto_vec.shape != (dim,):
            raise ValueError(f"AxisSpec.proto_vec shape must be (dim,) got={proto_vec.shape}")
        return AxisSpec(key=key, tags=tags, proto_vec=proto_vec)


@dataclass(frozen=True)
class VectorSpaceSpec:
    dim: int
    axes: list[AxisSpec]

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("VectorSpaceSpec.dim must be > 0")
        if len(self.axes) != self.dim:
            raise ValueError(
                f"VectorSpaceSpec.axes length must equal dim. dim={self.dim}, axes={len(self.axes)}"
            )
        keys = [a.key for a in self.axes]
        if len(set(keys)) != len(keys):
            raise ValueError("AxisSpec.key must be unique")
        for a in self.axes:
            if not isinstance(a.proto_vec, np.ndarray):
                raise TypeError("AxisSpec.proto_vec must be np.ndarray")
            if a.proto_vec.shape != (self.dim,):
                raise ValueError(
                    f"AxisSpec.proto_vec must have shape (dim,). got={a.proto_vec.shape}"
                )

    def axis_keys(self) -> list[str]:
        return [a.key for a in self.axes]

    def validate_vec(self, vec: np.ndarray, name: str = "vec") -> None:
        if not isinstance(vec, np.ndarray):
            raise TypeError(f"{name} must be np.ndarray")
        if vec.shape != (self.dim,):
            raise ValueError(f"{name} must have shape (dim,) got={vec.shape}")

    def zeros(self) -> np.ndarray:
        return np.zeros((self.dim,), dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "VectorSpaceSpec",
            "version": 1,
            "dim": int(self.dim),
            "axes": [a.to_dict() for a in self.axes],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "VectorSpaceSpec":
        dim = int(d["dim"])
        axes_raw = d.get("axes")
        if not isinstance(axes_raw, list):
            raise ValueError("VectorSpaceSpec.from_dict requires 'axes' list")
        axes = [AxisSpec.from_dict(x, dim=dim) for x in axes_raw]
        return VectorSpaceSpec(dim=dim, axes=axes)
