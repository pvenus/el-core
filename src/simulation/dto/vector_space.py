from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AxisSpec:
    key: str
    tags: list[str]
    proto_vec: np.ndarray  # (dim,)


@dataclass(frozen=True)
class VectorSpaceSpec:
    dim: int
    axes: list[AxisSpec]

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("VectorSpaceSpec.dim must be > 0")
        if len(self.axes) != self.dim:
            raise ValueError(
                f"VectorSpaceSpec.axes length must equal dim. "
                f"dim={self.dim}, axes={len(self.axes)}"
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
