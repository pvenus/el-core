# src/emo_phys/phys_base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PhysicsParams:
    """
    k_event: action/event 벡터가 상태에 미치는 스케일(이동량 계수)
    noise_std: passive_step(시간경과)에서의 외란(노이즈)
    """
    k_event: float = 0.7
    noise_std: float = 0.01


@dataclass
class PhysicsContext:
    tick: int = 0


class BasePhysics(ABC):
    def __init__(self, baseline: np.ndarray):
        self.baseline = np.asarray(baseline, dtype=np.float32)

    def distance_to_baseline(self, state: np.ndarray) -> float:
        v = np.asarray(state, dtype=np.float32) - self.baseline
        return float(np.linalg.norm(v))

    def restore_vector(self, state: np.ndarray) -> np.ndarray:
        """목표 방향(인지용): baseline - state"""
        state = np.asarray(state, dtype=np.float32)
        return (self.baseline - state).astype(np.float32)

    @abstractmethod
    def apply_event(
        self,
        state: np.ndarray,
        event_vec: np.ndarray,
        ctx: Optional[PhysicsContext] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        'event_vec'는 이제 목표 위치가 아니라,
        상태에 더할 '이동/행동 벡터(action delta)'로 해석하는 것이 기본이다.
        """
        ...

    @abstractmethod
    def passive_step(
        self,
        state: np.ndarray,
        ctx: Optional[PhysicsContext] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        자연회복(베이스라인으로 당김) 없이,
        시간 경과/외란/필드 등 '수동적 변화'만 반영하는 step.
        """
        ...
