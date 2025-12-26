# src/emo_phys/phys_simple.py

from __future__ import annotations

from typing import Optional
import numpy as np

from .phys_base import BasePhysics, PhysicsParams, PhysicsContext


class SimplePhysics(BasePhysics):
    """
    형 프로젝트 정의 반영 버전

    - apply_event: 선택한 벡터(=이동/행동 벡터)를 state에 더한다.
    - passive_step: 자연회복 없이 시간경과/외란만 적용한다.
    """

    def __init__(self, baseline: np.ndarray, params: Optional[PhysicsParams] = None):
        super().__init__(baseline)
        self.params = params or PhysicsParams()

    def apply_event(
        self,
        state: np.ndarray,
        event_vec: np.ndarray,
        ctx: Optional[PhysicsContext] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        action_vec = np.asarray(event_vec, dtype=np.float32)

        k = float(self.params.k_event)

        # ✅ 핵심 패치: 보간이 아니라 '이동 벡터 적용'
        next_state = state + k * action_vec
        return next_state

    def passive_step(
        self,
        state: np.ndarray,
        ctx: Optional[PhysicsContext] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if rng is None:
            rng = np.random.default_rng()

        noise_std = float(self.params.noise_std)
        noise = rng.normal(0.0, noise_std, size=state.shape).astype(np.float32)
        return state + noise
