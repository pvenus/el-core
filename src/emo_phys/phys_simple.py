# src/emo_phys/phys_simple.py

from typing import Optional

import numpy as np

from .phys_base import BasePhysics, PhysicsParams


class SimplePhysics(BasePhysics):
    """
    이번 주 end-to-end 데모용 간단 물리 모델.

    - 이벤트 벡터 방향으로 끌어당기고
    - baseline 쪽으로 복원시키고
    - 약간의 노이즈를 섞어서 상태를 움직인다.
    """

    def __init__(
        self,
        baseline: np.ndarray,
        params: Optional[PhysicsParams] = None,
    ):
        super().__init__(baseline)
        self.params = params or PhysicsParams()

    def step(
        self,
        state: np.ndarray,
        event_vec: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        event_vec = np.asarray(event_vec, dtype=np.float32)

        if rng is None:
            rng = np.random.default_rng()

        # 1) 이벤트 쪽으로 끌어당기는 힘
        F_event = (event_vec - state) * self.params.k_event

        # 2) baseline(기준 상태) 쪽으로 복원시키는 힘
        F_restore = (self.baseline - state) * self.params.k_restore

        # 3) 약간의 랜덤 노이즈
        noise = rng.normal(
            loc=0.0,
            scale=self.params.noise_std,
            size=state.shape,
        ).astype(np.float32)

        # 다음 상태 = 현재 + 세 힘의 합
        next_state = state + F_event + F_restore + noise
        return next_state
