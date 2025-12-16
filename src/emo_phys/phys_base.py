# src/emo_phys/phys_base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PhysicsParams:
    """
    감정 벡터에 적용할 간단한 물리 파라미터들.

    - k_event: 이벤트 방향으로 끌어당기는 힘 비율
    - k_restore: baseline(원래 상태)으로 복원시키는 힘 비율
    - noise_std: 무작위 노이즈 세기
    """
    k_event: float = 0.5
    k_restore: float = 0.1
    noise_std: float = 0.01


class BasePhysics(ABC):
    """
    모든 감정 물리 모델이 따라야 할 공통 인터페이스.

    여기서는 "LLM/PCA가 뭔지"는 모른다.
    단지 D차원 벡터를 받아서 상태를 업데이트하는 규칙만 정의한다.
    """

    def __init__(self, baseline: np.ndarray):
        """
        Args:
            baseline: 되돌아가고 싶은 기준 상태 벡터
                     (예: 중립 감정 상태, 기본 상태)
        """
        self.baseline = np.asarray(baseline, dtype=np.float32)

    @abstractmethod
    def step(
        self,
        state: np.ndarray,
        event_vec: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        한 스텝 동안 상태를 업데이트한다.

        Args:
            state: 현재 상태 벡터 (shape: [D])
            event_vec: 선택한 이벤트/단어의 벡터 (shape: [D])
            rng: 랜덤 생성기 (재현성 필요하면 외부에서 넘겨주기)

        Returns:
            next_state: 다음 상태 벡터
        """
        ...

    def distance_to_baseline(self, state: np.ndarray) -> float:
        """
        현재 상태가 baseline과 얼마나 떨어져 있는지 (L2 거리) 계산.
        나중에 보상 계산 등에 사용할 수 있음.
        """
        state = np.asarray(state, dtype=np.float32)
        diff = state - self.baseline
        return float(np.linalg.norm(diff, ord=2))
