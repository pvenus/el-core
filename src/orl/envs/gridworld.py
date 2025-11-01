# src/orl/envs/gridworld.py
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class GridWorldConfig:
    size: int = 5
    max_steps: int = 60
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (4, 4)
    walls: Tuple[Tuple[int, int], ...] = ()
    terminate_on_goal: bool = True  # ← 목표 도달 시 조기 종료

class GridWorld:
    def __init__(self, cfg: GridWorldConfig):
        self.cfg = cfg
        self._N = cfg.size * cfg.size
        self.reset()

    @property
    def n_actions(self): return 4

    @property
    def obs_dim(self): return self._N + self._N  # agent one-hot + goal one-hot

    def _one_hot_pos(self, pos):
        v = np.zeros(self._N, dtype=np.float32)
        idx = pos[0] * self.cfg.size + pos[1]
        v[idx] = 1.0
        return v

    def _obs(self):
        return np.concatenate([self._one_hot_pos(self.agent),
                               self._one_hot_pos(self.cfg.goal)], axis=0)

    def reset(self):
        # ✅ 시작 위치를 설정 파일 값으로 정확히 적용
        self.agent = tuple(self.cfg.start)
        self.steps = 0
        return self._obs()

    def step(self, action: int):
        ax, ay = self.agent
        nx, ny = ax, ay
        if action == 0: nx = max(0, ax - 1)                 # up
        elif action == 1: nx = min(self.cfg.size - 1, ax + 1)  # down
        elif action == 2: ny = max(0, ay - 1)               # left
        elif action == 3: ny = min(self.cfg.size - 1, ay + 1)  # right

        # 벽이면 이동 취소
        if (nx, ny) in self.cfg.walls:
            nx, ny = ax, ay

        self.agent = (nx, ny)

        reward = -0.01
        done = False
        # ✅ 목표 처리 + 조기 종료
        if self.agent == self.cfg.goal:
            reward = 1.0
            if self.cfg.terminate_on_goal:
                done = True

        self.steps += 1
        if self.steps >= self.cfg.max_steps:
            done = True

        return self._obs(), reward, done, {}

    def render_ascii(self) -> str:
        """A=Agent, G=Goal, #=Wall, *=Agent&Goal same cell"""
        out = []
        for i in range(self.cfg.size):
            row = []
            for j in range(self.cfg.size):
                cell = "."
                if (i, j) in self.cfg.walls:
                    cell = "#"
                if (i, j) == self.cfg.goal:
                    cell = "G"
                if (i, j) == self.agent:
                    cell = "A" if cell != "G" else "*"  # 겹치면 *
                row.append(cell)
            out.append(" ".join(row))
        return "\n".join(out)
