from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

@dataclass
class GridWorldConfig:
    size: int = 5
    max_steps: int = 60
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (4, 4)
    walls: Tuple[Tuple[int, int], ...] = ()

class GridWorld:
    def __init__(self, cfg: GridWorldConfig):
        self.cfg = cfg
        self.agent = (0, 0)
        self.steps = 0

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def obs_dim(self) -> int:
        n = self.cfg.size * self.cfg.size
        return n + n  # one-hot(agent) + one-hot(goal)

    def _one_hot(self, pos) -> np.ndarray:
        n = self.cfg.size * self.cfg.size
        idx = pos[0] * self.cfg.size + pos[1]
        v = np.zeros(n, dtype=np.float32)
        v[idx] = 1.0
        return v

    def _obs(self) -> np.ndarray:
        return np.concatenate([self._one_hot(self.agent), self._one_hot(self.cfg.goal)], axis=0)

    def reset(self) -> np.ndarray:
        self.agent = tuple(self.cfg.start)
        self.steps = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        ax, ay = self.agent
        nx, ny = ax, ay
        if action == 0: nx = max(0, ax - 1)
        elif action == 1: nx = min(self.cfg.size - 1, ax + 1)
        elif action == 2: ny = max(0, ay - 1)
        elif action == 3: ny = min(self.cfg.size - 1, ay + 1)

        if (nx, ny) in self.cfg.walls:
            nx, ny = ax, ay

        self.agent = (nx, ny)
        self.steps += 1

        done = False
        reward = -0.01
        if self.agent == self.cfg.goal:
            reward = 1.0
            done = True
        if self.steps >= self.cfg.max_steps:
            done = True

        return self._obs(), reward, done, {}

    def render_ascii(self) -> str:
        s = []
        for i in range(self.cfg.size):
            row = []
            for j in range(self.cfg.size):
                if (i, j) in self.cfg.walls: row.append("#")
                elif (i, j) == self.agent:   row.append("A")
                elif (i, j) == self.cfg.goal:row.append("G")
                else:                         row.append(".")
            s.append(" ".join(row))
        return "\n".join(s)
