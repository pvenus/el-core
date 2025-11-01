# src/orl/agents/dqn.py
from dataclasses import dataclass
from typing import Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ["DQNConfig", "DQNAgent"]

# -------------------------
# Config
# -------------------------
@dataclass
class DQNConfig:
    # (필수) 외부에서 셋업
    obs_dim: int = 0
    n_actions: int = 0

    # 네트워크/학습 하이퍼
    hidden: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    grad_clip: float = 10.0
    double_dqn: bool = True
    target_update_interval: int = 500  # gradient steps 기준

    # 실행 환경
    device: str = "cpu"
    seed: int = 7

    # ---- 아래는 trainer/replay/eps 관련 값들이 algo 섹션에 섞여 들어와도 에러 안 나도록 수용 ----
    # trainer-ish
    episodes: int = 0  # 에이전트 내부에서는 사용하지 않음(보관용)
    # replay-ish
    batch_size: int = 64
    buffer_size: int = 50_000
    start_learning: int = 1_000
    train_interval: int = 1
    # epsilon-ish
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 15_000

# -------------------------
# Q-Network
# -------------------------
class QNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -------------------------
# DQN Agent
# -------------------------
class DQNAgent:
    """
    - epsilon-greedy는 외부에서 eps 값을 넘겨주고 act()에서 사용
    - replay buffer는 외부(utils.replay.ReplayBuffer)가 제공 (sample -> (s,a,r,ns,d))
    - update()가 1 gradient step을 수행하고 loss 반환
    """
    def __init__(self, cfg: DQNConfig):
        assert cfg.obs_dim > 0 and cfg.n_actions > 0, "obs_dim / n_actions must be set"
        self.cfg = cfg

        # Seeds
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.device = torch.device(cfg.device)

        # Nets
        self.q = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(self.device)
        self.target = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()

        # Optim/Loss
        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.criterion = nn.SmoothL1Loss()

        self.n_actions = cfg.n_actions
        self.grad_steps = 0
        self.total_steps = 0
        self.last_target_update_step = 0
        self.target.load_state_dict(self.q.state_dict())  # 초기 하드 동기화

    # ---------- Acting ----------
    def act(self, obs_np: np.ndarray, eps: float) -> int:
        """epsilon-greedy action"""
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
            q = self.q(x)
            return int(q.argmax(dim=-1).item())

    # ---------- Learning ----------
    def update(self, batch) -> float:
        """
        batch: Transition(s, a, r, ns, d)
          - s, ns: torch.float [B, obs_dim]
          - a: torch.long [B, 1]
          - r, d: torch.float [B, 1]
        """
        self.q.train()

        s = batch.s.to(self.device)
        a = batch.a.to(self.device)
        r = batch.r.to(self.device)
        ns = batch.ns.to(self.device)
        d = batch.d.to(self.device)

        # Q(s,a)
        q_pred = self.q(s).gather(1, a)

        with torch.no_grad():
            if self.cfg.double_dqn:
                # action selection by online net
                next_actions = self.q(ns).argmax(dim=1, keepdim=True)
                # evaluation by target net
                q_next = self.target(ns).gather(1, next_actions)
            else:
                q_next = self.target(ns).max(dim=1, keepdim=True)[0]

            target = r + (1.0 - d) * self.cfg.gamma * q_next

        loss = self.criterion(q_pred, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.optim.step()

        self.grad_steps += 1
        return float(loss.item())

    def hard_update_target(self) -> None:
        """Q 네트워크 파라미터를 타깃 네트워크에 즉시 복사."""
        self.target.load_state_dict(self.q.state_dict())
        self.last_target_update_step = self.total_steps

    def maybe_update_target(self, force: bool = False) -> None:
        """조건 충족 시(또는 강제) 타깃 네트워크 동기화."""
        if force or (self.total_steps - self.last_target_update_step) >= self.cfg.target_update_interval:
            self.hard_update_target()


