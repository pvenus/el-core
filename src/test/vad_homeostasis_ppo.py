

# vad_homeostasis_ppo.py
# pip install gymnasium==0.29.1 stable-baselines3==2.3.2 torch numpy

import argparse
import gymnasium as gym
import numpy as np
from math import sqrt
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

# ===== Emotion Homeostasis Env =====
class EmotionHomeostasisEnv(gym.Env):
    """
    관찰: [vad(3), setpoint(3), event(3)]
    행동: force vector in R^3 (연속, -1~1)
    동역학: e_{t+1} = e_t + k_event*event + k_action*action - k_damp*(e_t - setpoint)
    보상: -||e - setpoint||^2 - lam_action||action||^2 - spike_penalty
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_len: int = 64,
        k_event: float = 0.50,
        k_action: float = 0.60,
        k_damp: float = 0.20,
        lam_action: float = 0.02,
        spike_thresh: float = 0.50,
        spike_penalty: float = 0.20,
        event_scale: float = 0.80,
        event_change_p: float = 0.15,
        seed: int = 42,
        setpoint_mode: str = "random",
        setpoint_fixed: Optional[Tuple[float, float, float]] = None,
        setpoint_scale: float = 0.2,
        setpoint_clip: float = 0.6,
        # new parameters for action/event process
        action_mode: str = "vec3", min_force: float = 0.0, max_force: float = 1.0, pos_alpha: float = 0.5,
        event_process: str = "mixed", event_ou_theta: float = 0.10, event_ou_sigma: float = 0.05,
    ):
        super().__init__()
        self.episode_len = episode_len
        self.k_event = k_event
        self.k_action = k_action
        self.k_damp = k_damp
        self.lam_action = lam_action
        self.spike_thresh = spike_thresh
        self.spike_penalty = spike_penalty
        self.event_scale = event_scale
        self.event_change_p = event_change_p

        self.rng = np.random.default_rng(seed)

        self.setpoint_mode = setpoint_mode
        self.setpoint_fixed = setpoint_fixed
        self.setpoint_scale = setpoint_scale
        self.setpoint_clip = setpoint_clip

        # action & position modulation
        self.action_mode = action_mode  # "vec3", "dir_mag", "dir_mag_pos"
        self.min_force = float(min_force)
        self.max_force = float(max_force)
        self.pos_alpha = float(pos_alpha)

        # event process (external force drift)
        self.event_process = event_process  # "jump", "ou", "mixed"
        self.event_ou_theta = float(event_ou_theta)
        self.event_ou_sigma = float(event_ou_sigma)
        self.event_mu = np.zeros(3, dtype=np.float32)

        # VAD, setpoint, event ∈ [-1, 1]^3
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        if self.action_mode == "vec3":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        elif self.action_mode == "dir_mag":
            # dir(3) + mag(1)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        elif self.action_mode == "dir_mag_pos":
            # dir(3) + mag(1) + pos(3)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        # state
        self.vad = np.zeros(3, dtype=np.float32)
        self.setpoint = np.zeros(3, dtype=np.float32)
        self.event = np.zeros(3, dtype=np.float32)
        self.t = 0
        self.prev_vad = self.vad.copy()

    def _sample_scenario(self):
        # setpoint sampling by mode
        if self.setpoint_mode == "fixed" and self.setpoint_fixed is not None:
            sp = np.asarray(self.setpoint_fixed, dtype=np.float32)
        else:  # random/default
            sp = self.rng.normal(loc=0.0, scale=self.setpoint_scale, size=3).astype(np.float32)
        self.setpoint = np.clip(sp, -self.setpoint_clip, self.setpoint_clip)
        # 시작 감정은 setpoint 주변에서 작은 잡음
        self.vad = np.clip(self.setpoint + self.rng.normal(0, 0.1, 3), -1, 1).astype(np.float32)
        # 초기 사건(자극)
        self.event = (self.rng.uniform(-1, 1, 3) * self.event_scale).astype(np.float32)

    def _maybe_change_event(self):
        # continuous OU-drift component
        if self.event_process in ("ou", "mixed"):
            theta = self.event_ou_theta
            sigma = self.event_ou_sigma
            drift = theta * (self.event_mu - self.event)
            noise = sigma * self.rng.normal(0.0, 1.0, 3)
            self.event = np.clip(self.event + drift + noise, -1.0, 1.0)
        # occasional jump component
        if self.event_process in ("jump", "mixed") and (self.rng.random() < self.event_change_p):
            target = (self.rng.uniform(-1, 1, 3) * self.event_scale).astype(np.float32)
            self.event = np.clip(0.85 * self.event + 0.15 * target, -1.0, 1.0)

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.vad, self.setpoint, self.event]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # optional external setpoint override via reset(options={"setpoint": np.array([...])})
        external_sp = None
        if options is not None and isinstance(options, dict) and "setpoint" in options:
            external_sp = np.asarray(options["setpoint"], dtype=np.float32)
            if external_sp.shape == (3,):
                self.setpoint = np.clip(external_sp, -self.setpoint_clip, self.setpoint_clip)
        if external_sp is None:
            self._sample_scenario()
        else:
            # when externally set, start near that setpoint
            self.vad = np.clip(self.setpoint + self.rng.normal(0, 0.1, 3), -1, 1).astype(np.float32)
            self.event = (self.rng.uniform(-1, 1, 3) * self.event_scale).astype(np.float32)
        self.t = 0
        self.prev_vad = self.vad.copy()
        return self._obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # decode action by mode
        if self.action_mode == "vec3":
            force = action.astype(np.float32)
            pos_gain = np.ones(3, dtype=np.float32)
        elif self.action_mode == "dir_mag":
            dir_raw = action[:3].astype(np.float32)
            mag_raw = float(action[3])
            # unit direction (fallback to small eps to avoid NaN)
            dir_norm = float(np.linalg.norm(dir_raw))
            dir_unit = dir_raw / (dir_norm + 1e-8)
            # map mag_raw∈[-1,1] → [min_force, max_force]
            mag = ( (mag_raw + 1.0) * 0.5 ) * (self.max_force - self.min_force) + self.min_force
            force = dir_unit * mag
            pos_gain = np.ones(3, dtype=np.float32)
        elif self.action_mode == "dir_mag_pos":
            dir_raw = action[:3].astype(np.float32)
            mag_raw = float(action[3])
            pos_raw = action[4:7].astype(np.float32)
            dir_norm = float(np.linalg.norm(dir_raw))
            dir_unit = dir_raw / (dir_norm + 1e-8)
            mag = ( (mag_raw + 1.0) * 0.5 ) * (self.max_force - self.min_force) + self.min_force
            # position gain per-axis: 1 + alpha * pos
            pos_unit = np.clip(pos_raw, -1.0, 1.0)
            pos_gain = 1.0 + self.pos_alpha * pos_unit
            force = dir_unit * mag
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        # effective force after position modulation (per-axis scaling)
        effective_force = (force * pos_gain).astype(np.float32)

        self._maybe_change_event()

        # Emotion dynamics
        # damping: 현재 감정이 setpoint로 살짝 되돌아가는 자연 관성
        dv = (self.k_event * self.event) + (self.k_action * effective_force) - (self.k_damp * (self.vad - self.setpoint))
        next_vad = np.clip(self.vad + dv, -1.0, 1.0)

        # reward: setpoint에 가까울수록 좋고, 과격한 행동엔 비용
        dist = np.linalg.norm(next_vad - self.setpoint)
        action_cost = self.lam_action * float(np.linalg.norm(effective_force))
        # 갑작스러운 감정 급변 패널티(스파이크 제어)
        jump = np.linalg.norm(next_vad - self.prev_vad)
        spike_cost = self.spike_penalty if jump > self.spike_thresh else 0.0

        reward = -(dist**2) - action_cost - spike_cost
        # alignment bonus: encourage force aligned with -(vad - setpoint)
        align = -(self.vad - self.setpoint)
        align_bonus = 0.02 * float(np.dot(align, effective_force) / (np.linalg.norm(align) + 1e-8))
        reward += align_bonus

        # update
        self.prev_vad = self.vad
        self.vad = next_vad
        self.t += 1

        terminated = False
        truncated = (self.t >= self.episode_len)

        info = {
            "dist": dist,
            "action_cost": action_cost,
            "jump": jump,
            "spike_cost": spike_cost,
            "vad": self.vad.copy(),
            "setpoint": self.setpoint.copy(),
            "event": self.event.copy(),
            "raw_action": action.copy(),
            "force": force.copy(),
            "effective_force": effective_force.copy(),
        }
        return self._obs(), float(reward), terminated, truncated, info


# ======= Train & Test with PPO (SB3) =======
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps for PPO")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Number of evaluation episodes after training")
    parser.add_argument("--eval-steps", type=int, default=64, help="Max steps per evaluation episode")
    parser.add_argument("--eval-deterministic", action="store_true", help="Use deterministic policy during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--sp-mode", choices=["random", "fixed"], default="random", help="Setpoint sampling mode")
    parser.add_argument("--sp-fixed", type=float, nargs=3, metavar=("V","A","D"), help="Fixed setpoint triple when --sp-mode fixed")
    parser.add_argument("--sp-scale", type=float, default=0.2, help="Stddev for random setpoint sampling")
    parser.add_argument("--sp-clip", type=float, default=0.6, help="Clip range for setpoint |value| <= sp_clip")
    parser.add_argument("--eval-setpoint", type=float, nargs=3, metavar=("V","A","D"), help="Override evaluation setpoint triple")
    parser.add_argument("--action-mode", choices=["vec3", "dir_mag", "dir_mag_pos"], default="vec3", help="Action parameterization")
    parser.add_argument("--min-force", type=float, default=0.05, help="Minimum force magnitude for dir_mag* modes")
    parser.add_argument("--max-force", type=float, default=1.2, help="Maximum force magnitude for dir_mag* modes")
    parser.add_argument("--pos-alpha", type=float, default=0.5, help="Per-axis gain strength for position modulation")
    parser.add_argument("--event-process", choices=["jump", "ou", "mixed"], default="ou", help="External event dynamics")
    parser.add_argument("--event-ou-theta", type=float, default=0.10, help="OU mean-reversion strength for event")
    parser.add_argument("--event-ou-sigma", type=float, default=0.03, help="OU noise scale for event")
    args = parser.parse_args()

    def make_env():
        return Monitor(EmotionHomeostasisEnv(
            episode_len=64,
            k_event=0.5, k_action=0.8, k_damp=0.3,
            lam_action=0.01, spike_thresh=0.5, spike_penalty=0.2,
            event_scale=0.8, event_change_p=0.05, seed=args.seed,
            setpoint_mode=args.sp_mode,
            setpoint_fixed=tuple(args.sp_fixed) if args.sp_fixed is not None else None,
            setpoint_scale=args.sp_scale,
            setpoint_clip=args.sp_clip,
            action_mode=args.action_mode,
            min_force=args.min_force,
            max_force=args.max_force,
            pos_alpha=args.pos_alpha,
            event_process=args.event_process,
            event_ou_theta=args.event_ou_theta,
            event_ou_sigma=args.event_ou_sigma
        ))

    env = DummyVecEnv([make_env])

    policy_kwargs = dict(net_arch=[128, 128])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.97,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.005,     # 감정 다양성 유지 vs 수렴의 균형
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    model.learn(total_timesteps=args.timesteps)
    model.save("ppo_vad_homeostasis.pt")

    # ---- evaluation ----
    total_rewards = []
    for ep in range(args.eval_episodes):
        test_env = EmotionHomeostasisEnv(
            seed=123 + ep,
            setpoint_mode=args.sp_mode,
            setpoint_fixed=tuple(args.sp_fixed) if args.sp_fixed is not None else None,
            setpoint_scale=args.sp_scale,
            setpoint_clip=args.sp_clip,
            action_mode=args.action_mode,
            min_force=args.min_force,
            max_force=args.max_force,
            pos_alpha=args.pos_alpha,
            event_process=args.event_process,
            event_ou_theta=args.event_ou_theta,
            event_ou_sigma=args.event_ou_sigma
        )
        if args.eval_setpoint is not None:
            obs, _ = test_env.reset(options={"setpoint": np.array(args.eval_setpoint, dtype=np.float32)})
        else:
            obs, _ = test_env.reset()
        total_r = 0.0
        traj = []
        for _ in range(args.eval_steps):
            action, _ = model.predict(obs, deterministic=args.eval_deterministic)
            obs, r, term, trunc, info = test_env.step(action)
            total_r += r
            traj.append((info["vad"].copy(), info["setpoint"].copy(), info["event"].copy(), r))
            if term or trunc:
                break
        total_rewards.append(total_r)
        print(f"[Eval ep={ep}] total_reward={total_r:.3f}")
    if total_rewards:
        mean_r = sum(total_rewards) / len(total_rewards)
        print(f"[Eval] mean_total_reward over {len(total_rewards)} eps = {mean_r:.3f}")

    # ---- save evaluation trajectories ----
    import pandas as pd, os
    os.makedirs("logs", exist_ok=True)
    flat = []
    for ep, traj in enumerate(all_trajs if 'all_trajs' in locals() else [traj]):
        for t, (vad, sp, ev, r) in enumerate(traj):
            flat.append({
                "episode": ep,
                "step": t,
                "vad_v": vad[0], "vad_a": vad[1], "vad_d": vad[2],
                "sp_v": sp[0], "sp_a": sp[1], "sp_d": sp[2],
                "ev_v": ev[0], "ev_a": ev[1], "ev_d": ev[2],
                "reward": r,
            })
    df = pd.DataFrame(flat)
    df.to_csv("logs/vad_eval_log.csv", index=False)
    print("✅ Saved evaluation trajectories to logs/vad_eval_log.csv")

    # print last episode's last 5 steps
    for i, (vad, sp, ev, r) in enumerate(traj[-5:]):
        print(f"t={len(traj)-5+i:>2d}  vad={vad.round(3)}  sp={sp.round(3)}  ev={ev.round(3)}  r={r:.3f}")