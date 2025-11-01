# scripts/train_dqn.py
import argparse
import yaml
import numpy as np
import torch
from orl.envs.gridworld import GridWorld, GridWorldConfig
from orl.agents.dqn import DQNAgent, DQNConfig
from orl.utils.replay import ReplayBuffer
from orl.utils.schedules import linear_eps_decay


# =====================================================
# EVALUATION
# =====================================================
def evaluate(env, agent, episodes: int = 5, render: bool = False):
    """Evaluate the trained agent without exploration."""
    agent.q.eval()
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        done, ep_ret, steps = False, 0.0, 0
        while not done and steps < env.cfg.max_steps:
            x = torch.from_numpy(np.expand_dims(obs, 0)).float().to(agent.device)
            with torch.no_grad():
                act = int(agent.q(x).argmax(dim=-1).item())
            obs, r, done, _ = env.step(act)
            ep_ret += r
            steps += 1
            if render:
                print(env.render_ascii(), "\n")
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))


# =====================================================
# MAIN TRAIN LOOP
# =====================================================
def main():
    # ------------------------------
    # 1. Config 로드 (인코딩 명시)
    # ------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/dqn.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ------------------------------
    # 2. Env / Agent 초기화
    # ------------------------------
    env_cfg = GridWorldConfig(**cfg["env"])
    env = GridWorld(env_cfg)

    algo_cfg = DQNConfig(**cfg["algo"])
    algo_cfg.obs_dim = env.obs_dim
    algo_cfg.n_actions = env.n_actions
    agent = DQNAgent(algo_cfg)

    buf = ReplayBuffer(algo_cfg.buffer_size)

    # ------------------------------
    # 3. Logging 설정 (기본값 안전 처리)
    # ------------------------------
    log_cfg = cfg.get("log", {})
    log_every = log_cfg.get("log_every", 20)
    eval_every = log_cfg.get("eval_every", 100)

    grad_steps = 0
    loss_hist, ret_hist = [], []

    # ------------------------------
    # 4. 학습 루프
    # ------------------------------
    for ep in range(1, algo_cfg.episodes + 1):
        obs = env.reset()
        done, ep_ret = False, 0.0

        # optional shaping용 변수 (맨해튼 거리 기반)
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        prev_phi = -manhattan(env.agent, env.cfg.goal)

        while not done:
            eps = linear_eps_decay(
                agent.total_steps, algo_cfg.eps_start, algo_cfg.eps_end, algo_cfg.eps_decay
            )
            a = agent.act(obs, eps)
            ns, r_env, done, _ = env.step(a)

            # --- Potential-based shaping 보상 추가 ---
            next_phi = -manhattan(env.agent, env.cfg.goal)
            beta = 0.01
            r = r_env + beta * (next_phi - prev_phi)
            prev_phi = next_phi
            # ------------------------------------------------

            buf.push(obs, a, r, ns, float(done))
            obs = ns
            ep_ret += r
            agent.total_steps += 1

            if len(buf) >= algo_cfg.start_learning and (
                agent.total_steps % algo_cfg.train_interval == 0
            ):
                loss = agent.update(buf.sample(algo_cfg.batch_size))
                loss_hist.append(loss)
                grad_steps += 1

                if grad_steps % algo_cfg.target_update_interval == 0:
                    agent.hard_update_target()

        ret_hist.append(ep_ret)

        # ------------------------------
        # 5. Logging
        # ------------------------------
        if ep % log_every == 0:
            ml = float(np.mean(loss_hist[-log_every:])) if loss_hist else float("nan")
            mr = float(np.mean(ret_hist[-log_every:]))
            print(
                f"[Ep {ep:4d}] steps={agent.total_steps:6d}  "
                f"ε≈{eps:0.3f}  loss={ml:0.4f}  return={mr:0.3f}"
            )

        if ep % eval_every == 0:
            m, s = evaluate(env, agent, episodes=10)
            print(f"  -> Eval: mean_return={m:0.3f} ± {s:0.3f}")

    # ------------------------------
    # 6. 최종 평가 및 모델 저장
    # ------------------------------
    m, s = evaluate(env, agent, episodes=20, render=True)
    print(f"[Final Eval] mean_return={m:0.3f} ± {s:0.3f}")

    torch.save(agent.q.state_dict(), "data/checkpoints/dqn_final.pt")
    print("✅ Model saved to data/checkpoints/dqn_final.pt")

# =====================================================
if __name__ == "__main__":
    main()
