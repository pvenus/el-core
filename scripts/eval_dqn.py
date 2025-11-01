import argparse, yaml, numpy as np, torch
from orl.envs.gridworld import GridWorld, GridWorldConfig
from orl.agents.dqn import DQNAgent, DQNConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/dqn.yaml")
    ap.add_argument("--model", type=str, default="data/checkpoints/dqn_final.pt")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    # -----------------------------
    # 1️⃣ Config 로드
    # -----------------------------
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # -----------------------------
    # 2️⃣ 환경 및 에이전트 초기화
    # -----------------------------
    env = GridWorld(GridWorldConfig(**cfg["env"]))
    algo_cfg = DQNConfig(**cfg["algo"])
    algo_cfg.obs_dim = env.obs_dim
    algo_cfg.n_actions = env.n_actions
    agent = DQNAgent(algo_cfg)

    # -----------------------------
    # 3️⃣ 학습된 모델 가중치 로드
    # -----------------------------
    print(f"Loading model from {args.model}")
    state_dict = torch.load(args.model, map_location=agent.device)
    agent.q.load_state_dict(state_dict)
    agent.q.eval()

    # -----------------------------
    # 4️⃣ 평가 루프
    # -----------------------------
    returns = []
    for ep in range(args.episodes):
        obs = env.reset()
        done, ep_ret, steps = False, 0.0, 0

        while not done and steps < env.cfg.max_steps:
            x = torch.from_numpy(np.expand_dims(obs, 0)).float().to(agent.device)
            with torch.no_grad():
                act = int(agent.q(x).argmax(dim=-1).item())
            obs, r, done, _ = env.step(act)
            ep_ret += r
            steps += 1

            if args.render:
                print(env.render_ascii(), "\n")

        returns.append(ep_ret)

    # -----------------------------
    # 5️⃣ 결과 출력
    # -----------------------------
    print(f"Eval mean_return={np.mean(returns):.3f} ± {np.std(returns):.3f}")


if __name__ == "__main__":
    main()
