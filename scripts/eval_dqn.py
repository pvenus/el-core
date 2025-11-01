import argparse, yaml, numpy as np
from orl.envs.gridworld import GridWorld, GridWorldConfig
from orl.agents.dqn import DQNAgent, DQNConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/dqn.yaml")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    env = GridWorld(GridWorldConfig(**cfg["env"]))
    agent = DQNAgent(env.obs_dim, env.n_actions, DQNConfig(**cfg["algo"]))

    # 간단한 평가: 학습 안된 가중치로 greedy 행동(참고용)
    # (학습된 가중치를 파일로 저장/로드하는 로직은 이후 추가)
    rets=[]
    for _ in range(args.episodes):
        obs = env.reset(); done=False; ep_ret=0.0; steps=0
        while not done and steps < env.cfg.max_steps:
            with agent.q.no_grad():
                act = int(agent.q.__call__(np.expand_dims(obs,0)).argmax(dim=-1).item())
            obs, r, done, _ = env.step(act); ep_ret += r; steps += 1
        if args.render: print(env.render_ascii(), "\n")
        rets.append(ep_ret)
    print(f"Eval mean_return={np.mean(rets):.3f} ± {np.std(rets):.3f}")

if __name__ == "__main__":
    main()
