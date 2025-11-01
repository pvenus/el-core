import argparse, yaml, numpy as np
import torch
from orl.envs.gridworld import GridWorld, GridWorldConfig
from orl.agents.dqn import DQNAgent, DQNConfig
from orl.utils.replay import ReplayBuffer
from orl.utils.schedules import linear_eps_decay

def evaluate(env, agent, episodes=5, render=False):
    agent.q.eval(); rets=[]
    for _ in range(episodes):
        obs = env.reset(); done=False; ep_ret=0.0; steps=0
        while not done and steps < env.cfg.max_steps:
            x = torch.from_numpy(np.expand_dims(obs, 0)).float().to(agent.device)
            with torch.no_grad():
                act = int(agent.q(x).argmax(dim=-1).item())
            obs, r, done, _ = env.step(act); ep_ret += r; steps += 1
        rets.append(ep_ret)
        if render: print(env.render_ascii(), "\n")
    return float(np.mean(rets)), float(np.std(rets))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/dqn.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    env_cfg = GridWorldConfig(**cfg["env"])
    env = GridWorld(env_cfg)

    algo_cfg = DQNConfig(**cfg["algo"])
    # env의 관찰공간/행동공간 크기를 config에 세팅 후 전달
    algo_cfg.obs_dim = env.obs_dim
    algo_cfg.n_actions = env.n_actions
    agent = DQNAgent(algo_cfg)

    buf = ReplayBuffer(algo_cfg.buffer_size)

    grad_steps = 0
    log_every = cfg["log"]["log_every"]
    eval_every = cfg["log"]["eval_every"]
    loss_hist, ret_hist = [], []

    for ep in range(1, algo_cfg.episodes + 1):
        obs = env.reset(); done=False; ep_ret=0.0
        while not done:
            eps = linear_eps_decay(agent.total_steps, algo_cfg.eps_start, algo_cfg.eps_end, algo_cfg.eps_decay)
            a = agent.act(obs, eps)
            ns, r, done, _ = env.step(a)
            buf.push(obs, a, r, ns, float(done))
            obs = ns; ep_ret += r; agent.total_steps += 1

            if len(buf) >= algo_cfg.start_learning and (agent.total_steps % algo_cfg.train_interval == 0):
                loss = agent.update(buf.sample(algo_cfg.batch_size))
                loss_hist.append(loss); grad_steps += 1
                if grad_steps % algo_cfg.target_update_interval == 0:
                    agent.hard_update_target()

        ret_hist.append(ep_ret)

        if ep % log_every == 0:
            ml = float(np.mean(loss_hist[-log_every:])) if loss_hist else float("nan")
            mr = float(np.mean(ret_hist[-log_every:]))
            print(f"[Ep {ep:4d}] steps={agent.total_steps:6d}  ε≈{eps:0.3f}  loss={ml:0.4f}  return={mr:0.3f}")

        if ep % eval_every == 0:
            m, s = evaluate(env, agent, episodes=10)
            print(f"  -> Eval: mean_return={m:0.3f} ± {s:0.3f}")

    m, s = evaluate(env, agent, episodes=20, render=True)
    print(f"[Final Eval] mean_return={m:0.3f} ± {s:0.3f}")

if __name__ == "__main__":
    main()
