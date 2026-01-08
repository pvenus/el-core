

import numpy as np

from sim_rl import SimEnv, load_policy_npz


def run_policy(
    model_path: str,
    steps: int = 3,
    seed: int = 0,
    greedy: bool = True,
):
    """Run a trained policy in the environment.

    Args:
        model_path: Path to saved .npz policy file
        steps: Max rollout steps
        seed: RNG seed
        greedy: If True, use argmax; else sample from distribution
    """
    env = SimEnv(dim=6)
    obs = env.reset(seed=seed)

    policy = load_policy_npz(
        model_path,
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
    )

    total_reward = 0.0
    done = False

    print("=" * 72)
    print(f"[RUN] model={model_path} greedy={greedy}")
    print("=" * 72)

    for t in range(steps):
        probs = policy.action_probs(obs)

        if greedy:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))

        next_obs, reward, done, info = env.step(action)

        total_reward += float(reward)

        print(
            f"[STEP {t:03d}] "
            f"action={env.action_ids[action]} "
            f"reward={float(reward):+.4f} "
            f"dist={info.dist_before:.4f}->{info.dist_after:.4f} "
            f"round={info.round_id} "
            f"choice={info.choice_id}"
        )

        obs = next_obs
        if done:
            print("[DONE] terminal condition reached")
            break

    print("=" * 72)
    print(f"Total reward: {total_reward:+.4f}")
    print("=" * 72)


if __name__ == "__main__":
    run_policy(
        model_path="data/models/policy_reinforce.npz",
        steps=3,
        seed=12345665,
        greedy=True,
    )