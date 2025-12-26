import numpy as np

from emo_phys import SimplePhysics, PhysicsParams, PhysicsContext


def run_demo_sequence():
    """
    baseline -> (이벤트 1회) -> (복원 step N회)
    + 각 tick에서 복원 목표 벡터(restore_vector)와 실제 이동(v_step)을 같이 출력
    """
    rng = np.random.default_rng(seed=42)

    baseline = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    params = PhysicsParams(k_event=0.7, noise_std=0.01)
    physics = SimplePhysics(baseline=baseline, params=params)

    ctx = PhysicsContext(tick=0)

    # 시작 상태
    state = baseline.copy()

    # 이벤트 벡터(예시)
    event_vec = np.array([2.0, 1.0, -1.0], dtype=np.float32)

    print("=== BEFORE EVENT ===")
    print(f"tick={ctx.tick}\tstate={state}\tdist={physics.distance_to_baseline(state):.4f}")

    # 이벤트 1회 적용
    s_event = physics.apply_event(state, event_vec, ctx=ctx, rng=rng)
    v_event = s_event - state

    print("\n=== AFTER EVENT (tick=0) ===")
    print(f"tick={ctx.tick}\tstate={s_event}\tdist={physics.distance_to_baseline(s_event):.4f}")
    print(f"v_event(delta) = {v_event}")

    # 복원 dynamics
    print("\n=== PASSIVE STEPS ===")
    state = s_event

    for t in range(1, 11):
        ctx.tick = t

        v_restore_ideal = physics.restore_vector(state)  # 목표(의도)
        next_state = physics.passive_step(state, ctx=ctx, rng=rng)
        v_step = next_state - state                      # 실제 이동

        dist_now = physics.distance_to_baseline(state)
        dist_next = physics.distance_to_baseline(next_state)

        print(f"[tick={ctx.tick}]")
        print(f"  state        = {state} (dist={dist_now:.4f})")
        print(f"  v_restore*   = {v_restore_ideal}")
        print(f"  v_step(real) = {v_step}")
        print(f"  next_state   = {next_state} (dist={dist_next:.4f})")
        print("")

        state = next_state

    print("=== FINAL ===")
    print(f"tick={ctx.tick}\tstate={state}\tdist={physics.distance_to_baseline(state):.4f}")


if __name__ == "__main__":
    run_demo_sequence()
