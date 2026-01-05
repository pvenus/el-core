# src/emo_phys/tests/test_simple_physics.py

import numpy as np

from emo_phys import SimplePhysics, PhysicsParams


def run_demo_steps():
    """
    간단히 상태가 어떻게 움직이는지 눈으로 확인하는 데모.
    """
    rng = np.random.default_rng(seed=42)

    # 3차원 PCA 공간이라고 가정.
    baseline = np.array([0.0, 0.0, 0.0], dtype=np.float32)        # 기준점(중립 상태)
    initial_state = np.array([1.0, -0.5, 0.8], dtype=np.float32)  # 시작 상태
    event_vec = np.array([2.0, 1.0, -1.0], dtype=np.float32)      # 어떤 이벤트 벡터

    params = PhysicsParams(
        k_event=0.5,
        k_restore=0.1,
        noise_std=0.01,
    )

    physics = SimplePhysics(baseline=baseline, params=params)

    state = initial_state.copy()

    print("step\tstate\t\t\t\t distance_to_baseline")
    print("-" * 60)
    for t in range(10):
        dist = physics.distance_to_baseline(state)
        print(f"{t}\t{state}\t {dist:.4f}")

        # 한 스텝 업데이트
        state = physics.step(state, event_vec, rng=rng)

    # 마지막 상태까지 본 뒤 출력
    dist = physics.distance_to_baseline(state)
    print(f"final\t{state}\t {dist:.4f}")


def test_simple_physics_moves_towards_baseline():
    """
    아주 느슨한 단위 테스트:
    첫 스텝에서 baseline과의 거리가
    '극단적으로' 멀어지지 않는지만 확인한다.
    (완벽한 수렴 테스트가 아니라 sanity check 용도)
    """
    rng = np.random.default_rng(seed=0)

    baseline = np.zeros(3, dtype=np.float32)
    initial_state = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    event_vec = np.array([2.0, 0.0, 0.0], dtype=np.float32)

    params = PhysicsParams(k_event=0.5, k_restore=0.1, noise_std=0.0)
    physics = SimplePhysics(baseline=baseline, params=params)

    prev_dist = physics.distance_to_baseline(initial_state)
    next_state = physics.step(initial_state, event_vec, rng=rng)
    next_dist = physics.distance_to_baseline(next_state)

    # baseline(0,0,0) 기준으로 완전 이상하게 멀어지지는 않는지만 체크
    assert next_dist < prev_dist * 2.0


if __name__ == "__main__":
    # pytest 없이도 그냥 돌려볼 수 있게
    run_demo_steps()
