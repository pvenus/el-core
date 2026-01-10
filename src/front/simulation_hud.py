# front/simulation.py
from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 3D plot
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ----------------------------
# Import sim engine (robust)
# ----------------------------
try:
    # (A) if sim_engine is top-level package
    from sim_engine.sim_agent import AgentState, SimAgent
    from sim_engine.sim_config import SimulationConfig
    from sim_engine.sim_runner import TurnSimulation
    from sim_engine.sim_types import TurnInput, TurnResult, ActionEffect
except Exception:
    # (B) if it's under "simulation" package
    from simulation.sim_agent import AgentState, SimAgent
    from simulation.sim_config import SimulationConfig
    from simulation.sim_runner import SimRunner
    from simulation.dto.turn import TurnInput, TurnResult, ActionEffect


# ----------------------------
# vector helpers (UI only)
# ----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def v_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def v_sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def v_scale(a: List[float], s: float) -> List[float]:
    return [x * float(s) for x in a]


def v_norm(a: List[float]) -> float:
    return sum(x * x for x in a) ** 0.5


def v_normalize(a: List[float], eps: float = 1e-12) -> List[float]:
    n = v_norm(a)
    if n < eps:
        return [0.0 for _ in a]
    return [x / n for x in a]


def parse_vector(text: str, dim: int, default: Optional[List[float]] = None) -> List[float]:
    if default is None:
        default = [0.0] * dim
    s = (text or "").strip()
    if not s:
        return default
    try:
        if s.startswith("["):
            v = json.loads(s)
        else:
            v = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
        if len(v) != dim:
            raise ValueError(f"dim mismatch: expected {dim}, got {len(v)}")
        return [float(x) for x in v]
    except Exception as e:
        raise ValueError(f"Invalid vector text: {e}")


def gauge(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return (clamp(value, vmin, vmax) - vmin) / (vmax - vmin)


def results_to_df(results: List[TurnResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "turn": r.turn,
                "action_id": r.action_id,
                "distance_to_comfort": r.distance_to_comfort,
                "in_comfort": r.in_comfort,
                "before_vec": r.before_vec,
                "after_vec": r.after_vec,
                "applied_event_delta": r.applied_event_delta,
                "applied_action_delta": r.applied_action_delta,
                "vars_before": r.vars_before,
                "vars_after": r.vars_after,
                "meta": r.meta,
            }
        )
    return pd.DataFrame(rows)


def get_path_from_results(agent_current: List[float], results: List[TurnResult]) -> List[List[float]]:
    if not results:
        return [list(agent_current)]
    path = [list(results[0].before_vec)]
    for r in results:
        path.append(list(r.after_vec))
    return path


# ----------------------------
# plotting
# ----------------------------
def plot_state_2d(
    current: List[float],
    comfort: List[float],
    radius: float,
    path: List[List[float]],
    last_before: Optional[List[float]] = None,
    last_after: Optional[List[float]] = None,
    show_arrow: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    circ = plt.Circle((comfort[0], comfort[1]), radius, fill=False)
    ax.add_patch(circ)

    if len(path) >= 2:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, marker="o", linewidth=1)

    ax.scatter([comfort[0]], [comfort[1]], marker="x")
    ax.scatter([current[0]], [current[1]])

    if show_arrow and last_before is not None and last_after is not None:
        ax.annotate(
            "",
            xy=(last_after[0], last_after[1]),
            xytext=(last_before[0], last_before[1]),
            arrowprops=dict(arrowstyle="->"),
        )

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title("2D state view (current / comfort / path)")
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, alpha=0.2)
    return fig


def plot_state_3d(
    current: List[float],
    comfort: List[float],
    radius: float,
    path: List[List[float]],
    last_before: Optional[List[float]] = None,
    last_after: Optional[List[float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if len(path) >= 2:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax.plot(xs, ys, zs, marker="o", linewidth=1)

    ax.scatter([comfort[0]], [comfort[1]], [comfort[2]], marker="x")
    ax.scatter([current[0]], [current[1]], [current[2]])

    if last_before is not None and last_after is not None:
        ax.plot(
            [last_before[0], last_after[0]],
            [last_before[1], last_after[1]],
            [last_before[2], last_after[2]],
            linewidth=2,
        )

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("x2")
    ax.set_title("3D state view (current / comfort / path)")
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if zlim:
        ax.set_zlim(zlim[0], zlim[1])
    return fig


# ----------------------------
# Streamlit page entry
# ----------------------------
def render_simulation() -> None:
    st.title("Simulation")
    st.caption("sim_engine HUD (draft) — TurnResult 중심 가시화 + 조작성 테스트")

    # -------- session state --------
    if "sim_turn" not in st.session_state:
        st.session_state.sim_turn = 0
    if "sim_agent" not in st.session_state:
        st.session_state.sim_agent = None
    if "sim_runner" not in st.session_state:
        st.session_state.sim_runner = None
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = []
    if "sim_vis_dim" not in st.session_state:
        st.session_state.sim_vis_dim = 2
    if "sim_gmin" not in st.session_state:
        st.session_state.sim_gmin = -2.0
    if "sim_gmax" not in st.session_state:
        st.session_state.sim_gmax = 2.0

    # -------- sidebar controls --------
    with st.sidebar:
        st.subheader("Simulation HUD")

        st.session_state.sim_vis_dim = st.radio("view dim", options=[2, 3], index=0, horizontal=True)

        use_advanced_dim = st.checkbox("advanced dim (4D+)", value=False)
        if use_advanced_dim:
            dim = st.number_input("dim", min_value=2, max_value=64, value=max(8, st.session_state.sim_vis_dim), step=1)
        else:
            dim = st.session_state.sim_vis_dim

        st.divider()
        st.subheader("Gauge range")
        st.session_state.sim_gmin = st.number_input("gauge_min", value=float(st.session_state.sim_gmin), step=0.5)
        st.session_state.sim_gmax = st.number_input("gauge_max", value=float(st.session_state.sim_gmax), step=0.5)

        st.divider()
        st.subheader("Init state")
        comfort_radius = st.number_input("comfort_radius", min_value=0.0, value=1.0, step=0.1)

        default_comfort = ",".join(["0"] * int(dim))
        default_current = ",".join(["0.2"] + ["0"] * (int(dim) - 1))
        comfort_vec_txt = st.text_input("comfort_vec", value=default_comfort)
        current_vec_txt = st.text_input("current_vec", value=default_current)
        vars_json_txt = st.text_area("vars (JSON)", value='{"energy": 10, "stamina": 5}', height=80)

        st.divider()
        st.subheader("SimulationConfig")
        dt = st.number_input("dt", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
        damping = st.number_input("damping(0~1)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

        clamp_norm_on = st.checkbox("clamp_norm", value=False)
        clamp_norm_val = st.number_input("clamp_norm value", min_value=0.01, value=5.0, step=0.5, disabled=not clamp_norm_on)

        noise_scale = st.number_input("noise_scale", min_value=0.0, value=0.0, step=0.01)

        comfort_override_on = st.checkbox("comfort_radius_override", value=False)
        comfort_override_val = st.number_input("override value", min_value=0.0, value=float(comfort_radius), step=0.1, disabled=not comfort_override_on)

        st.divider()
        st.subheader("vars clamp (optional)")
        vars_min_txt = st.text_input('vars_min (JSON)', value='{"energy": 0}')
        vars_max_txt = st.text_input('vars_max (JSON)', value='{"energy": 10}')

        st.divider()
        if st.button("✅ Reset / Build Agent", use_container_width=True):
            try:
                comfort_vec = parse_vector(comfort_vec_txt, int(dim))
                current_vec = parse_vector(current_vec_txt, int(dim))
                vars_dict = json.loads(vars_json_txt) if vars_json_txt.strip() else {}
                vars_dict = {k: float(v) for k, v in vars_dict.items()}

                cfg = SimulationConfig(
                    dt=float(dt),
                    damping=float(damping),
                    clamp_norm=float(clamp_norm_val) if clamp_norm_on else None,
                    noise_scale=float(noise_scale),
                    comfort_radius_override=float(comfort_override_val) if comfort_override_on else None,
                    vars_min=json.loads(vars_min_txt) if vars_min_txt.strip() else {},
                    vars_max=json.loads(vars_max_txt) if vars_max_txt.strip() else {},
                )
                cfg.validate()

                agent_state = AgentState(
                    comfort_vec=comfort_vec,
                    comfort_radius=float(comfort_radius),
                    current_vec=current_vec,
                    vars=vars_dict,
                )

                st.session_state.sim_agent = SimAgent(state=agent_state)
                st.session_state.sim_runner = SimRunner(cfg=cfg)
                st.session_state.sim_turn = 0
                st.session_state.sim_results = []
                st.success("Initialized.")
            except Exception as e:
                st.error(f"Init failed: {e}")

    # -------- load objects --------
    agent: Optional[SimAgent] = st.session_state.sim_agent
    runner: Optional[SimRunner] = st.session_state.sim_runner
    if agent is None or runner is None:
        st.info("왼쪽 사이드바에서 Reset / Build Agent 먼저 눌러줘.")
        return

    comfort_r = runner.cfg.get_comfort_radius(agent.state.comfort_radius)
    dist = agent.distance_to_comfort()

    # -------- top metrics --------
    c1, c2, c3, c4 = st.columns([0.8, 1.0, 1.0, 1.0])
    c1.metric("turn", int(st.session_state.sim_turn))
    c2.metric("comfort_radius", float(comfort_r))
    c3.metric("distance_to_comfort", float(dist))
    c4.metric("in_comfort", bool(dist <= comfort_r))

    with st.expander("Vectors (debug)", expanded=False):
        st.write("comfort_vec =", agent.state.comfort_vec)
        st.write("current_vec =", agent.state.current_vec)

    # -------- gauges --------
    st.markdown("### current_vec axes")
    cols = st.columns(4)
    for i, val in enumerate(agent.state.current_vec):
        with cols[i % 4]:
            st.write(f"axis {i}: **{val:.4f}**")
            st.progress(gauge(val, float(st.session_state.sim_gmin), float(st.session_state.sim_gmax)))

    st.divider()

    # -------- visualization --------
    st.subheader("2D/3D View")
    vis_dim = int(st.session_state.sim_vis_dim)

    with st.expander("view window (optional)", expanded=False):
        auto_view = st.checkbox("auto view", value=True)
        if not auto_view:
            xlim = st.slider("x0 range", -10.0, 10.0, (-3.0, 3.0))
            ylim = st.slider("x1 range", -10.0, 10.0, (-3.0, 3.0))
            zlim = st.slider("x2 range", -10.0, 10.0, (-3.0, 3.0)) if vis_dim == 3 else None
        else:
            xlim = ylim = zlim = None

    plot_box = st.empty()

    def render_plot(last_before=None, last_after=None):
        path = get_path_from_results(agent.state.current_vec, st.session_state.sim_results)
        cur = agent.state.current_vec + [0.0] * max(0, vis_dim - len(agent.state.current_vec))
        com = agent.state.comfort_vec + [0.0] * max(0, vis_dim - len(agent.state.comfort_vec))

        if vis_dim == 2:
            fig = plot_state_2d(
                current=cur[:2],
                comfort=com[:2],
                radius=float(comfort_r),
                path=[p[:2] for p in path],
                last_before=last_before[:2] if last_before else None,
                last_after=last_after[:2] if last_after else None,
                xlim=xlim,
                ylim=ylim,
            )
        else:
            fig = plot_state_3d(
                current=cur[:3],
                comfort=com[:3],
                radius=float(comfort_r),
                path=[p[:3] for p in path],
                last_before=last_before[:3] if last_before else None,
                last_after=last_after[:3] if last_after else None,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
            )
        plot_box.pyplot(fig, clear_figure=True)

    render_plot()

    st.divider()

    # -------- input panels (Event + ActionEffect + Turn) --------
    left, right = st.columns([1.05, 0.95])

    with left:
        st.subheader("Event (external) — add to agent.events")

        e_start = st.number_input("event.start_turn", min_value=0, value=int(st.session_state.sim_turn), step=1)
        e_dur = st.number_input("event.duration", min_value=1, value=3, step=1)
        e_mag = st.slider("event.magnitude", min_value=0.0, max_value=2.0, value=0.2, step=0.01)

        e_preset = st.selectbox(
            "event.direction preset",
            ["toward comfort", "away from comfort", "axis unit (+x0)", "manual sliders"],
            index=0,
        )

        dim = len(agent.state.current_vec)
        dir_vec = [0.0] * dim
        if e_preset == "toward comfort":
            dir_vec = v_normalize(v_sub(agent.state.comfort_vec, agent.state.current_vec))
        elif e_preset == "away from comfort":
            dir_vec = v_normalize(v_sub(agent.state.current_vec, agent.state.comfort_vec))
        elif e_preset == "axis unit (+x0)":
            dir_vec = [0.0] * dim
            dir_vec[0] = 1.0
        else:
            base_axes = min(dim, 3 if vis_dim == 3 else 2)
            for k in range(base_axes):
                dir_vec[k] = st.slider(f"evt.dir axis {k}", -1.0, 1.0, 0.0, 0.01, key=f"evt_dir_{k}")
            if dim > base_axes:
                with st.expander("more axes"):
                    for k in range(base_axes, dim):
                        dir_vec[k] = st.slider(f"evt.dir axis {k}", -1.0, 1.0, 0.0, 0.01, key=f"evt_dir_more_{k}")
            dir_vec = v_normalize(dir_vec)

        st.write("event.direction =", [round(x, 4) for x in dir_vec[: min(6, dim)]])

        if st.button("➕ Add Event", use_container_width=True):
            try:
                agent.add_event(start_turn=int(e_start), duration=int(e_dur), direction=dir_vec, magnitude=float(e_mag))
                st.success("Event added.")
            except Exception as e:
                st.error(f"Add event failed: {e}")

    with right:
        st.subheader("Turn Step — ActionEffect + runner.step()")

        action_id = st.text_input("action_id", value="act_demo")

        a_preset = st.selectbox(
            "action delta preset",
            ["toward comfort", "away from comfort", "axis unit (+x0)", "manual sliders"],
            index=0,
        )
        a_mag = st.slider("action magnitude", min_value=0.0, max_value=2.0, value=0.2, step=0.01)

        dim = len(agent.state.current_vec)
        action_dir = [0.0] * dim
        if a_preset == "toward comfort":
            action_dir = v_normalize(v_sub(agent.state.comfort_vec, agent.state.current_vec))
        elif a_preset == "away from comfort":
            action_dir = v_normalize(v_sub(agent.state.current_vec, agent.state.comfort_vec))
        elif a_preset == "axis unit (+x0)":
            action_dir = [0.0] * dim
            action_dir[0] = 1.0
        else:
            base_axes = min(dim, 3 if vis_dim == 3 else 2)
            for k in range(base_axes):
                action_dir[k] = st.slider(f"act.dir axis {k}", -1.0, 1.0, 0.0, 0.01, key=f"act_dir_{k}")
            if dim > base_axes:
                with st.expander("more axes"):
                    for k in range(base_axes, dim):
                        action_dir[k] = st.slider(
                            f"act.dir axis {k}",
                            -1.0, 1.0, 0.0, 0.01,
                            key=f"act_dir_more_{k}",
                        )
            action_dir = v_normalize(action_dir)

        action_delta_vec = v_scale(action_dir, float(a_mag))
        st.write("action delta_vec =", [round(x, 4) for x in action_delta_vec[: min(6, dim)]])

        st.caption("delta_vars")
        energy_d = st.slider("Δenergy", -5.0, 5.0, -0.5, 0.1)
        stamina_d = st.slider("Δstamina", -5.0, 5.0, 0.0, 0.1)
        extra_vars_txt = st.text_area("extra delta_vars (JSON)", value="{}", height=80)
        meta_txt = st.text_area("meta (JSON)", value='{"note":"sim ui"}', height=80)

        animate = st.checkbox("animate", value=True)
        frame_delay = st.slider("frame delay (sec)", 0.0, 0.5, 0.12, 0.01)

        def build_action_effect() -> ActionEffect:
            delta_vars = {"energy": float(energy_d), "stamina": float(stamina_d)}
            extra = json.loads(extra_vars_txt) if extra_vars_txt.strip() else {}
            for k, v in extra.items():
                delta_vars[str(k)] = float(v)
            delta_vars = {k: v for k, v in delta_vars.items() if abs(v) > 1e-12}
            return ActionEffect(delta_vec=list(action_delta_vec), delta_vars=delta_vars)

        def build_meta() -> Dict[str, Any]:
            return json.loads(meta_txt) if meta_txt.strip() else {}

        def do_one_turn():
            ti = TurnInput(
                turn=int(st.session_state.sim_turn),
                action_id=action_id.strip() or None,
                action_effect=build_action_effect(),
                external_event_delta=None,
                meta=build_meta(),
            )
            before = list(agent.state.current_vec)
            r = runner.step(agent, ti)
            after = list(agent.state.current_vec)
            st.session_state.sim_results.append(r)
            st.session_state.sim_turn += 1
            return before, after, r

        b1, b2 = st.columns([1.0, 1.0])
        with b1:
            do_step = st.button("▶ Step 1", use_container_width=True)
        with b2:
            n_steps = st.number_input("Run N", min_value=1, max_value=500, value=15, step=1)

        do_run_n = st.button("⏩ Run N turns", use_container_width=True)

        if do_step:
            try:
                before, after, r = do_one_turn()
                if animate:
                    frames = 6
                    for i in range(frames):
                        alpha = i / (frames - 1)
                        interp = [b + (a - b) * alpha for b, a in zip(before, after)]
                        render_plot(last_before=before, last_after=interp)
                        time.sleep(float(frame_delay))
                else:
                    render_plot(last_before=before, last_after=after)
                st.success(f"Step ok: turn={r.turn}")
            except Exception as e:
                st.error(f"Step failed: {e}")

        if do_run_n:
            try:
                for _ in range(int(n_steps)):
                    before, after, r = do_one_turn()
                    if animate:
                        render_plot(last_before=before, last_after=after)
                        time.sleep(float(frame_delay))
                render_plot()
                st.success(f"Run N ok: +{int(n_steps)}")
            except Exception as e:
                st.error(f"Run N failed: {e}")

    st.divider()

    # -------- results --------
    st.subheader("TurnResult logs")
    if len(st.session_state.sim_results) == 0:
        st.info("No results yet. Step 해봐.")
        return

    df = results_to_df(st.session_state.sim_results)
    st.dataframe(df[["turn", "action_id", "distance_to_comfort", "in_comfort"]], use_container_width=True)

    st.markdown("### distance_to_comfort")
    fig = plt.figure()
    plt.plot(df["turn"], df["distance_to_comfort"])
    plt.xlabel("turn")
    plt.ylabel("distance_to_comfort")
    st.pyplot(fig, clear_figure=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.sim_results = []
            st.session_state.sim_turn = 0
            render_plot()
            st.success("Cleared.")
    with c2:
        st.download_button(
            "📦 Download results.json",
            data=json.dumps([asdict(r) for r in st.session_state.sim_results], ensure_ascii=False, indent=2),
            file_name="results.json",
            mime="application/json",
            use_container_width=True,
        )
    with c3:
        with st.expander("raw TurnResult", expanded=False):
            st.json([asdict(r) for r in st.session_state.sim_results])

render_simulation()