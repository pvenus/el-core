# src/front/simulation.py
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH so `import src....` works when Streamlit changes CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../el-core
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import numpy as np
import streamlit as st

from src.simulation.dto.vector_space import VectorSpaceSpec
from src.simulation.dto.step_io import StepInput, StepResult
from src.simulation.dto.impact import Impact
from src.simulation.sim_manager import SimulationManager
from src.simulation.sim_builders import (
    DemoSpaceBuilder,
    DemoAgentSpecSource,
    ListAgentSpecSource,
    build_demo_step_input_json,  # -> list[dict] (10 samples)
)


# =========================================================
# Session helpers
# =========================================================
def _ensure_state():
    st.session_state.setdefault("space_spec", None)
    st.session_state.setdefault("agent_specs", None)
    st.session_state.setdefault("manager", None)

    # single-agent queue
    st.session_state.setdefault("impact_queue", [])  # list[Impact]
    st.session_state.setdefault("history", [])       # list[StepResult]
    st.session_state.setdefault("demo_samples", None)

    # display only
    st.session_state.setdefault("progress_range", 3.0)

    # impact form defaults (these are used as "initial values" only)
    st.session_state.setdefault("impact_target_sel", None)
    st.session_state.setdefault("impact_magnitude", 1.0)
    st.session_state.setdefault("impact_duration", 1)

    # axis controls
    st.session_state.setdefault("axis_enabled", {})  # axis_key -> bool
    st.session_state.setdefault("axis_sign", {})     # axis_key -> float [-1..1]
    st.session_state.setdefault("axis_weight", {})   # axis_key -> float >=0


def _clear_runtime():
    st.session_state["manager"] = None
    st.session_state["impact_queue"] = []
    st.session_state["history"] = []
    st.session_state["demo_samples"] = None


# =========================================================
# Math helpers
# =========================================================
def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v, dtype=float)
    return v / n


def _axis_progress(value: float, *, rng: float) -> float:
    """
    표시 전용:
    [-rng, +rng] 범위를 [0..1]로 매핑해서 progress bar에 표시
    """
    if rng <= 0:
        return 0.5
    x = (value + rng) / (2 * rng)
    return float(min(1.0, max(0.0, x)))


def _impact_json_to_form(space: VectorSpaceSpec, agent_ids: list[str], payload: dict):
    """
    payload:
    {
      "target": "agent_01",
      "magnitude": 1.0,
      "duration": 1,
      "axis_weights": {"axis_00": 1.0, "axis_03": -0.5}
    }
    axis_weights 값은 eff = sign*weight로 취급.
    """
    target = payload.get("target") or (agent_ids[0] if agent_ids else "")
    if target not in agent_ids:
        target = agent_ids[0] if agent_ids else ""

    st.session_state["impact_target_sel"] = target
    st.session_state["impact_magnitude"] = float(payload.get("magnitude", 1.0))
    st.session_state["impact_duration"] = int(payload.get("duration", 1))

    aw = payload.get("axis_weights") or {}
    for axis in space.axes:
        k = axis.key
        eff = float(aw.get(k, 0.0))
        st.session_state["axis_enabled"][k] = (abs(eff) > 0)

        if abs(eff) > 0:
            st.session_state["axis_sign"][k] = 1.0 if eff > 0 else -1.0
            st.session_state["axis_weight"][k] = abs(eff)
        else:
            st.session_state["axis_sign"][k] = 0.0
            st.session_state["axis_weight"][k] = 1.0


def _form_to_direction(space: VectorSpaceSpec) -> np.ndarray:
    """
    direction = normalize( Σ proto_vec * (sign*weight) )
    """
    v = np.zeros((space.dim,), dtype=float)
    for axis in space.axes:
        k = axis.key
        if not bool(st.session_state["axis_enabled"].get(k, False)):
            continue
        sign = float(st.session_state["axis_sign"].get(k, 0.0))      # -1..1
        weight = float(st.session_state["axis_weight"].get(k, 1.0))  # >=0
        eff = sign * weight
        if abs(eff) < 1e-12:
            continue
        v = v + axis.proto_vec * eff
    return _normalize(v)


def _current_form_as_json(selected_agent: str, space: VectorSpaceSpec) -> dict:
    axis_weights = {}
    for axis in space.axes:
        k = axis.key
        if st.session_state["axis_enabled"].get(k, False):
            eff = float(st.session_state["axis_sign"][k]) * float(st.session_state["axis_weight"][k])
            if abs(eff) > 1e-12:
                axis_weights[k] = eff

    return {
        "target": st.session_state.get("impact_target_sel", selected_agent) or selected_agent,
        "magnitude": float(st.session_state.get("impact_magnitude", 1.0)),
        "duration": int(st.session_state.get("impact_duration", 1)),
        "axis_weights": axis_weights,
    }


# =========================================================
# Render
# =========================================================
def render():
    st.title("Simulation")

    _ensure_state()

    # ==========================================
    # Sidebar: Setup
    # ==========================================
    with st.sidebar:
        st.header("Setup")

        # ---- Space ----
        st.subheader("VectorSpace")
        dim = st.number_input("dim", min_value=1, max_value=64, value=7, step=1, key="dim_in")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Create Space", use_container_width=True):
                st.session_state.space_spec = DemoSpaceBuilder(dim=int(dim)).build()
                st.session_state.agent_specs = None
                _clear_runtime()
        with c2:
            if st.button("Clear All", use_container_width=True):
                st.session_state.space_spec = None
                st.session_state.agent_specs = None
                _clear_runtime()

        space_upload = st.file_uploader("Load Space JSON", type=["json"], key="space_upload")
        if space_upload is not None:
            try:
                raw = json.loads(space_upload.getvalue().decode("utf-8"))
                st.session_state.space_spec = VectorSpaceSpec.from_dict(raw)
                st.session_state.agent_specs = None
                _clear_runtime()
                st.success("Space loaded.")
            except Exception as e:
                st.error(f"Space load failed: {e}")

        st.divider()

        # ---- AgentSpecs ----
        st.subheader("AgentSpecs (default 1)")
        radius = st.number_input("radius", min_value=0.0, max_value=100.0, value=1.5, step=0.1, key="radius_in")

        c3, c4 = st.columns(2)
        with c3:
            if st.button("Create AgentSpecs", use_container_width=True):
                if st.session_state.space_spec is None:
                    st.warning("Create/Load Space first.")
                else:
                    src = DemoAgentSpecSource(radius=float(radius))
                    st.session_state.agent_specs = src.build_agent_specs(st.session_state.space_spec)
                    _clear_runtime()
        with c4:
            if st.button("Clear AgentSpecs", use_container_width=True):
                st.session_state.agent_specs = None
                _clear_runtime()

        st.divider()

        # ---- Manager ----
        st.subheader("Manager")
        agent_count = st.number_input("agent_count (take N)", min_value=1, max_value=10, value=1, step=1, key="agent_count_in")

        if st.button("Create/Reset Manager", use_container_width=True):
            if st.session_state.space_spec is None or not st.session_state.agent_specs:
                st.warning("Space + AgentSpecs are required.")
            else:
                src = ListAgentSpecSource(specs=st.session_state.agent_specs)
                st.session_state.manager = SimulationManager.create(
                    space_spec=st.session_state.space_spec,
                    agent_spec_source=src,
                    agent_count=int(agent_count),
                )
                st.session_state.impact_queue = []
                st.session_state.history = []

                # demo samples
                first_agent_id = st.session_state.manager.agents[0].agent_id
                st.session_state.demo_samples = build_demo_step_input_json(
                    space_dim=st.session_state.space_spec.dim,
                    agent_id=first_agent_id,
                )

                # impact form 초기 타겟 세팅
                st.session_state.impact_target_sel = first_agent_id

                st.success("Manager created.")

    # ==========================================
    # Main summary
    # ==========================================
    space: VectorSpaceSpec | None = st.session_state.space_spec
    mgr: SimulationManager | None = st.session_state.manager

    st.subheader("Current Setup")
    cols = st.columns(3)
    cols[0].metric("Space", "OK" if space else "None")
    cols[1].metric("AgentSpecs", str(len(st.session_state.agent_specs)) if st.session_state.agent_specs else "0")
    cols[2].metric("Manager", "OK" if mgr else "None")

    if mgr is None or space is None:
        st.info("좌측에서 Space/AgentSpecs/Manager를 만들면 아래 뷰가 활성화돼.")
        return

    snap = mgr.snapshot()
    agent_ids = list(snap.keys())
    if not agent_ids:
        st.warning("No agents in manager.")
        return

    # Selected agent (for state view + stepping)
    selected_agent = st.selectbox("Active agent (for Step)", agent_ids, index=0, key="active_agent_sel")

    # =========================================================
    # A) Agent State (left) | C) Impact Queue (right)
    # =========================================================
    st.header("Agent State & Impact Queue")
    col_state, col_queue = st.columns([2, 1], gap="large")

    with col_state:
        st.subheader("Agent State View")

        s = snap[selected_agent]
        st.write(
            {
                "agent_id": selected_agent,
                "radius": s["radius"],
                "distance_to_comfort": s["distance_to_comfort"],
                "is_in_comfort": s["is_in_comfort"],
                "active_impacts": s["active_impacts"],
            }
        )

        # display only control (no effect on simulation)
        rng = st.slider(
            "progress range (display only, ±range)",
            min_value=0.5,
            max_value=10.0,
            value=float(st.session_state.get("progress_range", 3.0)),
            step=0.5,
            key="progress_range",
        )

        current_vec = s["current_vec"]
        st.caption("current_vec by axis")
        for i, axis in enumerate(space.axes):
            val = float(current_vec[i])
            st.caption(f"{axis.key}: {val:.4f}")
            st.progress(_axis_progress(val, rng=float(rng)))

    with col_queue:
        st.subheader("Impact Queue (details)")
        q: list[Impact] = st.session_state["impact_queue"]

        if not q:
            st.info("Queue is empty.")
        else:
            rows = []
            for idx, imp in enumerate(q):
                rows.append(
                    {
                        "idx": idx,
                        "magnitude": float(imp.magnitude),
                        "duration": int(imp.duration),
                        "direction_preview": [float(x) for x in imp.direction[: min(6, space.dim)]],
                        "direction_norm": float(np.linalg.norm(imp.direction)),
                    }
                )
            st.dataframe(rows, use_container_width=True, height=280)

            st.divider()
            rm_idx = st.number_input(
                "remove idx",
                min_value=0,
                max_value=max(0, len(q) - 1),
                value=0,
                step=1,
                key="queue_remove_idx",
            )
            r1, r2 = st.columns(2)
            with r1:
                if st.button("Remove", use_container_width=True):
                    try:
                        q.pop(int(rm_idx))
                        st.session_state["impact_queue"] = q
                        st.success("Removed.")
                    except Exception as e:
                        st.error(f"Remove failed: {e}")
            with r2:
                if st.button("Clear", use_container_width=True):
                    st.session_state["impact_queue"] = []
                    st.success("Cleared.")

        st.divider()
        st.subheader("Demo StepInput Samples")
        samples = st.session_state.get("demo_samples") or []
        if samples:
            idx = st.selectbox("sample idx", list(range(1, len(samples) + 1)), index=0, key="demo_sample_idx")
            if st.button("Load sample → Queue", use_container_width=True):
                payload = samples[int(idx) - 1]
                step_in = StepInput.from_dict(payload)
                # 샘플의 agent_id가 다르면 active agent를 샘플로 바꾸는 게 자연스러움
                if step_in.agent_id in agent_ids:
                    st.session_state["active_agent_sel"] = step_in.agent_id
                st.session_state["impact_queue"] = list(step_in.impacts)
                st.success(f"Loaded sample #{idx} into queue.")
        else:
            st.caption("Manager 생성 시 샘플이 채워져.")

    # =========================================================
    # B) Impact Create (left: JSON->UI | right: UI Form)
    # =========================================================
    st.header("Impact Create")
    col_json, col_form = st.columns([1, 1], gap="large")

    with col_json:
        st.subheader("1) JSON → UI")
        example_json = {
            "target": selected_agent,
            "magnitude": 1.0,
            "duration": 1,
            "axis_weights": {space.axes[0].key: 1.0},
        }
        st.caption("JSON을 붙여넣고 Load를 누르면 오른쪽 UI Form에 반영돼. 이후 우측에서 수정 가능.")
        impact_json_text = st.text_area(
            "impact json",
            value=json.dumps(example_json, ensure_ascii=False, indent=2),
            height=220,
            key="impact_json_text",
        )

        j1, j2 = st.columns(2)
        with j1:
            if st.button("Load JSON into UI", use_container_width=True):
                try:
                    payload = json.loads(impact_json_text)
                    _impact_json_to_form(space, agent_ids, payload)
                    st.rerun()
                except Exception as e:
                    st.error(f"JSON load failed: {e}")
        with j2:
            payload = _current_form_as_json(selected_agent, space)
            st.download_button(
                "Download impact.json",
                data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="impact.json",
                mime="application/json",
                use_container_width=True,
            )

    with col_form:
        st.subheader("2) UI Form")

        # --- target init (before widget) ---
        if st.session_state["impact_target_sel"] is None or st.session_state["impact_target_sel"] not in agent_ids:
            st.session_state["impact_target_sel"] = selected_agent

        impact_target_sel = st.selectbox(
            "target",
            agent_ids,
            index=agent_ids.index(st.session_state["impact_target_sel"]) if st.session_state["impact_target_sel"] in agent_ids else 0,
            key="impact_target_sel",
        )

        impact_magnitude = st.slider(
            "magnitude",
            0.0,
            10.0,
            float(st.session_state.get("impact_magnitude", 1.0)),
            0.05,
            key="impact_magnitude",
        )
        impact_duration = st.slider(
            "duration",
            1,
            50,
            int(st.session_state.get("impact_duration", 1)),
            1,
            key="impact_duration",
        )

        st.caption("축 체크 + sign(-1~+1) + weight(>=0) → direction 생성 (proto_vec 합성 후 normalize)")

        for axis in space.axes:
            k = axis.key
            st.session_state["axis_enabled"].setdefault(k, False)
            st.session_state["axis_sign"].setdefault(k, 0.0)
            st.session_state["axis_weight"].setdefault(k, 1.0)

            c1, c2, c3 = st.columns([1, 2, 2])
            with c1:
                st.session_state["axis_enabled"][k] = st.checkbox(
                    k,
                    value=bool(st.session_state["axis_enabled"][k]),
                    key=f"en_{k}",
                )
            with c2:
                st.session_state["axis_sign"][k] = st.slider(
                    f"sign {k}",
                    -1.0,
                    1.0,
                    float(st.session_state["axis_sign"][k]),
                    0.05,
                    key=f"sg_{k}",
                )
            with c3:
                st.session_state["axis_weight"][k] = st.number_input(
                    f"weight {k}",
                    0.0,
                    100.0,
                    float(st.session_state["axis_weight"][k]),
                    0.1,
                    key=f"wt_{k}",
                )

        direction = _form_to_direction(space)
        st.write(
            {
                "direction_norm": float(np.linalg.norm(direction)),
                "direction_preview": [float(x) for x in direction[: min(8, space.dim)]],
            }
        )

        if st.button("Add Impact to Queue", use_container_width=True):
            if float(np.linalg.norm(direction)) < 1e-12:
                st.warning("direction is zero. Select at least one axis with non-zero effective weight.")
            else:
                imp = Impact(
                    direction=direction,
                    magnitude=float(impact_magnitude),
                    duration=int(impact_duration),
                    delta_vars={},  # vars 보류
                    profile={},     # meta/profile 보류
                )
                st.session_state["impact_queue"].append(imp)

                # # 타겟이 다르면 active agent 변경이 직관적
                # if impact_target_sel in agent_ids:
                #     st.session_state["active_agent_sel"] = impact_target_sel
                #     st.rerun()

                st.success("Queued.")

    # =========================================================
    # D) Step + History
    # =========================================================
    st.header("Step + History")

    c_step1, c_step2 = st.columns([1, 1], gap="large")
    with c_step1:
        if st.button("Step (apply queue)", use_container_width=True):
            step_input = StepInput(
                agent_id=selected_agent,
                impacts=st.session_state["impact_queue"],
                metadata={"src": "ui"},
            )
            out: StepResult = mgr.step(step_input)
            st.session_state["history"].append(out)
            st.session_state["impact_queue"] = []
            st.success(f"Stepped: {out.step_idx}")

    with c_step2:
        if st.button("Reset History", use_container_width=True):
            st.session_state["history"] = []
            st.success("History cleared.")

    hist: list[StepResult] = st.session_state["history"]
    if not hist:
        st.info("No history yet.")
        return

    st.write(f"history length: {len(hist)}")
    idx = st.slider("history index", min_value=1, max_value=len(hist), value=len(hist), step=1, key="hist_idx")
    step_result = hist[int(idx) - 1]

    st.subheader(f"StepResult #{step_result.step_idx} (agent={step_result.agent_id})")

    mcols = st.columns(3)
    mcols[0].metric("distance_to_comfort", f'{float(step_result.metrics.get("distance_to_comfort", 0.0)):.4f}')
    mcols[1].metric("in_comfort", str(bool(step_result.metrics.get("in_comfort", False))))
    mcols[2].metric("delta_vec_norm", f'{float(step_result.metrics.get("delta_vec_norm", 0.0)):.4f}')

    with st.expander("StepResult JSON", expanded=False):
        st.json(step_result.to_dict())


# streamlit entry
if __name__ == "__main__":
    render()
