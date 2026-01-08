# sim_engine/test/streamlit_sim_engine.py
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List

import streamlit as st

from sim_engine.dto import AgentState, Impact, TurnInput
from sim_engine.sim_agent import SimAgent
from sim_engine.sim_config import SimulationConfig
from sim_engine.sim_runner import TurnSimulation


def parse_vector(text: str) -> List[float]:
    """
    "0.1, 0.2, -0.3" 같은 입력을 벡터로 파싱
    """
    try:
        parts = [p.strip() for p in text.split(",")]
        return [float(p) for p in parts if p]
    except Exception as e:
        raise ValueError(f"Invalid vector text: {e}")


def render():
    st.set_page_config(page_title="sim_engine test", layout="wide")
    st.title("🧠 sim_engine (A안) - Vector Impact Simulator")

    # ---------- session init ----------
    if "sim" not in st.session_state:
        dim = 8
        init_state = AgentState(
            comfort_vec=[0.0] * dim,
            comfort_radius=1.0,
            current_vec=[0.0] * dim,
            vars={},
        )
        agent = SimAgent(state=init_state)
        cfg = SimulationConfig(damping=None, noise_std=0.0, normalize_vec=False, clamp_norm=None)
        st.session_state.sim = TurnSimulation(agent=agent, cfg=cfg, seed=42)
        st.session_state.results = []

    sim: TurnSimulation = st.session_state.sim

    # ---------- sidebar (config/state) ----------
    with st.sidebar:
        st.header("⚙️ Config")
        damping = st.number_input("damping (None이면 미사용)", value=float(sim.cfg.damping) if sim.cfg.damping is not None else 1.0)
        use_damping = st.checkbox("use damping", value=sim.cfg.damping is not None)

        noise_std = st.number_input("noise_std", value=float(sim.cfg.noise_std), min_value=0.0)
        normalize_vec = st.checkbox("normalize_vec", value=bool(sim.cfg.normalize_vec))

        clamp_norm = st.number_input("clamp_norm (0이면 미사용)", value=float(sim.cfg.clamp_norm) if sim.cfg.clamp_norm is not None else 0.0, min_value=0.0)

        if st.button("✅ Apply Config", use_container_width=True):
            sim.cfg = SimulationConfig(
                damping=float(damping) if use_damping else None,
                noise_std=float(noise_std),
                normalize_vec=bool(normalize_vec),
                clamp_norm=float(clamp_norm) if clamp_norm > 0 else None,
            )
            st.success("Config 적용됨")

        st.divider()
        st.header("🧍 Current State")
        st.json(sim.agent.snapshot())

    # ---------- main input ----------
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📥 Impact Input")
        dim = sim.agent.dim
        default_vec = ", ".join(["0.0"] * dim)
        vec_text = st.text_input("impact vec (comma separated)", value=default_vec)
        scale = st.number_input("scale", value=1.0)
        source = st.text_input("source/meta (optional)", value="manual")

        vars_json = st.text_area("vars_delta (optional, JSON)", value="{}")

        run_btn = st.button("▶ Step", use_container_width=True)

        if run_btn:
            try:
                vec = parse_vector(vec_text)
                if len(vec) != dim:
                    st.error(f"dim mismatch: expected {dim}, got {len(vec)}")
                else:
                    try:
                        vars_delta = json.loads(vars_json) if vars_json.strip() else {}
                    except Exception:
                        vars_delta = {}

                    impact = Impact(vec=vec, scale=float(scale), meta={"source": source, "vars_delta": vars_delta})
                    ti = TurnInput(impacts=[impact], meta={"ui": "streamlit"})
                    result = sim.step(ti)
                    st.session_state.results.append(result)
                    st.success(f"Turn {result.turn} 완료")
            except Exception as e:
                st.exception(e)

    with c2:
        st.subheader("🧾 Last Result")
        if st.session_state.results:
            last = st.session_state.results[-1]
            st.write(f"turn: **{last.turn}**")
            st.write(f"in_comfort: **{last.in_comfort}**")
            st.write(f"distance_to_comfort: **{last.distance_to_comfort:.4f}**")
            st.write("applied_delta:")
            st.code(last.applied_delta)
            st.write("metrics:")
            st.json(last.metrics)
        else:
            st.info("아직 결과 없음")

    st.divider()

    # ---------- history table ----------
    st.subheader("📚 History")
    results = st.session_state.results
    if not results:
        st.caption("step을 실행하면 여기에 턴 기록이 쌓여.")
        return

    # 간단 테이블
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "turn": r.turn,
                "in_comfort": r.in_comfort,
                "distance": r.distance_to_comfort,
                "delta_norm": r.metrics.get("delta_norm", 0.0),
                "vars_delta": r.metrics.get("vars_delta", {}),
            }
        )
    st.dataframe(rows, use_container_width=True)

    # 상세 보기
    idx = st.number_input("detail index (0..n-1)", min_value=0, max_value=len(results) - 1, value=len(results) - 1)
    r = results[int(idx)]
    st.markdown("### 🔍 Detail")
    st.write("before:")
    st.json(asdict(r.before))
    st.write("after:")
    st.json(asdict(r.after))
    st.write("applied_impacts:")
    st.json([asdict(x) for x in r.applied_impacts])
    st.write("meta:")
    st.json(r.meta)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("🧹 Reset Run", use_container_width=True):
            # reset sim
            dim = sim.agent.dim
            sim.agent.state = AgentState(
                comfort_vec=[0.0] * dim,
                comfort_radius=1.0,
                current_vec=[0.0] * dim,
                vars={},
            )
            st.session_state.results = []
            st.success("초기화 완료")

    with b2:
        if st.button("📦 Export Results JSON", use_container_width=True):
            data = [asdict(x) for x in results]
            st.download_button(
                "Download results.json",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="results.json",
                mime="application/json",
                use_container_width=True,
            )
