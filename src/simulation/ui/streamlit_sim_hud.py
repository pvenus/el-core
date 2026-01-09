# sim_engine/test/streamlit_sim_engine.py
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import streamlit as st

# --- sim_engine imports (현재 브랜치 구조에 맞게 필요 시 경로만 조정) ---
from simulation.sim_agent import SimAgent
from simulation.sim_runner import SimRunner
from simulation.dto.agent import AgentSpec, AgentState
from simulation.dto.impact import Impact
from simulation.dto.turn import TurnInput, TurnResult


# -----------------------------
# UI helpers
# -----------------------------
def _parse_axis_keys(text: str) -> List[str]:
    keys = []
    for ln in (text or "").splitlines():
        k = ln.strip()
        if k:
            keys.append(k)
    return keys


def _validate_axes(keys: List[str]) -> Tuple[bool, str]:
    if not keys:
        return False, "axes가 비어있어. 줄바꿈으로 축 이름을 입력해줘."
    if len(set(keys)) != len(keys):
        dup = [k for k in set(keys) if keys.count(k) > 1]
        return False, f"axes key 중복: {dup}"
    return True, ""


def _scale_to_01(x: float, vmin: float = -1.0, vmax: float = 1.0) -> float:
    if vmax <= vmin:
        return 0.0
    y = (x - vmin) / (vmax - vmin)
    return max(0.0, min(1.0, y))


def _colored_delta(delta: float, eps: float = 1e-9) -> str:
    if abs(delta) < eps:
        return "<span style='color:#888'>(±0)</span>"
    if delta > 0:
        return f"<span style='color:#1a7f37'>(+{delta:.3f})</span>"
    return f"<span style='color:#b42318'>({delta:.3f})</span>"


def _safe_json(text: str) -> Dict:
    if not (text or "").strip():
        return {}
    return json.loads(text)


def _unit_dir_for_axis(dim: int, axis_idx: int) -> List[float]:
    v = [0.0] * dim
    v[axis_idx] = 1.0
    return v


def _default_agent_for_dim(dim: int) -> SimAgent:
    # comfort_vec은 일단 0벡터(형이 prototype은 나중에 채운다고 했으니)
    spec = AgentSpec(
        dim=dim,
        comfort_vec=[0.0] * dim,
        comfort_radius=1.0,
        vars={"energy": 1.0, "stamina": 1.0, "stress": 0.0},
        meta={},
    )
    state = AgentState(
        turn=0,
        current_vec=[0.0] * dim,
        vars=dict(spec.vars),
    )
    return SimAgent(spec=spec, state=state)


def _impact_id(imp: Impact) -> str:
    # profile 안에 action_id 같은게 있을 수도 있으니 표시용으로 뽑아줌
    prof = getattr(imp, "profile", {}) or {}
    for k in ("action_id", "id", "name", "key"):
        if k in prof:
            return str(prof[k])
    return "impact"


# -----------------------------
# Streamlit render
# -----------------------------
def render():
    st.set_page_config(page_title="EL-Core Sim HUD", layout="wide")

    # ---------- init session state ----------
    if "axes_text" not in st.session_state:
        st.session_state.axes_text = "Joy\nAnger\nFear\nCalm"
    if "agent" not in st.session_state:
        agent = _default_agent_for_dim(dim=len(_parse_axis_keys(st.session_state.axes_text)))
        st.session_state.agent = agent
        st.session_state.runner = SimRunner(agent)
        st.session_state.history: List[TurnResult] = []
    if "pending_impacts" not in st.session_state:
        st.session_state.pending_impacts = []  # type: ignore

    # ---------- left sidebar: space axes ----------
    with st.sidebar:
        st.header("🧠 Emotion Space Spec")
        axes_text = st.text_area(
            "Axes keys (one per line)  —  *줄 수 = dim*",
            value=st.session_state.axes_text,
            height=160,
        )

        keys = _parse_axis_keys(axes_text)
        ok, msg = _validate_axes(keys)
        if not ok:
            st.error(msg)
        else:
            st.caption(f"dim = {len(keys)}  |  unique keys ✅")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Apply Axes (Reset)", disabled=not ok, use_container_width=True):
                st.session_state.axes_text = axes_text
                dim = len(keys)
                st.session_state.agent = _default_agent_for_dim(dim=dim)
                st.session_state.runner = SimRunner(st.session_state.agent)
                st.session_state.history = []
                st.session_state.pending_impacts = []
                st.rerun()
        with col_b:
            if st.button("Reset Only State", use_container_width=True):
                # axes는 유지하고 state/runner/history만 리셋
                st.session_state.axes_text = axes_text
                dim = max(1, len(keys))
                st.session_state.agent = _default_agent_for_dim(dim=dim)
                st.session_state.runner = SimRunner(st.session_state.agent)
                st.session_state.history = []
                st.session_state.pending_impacts = []
                st.rerun()

        st.divider()
        st.header("⚡ Add Impact")
        if ok:
            dim = len(keys)

            add_mode = st.radio(
                "Direction input mode",
                ["By Axis (+1 on one axis)", "Raw Vector (JSON list)"],
                horizontal=False,
            )

            if add_mode == "By Axis (+1 on one axis)":
                axis_key = st.selectbox("Axis", keys)
                axis_idx = keys.index(axis_key)
                direction = _unit_dir_for_axis(dim, axis_idx)
            else:
                raw = st.text_area("direction (JSON list)", value=json.dumps([0.0] * dim), height=90)
                try:
                    direction = json.loads(raw)
                    if not isinstance(direction, list) or len(direction) != dim:
                        raise ValueError("direction must be list with len=dim")
                    direction = [float(x) for x in direction]
                except Exception as e:
                    st.error(f"direction parse error: {e}")
                    direction = [0.0] * dim

            magnitude = st.slider("magnitude", 0.0, 2.0, 0.35, 0.01)
            duration = st.slider("duration (turns)", 1, 10, 3, 1)

            delta_vars_text = st.text_area(
                "delta_vars (JSON object)",
                value=json.dumps({"energy": -0.05}, ensure_ascii=False),
                height=90,
            )

            profile_text = st.text_area(
                "profile/meta (JSON object) — optional",
                value=json.dumps({"action_id": "buff_demo"}, ensure_ascii=False),
                height=70,
            )

            if st.button("Add to Pending Stack", use_container_width=True):
                try:
                    delta_vars = _safe_json(delta_vars_text)
                    profile = _safe_json(profile_text)
                    imp = Impact(
                        direction=direction,
                        magnitude=float(magnitude),
                        duration=int(duration),
                        delta_vars={k: float(v) for k, v in (delta_vars or {}).items()},
                        profile=dict(profile or {}),
                    )
                    st.session_state.pending_impacts.append(imp)
                    st.success("Impact added.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add impact: {e}")

    # ---------- compute current HUD values ----------
    axes_keys = _parse_axis_keys(st.session_state.axes_text)
    dim = max(1, len(axes_keys))

    agent: SimAgent = st.session_state.agent
    runner: SimRunner = st.session_state.runner
    history: List[TurnResult] = st.session_state.history

    turn = agent.state.turn
    dist = agent.distance_to_comfort(agent.state)
    in_c = agent.in_comfort(agent.state)

    last: Optional[TurnResult] = history[-1] if history else None
    before_vec = last.before.current_vec if last else agent.state.current_vec
    after_vec = last.after.current_vec if last else agent.state.current_vec

    # ---------- top HUD ----------
    c0, c1, c2, c3 = st.columns([2.2, 1, 1, 1])
    with c0:
        st.markdown("## EL-Core Sim HUD")
    with c1:
        st.metric("Turn", int(turn))
    with c2:
        st.metric("InComfort", "✅" if in_c else "❌")
    with c3:
        st.metric("Dist", f"{float(dist):.3f}")

    st.divider()

    # ---------- main split: Emotion gauges | Vars gauges ----------
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.subheader("🧠 Emotion Space (Gauges)")
        if len(axes_keys) != agent.spec.dim:
            st.warning(
                f"axes dim({len(axes_keys)}) != agent.dim({agent.spec.dim})  →  axes를 Apply Axes로 맞춰줘."
            )

        # show per-axis
        for i in range(min(agent.spec.dim, len(axes_keys))):
            key = axes_keys[i]
            curr = float(after_vec[i])
            prev = float(before_vec[i])
            delta = curr - prev

            comfort_comp = float(agent.spec.comfort_vec[i]) if i < len(agent.spec.comfort_vec) else 0.0

            row = st.columns([1.2, 2.2, 0.9])
            with row[0]:
                st.markdown(f"**[{key}]**")
                st.caption(f"comfort={comfort_comp:.2f}")
            with row[1]:
                # gauge (0..1) mapped from [-1,1]
                st.progress(_scale_to_01(curr, -1.0, 1.0))
                st.caption(
                    f"curr={curr:.3f}  prev={prev:.3f}  "
                    + ("" if last is None else "")
                )
            with row[2]:
                st.markdown(_colored_delta(delta), unsafe_allow_html=True)

        st.caption("Legend: progress is scaled from [-1, 1]. (나중에 축 스케일 정책 추가 가능)")

    with right:
        st.subheader("🧍 Agent Status (Gauges)")
        # vars list
        vars_curr = dict(agent.state.vars or {})
        vars_prev = dict(last.before.vars) if last else vars_curr
        vars_delta = dict(last.applied_delta_vars) if last else {}

        if not vars_curr:
            st.info("Agent vars가 비어있어. (energy/stamina/stress 같은 기본 vars 넣는 걸 추천)")
        else:
            for k in sorted(vars_curr.keys()):
                v = float(vars_curr.get(k, 0.0))
                dv = float(vars_delta.get(k, 0.0))
                cols = st.columns([1.0, 2.2, 0.9])
                with cols[0]:
                    st.markdown(f"**[{k}]**")
                with cols[1]:
                    # simple gauge: assume 0..1 for energy/stamina, 0..2 for stress etc.
                    vmax = 1.0
                    if k.lower() in ("stress",):
                        vmax = 2.0
                    st.progress(_scale_to_01(v, 0.0, vmax))
                    st.caption(f"value={v:.3f}")
                with cols[2]:
                    st.markdown(_colored_delta(dv), unsafe_allow_html=True)

    st.divider()

    # ---------- bottom split: impacts stack | change log ----------
    bleft, bright = st.columns([1.05, 1.0], gap="large")

    with bleft:
        st.subheader("⚡ Impacts (Pending + Active)")

        pending: List[Impact] = list(st.session_state.pending_impacts)
        active = agent.snapshot_active_impacts() if hasattr(agent, "snapshot_active_impacts") else []

        st.caption(f"pending={len(pending)} | active={len(active)}")

        def _impact_row(imp: Impact) -> Dict:
            return {
                "id": _impact_id(imp),
                "mag": float(getattr(imp, "magnitude", 0.0)),
                "left": int(getattr(imp, "duration", 0)),
                "delta_vars": dict(getattr(imp, "delta_vars", {}) or {}),
            }

        if pending:
            st.markdown("**Pending (will be injected on next step)**")
            st.dataframe([_impact_row(x) for x in pending], use_container_width=True, height=160)

        if active:
            st.markdown("**Active (stack in agent)**")
            st.dataframe([_impact_row(x) for x in active], use_container_width=True, height=200)

        cA, cB = st.columns(2)
        with cA:
            if st.button("Clear Pending", use_container_width=True, disabled=(not pending)):
                st.session_state.pending_impacts = []
                st.rerun()
        with cB:
            if st.button("Inject Pending Now", use_container_width=True, disabled=(not pending)):
                # inject as TurnInput impacts_new
                # 실제 적용은 step 버튼에서 하도록 하고, 여기서는 just mark
                st.info("Pending impacts will be used on next Step. (Step을 눌러 적용해줘)")

    with bright:
        st.subheader("📜 Change Log (Last Turn)")

        if last is None:
            st.info("아직 실행된 턴이 없어. Step을 눌러봐.")
        else:
            st.markdown(f"**Turn {last.turn} → {last.after.turn}**")

            # axis changes
            changed_axes = []
            for i in range(min(agent.spec.dim, len(axes_keys))):
                d = float(last.after.current_vec[i] - last.before.current_vec[i])
                if abs(d) > 1e-9:
                    changed_axes.append((axes_keys[i], d))

            if changed_axes:
                st.markdown("**Vector Δ**")
                for name, d in changed_axes[:30]:
                    st.markdown(f"- {name} {_colored_delta(d)}", unsafe_allow_html=True)
            else:
                st.caption("Vector Δ: no changes")

            # vars changes
            if last.applied_delta_vars:
                st.markdown("**vars Δ**")
                for k, dv in last.applied_delta_vars.items():
                    st.markdown(f"- {k} {_colored_delta(float(dv))}", unsafe_allow_html=True)

            # impacts applied
            if last.impacts_applied:
                st.markdown("**impacts applied**")
                for imp in last.impacts_applied[:20]:
                    st.markdown(f"- `{_impact_id(imp)}` mag={imp.magnitude:.3f} left={imp.duration}")
            else:
                st.caption("impacts applied: empty")

            with st.expander("Raw TurnResult (debug)"):
                st.json(asdict(last))

    st.divider()

    # ---------- controls ----------
    st.subheader("🎛 Controls")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1.2])
    with col1:
        if st.button("Step", use_container_width=True):
            impacts_new = list(st.session_state.pending_impacts)
            st.session_state.pending_impacts = []

            turn_input = TurnInput(
                impacts=impacts_new,
                meta={"source": "ui_step"},
            )
            res = runner.step_with_input(turn_input)
            history.append(res)
            st.session_state.history = history
            st.rerun()

    with col2:
        n = st.number_input("Auto N", min_value=1, max_value=200, value=10, step=1)
        if st.button("Run N", use_container_width=True):
            impacts_new = list(st.session_state.pending_impacts)
            st.session_state.pending_impacts = []

            for _ in range(int(n)):
                turn_input = TurnInput(impacts=impacts_new, meta={"source": "ui_run_n"})
                res = runner.step_with_input(turn_input)
                history.append(res)
                impacts_new = []  # 첫 턴만 신규 impacts 적용(원하면 유지로 바꿀 수도)
            st.session_state.history = history
            st.rerun()

    with col3:
        if st.button("Pause ▌▌", use_container_width=True, disabled=True):
            # true auto-loop는 별도 구현 필요(지금은 Run N으로 대체)
            pass

    with col4:
        if st.button("Reset", use_container_width=True):
            # axes 유지, state만 리셋
            dim = len(_parse_axis_keys(st.session_state.axes_text))
            st.session_state.agent = _default_agent_for_dim(dim=dim)
            st.session_state.runner = SimRunner(st.session_state.agent)
            st.session_state.history = []
            st.session_state.pending_impacts = []
            st.rerun()

    with col5:
        if st.button("Export HUD Snapshot", use_container_width=True):
            snapshot = {
                "axes": axes_keys,
                "agent_spec": asdict(agent.spec),
                "agent_state": asdict(agent.state),
                "history_len": len(history),
                "last_turn": asdict(last) if last else None,
                "pending_impacts": [asdict(x) for x in st.session_state.pending_impacts],
            }
            st.download_button(
                "Download snapshot.json",
                data=json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="sim_hud_snapshot.json",
                mime="application/json",
                use_container_width=True,
            )

render()