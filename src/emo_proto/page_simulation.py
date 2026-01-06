from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np
import streamlit as st

from emo_proto.emotion_store import (
    load_emotions,
    list_emotions,
    score_query_against_emotions,
)

TICK_SEC_FIXED = 0.25


# -----------------------------
# Helpers
# -----------------------------
def _get_emos_in_order(data: dict) -> List[dict]:
    return list_emotions(data)


def _normalize_dist(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= eps:
        return np.ones_like(x) / max(1, x.size)
    return x / s


def _dist_from_results(emos: List[dict], results: List[dict]) -> np.ndarray:
    id_to_percent = {r["id"]: float(r["percent"]) for r in results}
    dist = np.array([id_to_percent.get(e["id"], 0.0) for e in emos], dtype=float)
    return _normalize_dist(dist)


def _render_progress_state_balanced(title: str, labels: List[str], dist: np.ndarray):
    """
    Current state:
    - 게이지 위치 고정(정렬 X)
    - emotions.json 순서대로 (0,1), (2,3) ... 좌/우 균형 배치
    """
    st.subheader(title)

    perc = (dist * 100.0).astype(float)
    n = len(labels)

    for i in range(0, n, 2):
        c1, c2 = st.columns(2, vertical_alignment="center")

        # LEFT
        with c1:
            name = labels[i]
            p = float(perc[i])
            st.markdown(f"**{name} — {p:.2f}%**")
            st.progress(min(1.0, max(0.0, p / 100.0)))

        # RIGHT (if exists)
        with c2:
            if i + 1 < n:
                name = labels[i + 1]
                p = float(perc[i + 1])
                st.markdown(f"**{name} — {p:.2f}%**")
                st.progress(min(1.0, max(0.0, p / 100.0)))
            else:
                st.empty()


def _render_dist_table_compact(title: str, labels: List[str], dist: np.ndarray, max_rows: int = 10):
    st.markdown(f"**{title}**")
    perc = (dist * 100.0).astype(float)
    order = np.argsort(-perc)
    rows = [{"emotion": labels[i], "percent": float(perc[i])} for i in order[:max_rows]]
    st.dataframe(rows, use_container_width=True, height=240)


# -----------------------------
# Simulation core
# -----------------------------
@dataclass
class SimEvent:
    text: str
    target: np.ndarray
    strength: float
    duration_sec: float
    steps_total: int
    steps_done: int = 0

    @property
    def done(self) -> bool:
        return self.steps_done >= self.steps_total

    @property
    def remaining_sec(self) -> float:
        return max(0.0, (self.steps_total - self.steps_done) * TICK_SEC_FIXED)


def _step_state_towards(
    state: np.ndarray,
    target: np.ndarray,
    *,
    duration_sec: float,
    strength: float,
    resistance: float,
) -> np.ndarray:
    duration_sec = max(1e-6, float(duration_sec))
    base_alpha = TICK_SEC_FIXED / duration_sec
    alpha = base_alpha * (1.0 - float(resistance))
    alpha = float(np.clip(alpha, 0.0, 1.0))

    s = state + alpha * float(strength) * (target - state)
    return _normalize_dist(s)


def _init_session(N: int):
    if "sim_state" not in st.session_state or st.session_state["sim_state"] is None:
        st.session_state["sim_state"] = np.ones(N, dtype=float) / max(1, N)

    if "sim_events" not in st.session_state:
        st.session_state["sim_events"] = []

    if "sim_running" not in st.session_state:
        st.session_state["sim_running"] = False

    if "sim_log" not in st.session_state:
        st.session_state["sim_log"] = []

    if "sim_last_target" not in st.session_state:
        st.session_state["sim_last_target"] = None

    if "sim_last_input" not in st.session_state:
        st.session_state["sim_last_input"] = ""


def _do_one_tick(resistance: float):
    evs: List[SimEvent] = st.session_state["sim_events"]
    if not evs:
        return

    ev = evs[0]
    st.session_state["sim_state"] = _step_state_towards(
        st.session_state["sim_state"],
        ev.target,
        duration_sec=ev.duration_sec,
        strength=ev.strength,
        resistance=float(resistance),
    )

    ev.steps_done += 1
    if ev.done:
        st.session_state["sim_log"].append(f"[DONE] '{ev.text[:60]}'")
        evs.pop(0)


def render(model, data_path: str):
    st.header("Simulation")

    data = load_emotions(data_path)
    emos = _get_emos_in_order(data)

    if not emos:
        st.info("emotions.json에 감정이 없어. Editor에서 감정을 먼저 추가해줘.")
        return

    has_any_proto = any(((e.get("prototype") or {}).get("vector") is not None) for e in emos)
    if not has_any_proto:
        st.warning("감정 prototype이 없어. Editor에서 'Compute ALL vectors' 먼저 실행해줘.")
        return

    N = len(emos)
    _init_session(N)

    labels = [e["name"] for e in emos]

    # =========================================================
    # 1) TOP: Input + Transform (근접 배치)
    # =========================================================
    st.subheader("Input → Event")
    top_l, top_r = st.columns([2, 1], vertical_alignment="top")

    with top_l:
        text = st.text_area(
            "Natural language input",
            value=st.session_state.get("sim_last_input", "He betrayed me again."),
            height=110,
        )

    with top_r:
        st.markdown("**Transform options**")
        use_centering = st.checkbox("Use centering", value=True)
        use_topk = st.checkbox("Use Top-K axes", value=True)
        topk = st.number_input("Top-K", 4, 512, 32, 4, disabled=(not use_topk))
        temperature = st.slider("Softmax temperature", 0.05, 2.0, 0.35, 0.05)
        st.caption(f"Tick fixed: **{TICK_SEC_FIXED:.2f}s**")

    st.divider()

    # =========================================================
    # 2) SAME LINE: Event controls vs Current state
    # =========================================================
    row_l, row_r = st.columns([1, 1], vertical_alignment="top")

    with row_l:
        st.subheader("Event controls")
        duration_sec = st.slider("Duration (sec)", 0.5, 8.0, 1.0, 0.25)
        strength = st.slider("Strength", 0.0, 2.0, 1.0, 0.05)
        resistance = st.slider("Resistance", 0.0, 0.95, 0.0, 0.05)

        b1, b2, b3 = st.columns(3)
        add_clicked = b1.button("Add Event", type="primary", use_container_width=True)
        play_pause_clicked = b2.button(
            "Play" if not st.session_state["sim_running"] else "Pause",
            use_container_width=True,
        )
        reset_clicked = b3.button("Reset", use_container_width=True)

        st.markdown("**Queue**")
        evs: List[SimEvent] = st.session_state["sim_events"]
        if not evs:
            st.caption("No events.")
        else:
            head = evs[0]
            st.caption(
                f"Active: '{head.text[:50]}' | {head.steps_done}/{head.steps_total} "
                f"| remaining {head.remaining_sec:.2f}s"
            )
            for e in evs[1:4]:
                st.caption(f"- {e.text[:60]}")

    with row_r:
        _render_progress_state_balanced("Current emotion state", labels, st.session_state["sim_state"])

    st.divider()

    # =========================================================
    # 3) BELOW: Added vector vs Log (좌우)
    # =========================================================
    bot_l, bot_r = st.columns([1, 1], vertical_alignment="top")

    with bot_l:
        st.subheader("Added event vector")
        last_target = st.session_state.get("sim_last_target", None)
        if last_target is None:
            st.info("Add Event를 누르면 변환된 감정 벡터가 여기 표시돼.")
        else:
            _render_dist_table_compact("Latest added event distribution (Top 10)", labels, last_target, max_rows=10)

    with bot_r:
        st.subheader("Applied log")
        logs = st.session_state["sim_log"][-60:]
        st.text("\n".join(logs) if logs else "")

    # =========================================================
    # Actions
    # =========================================================
    if reset_clicked:
        st.session_state["sim_state"] = np.ones(N, dtype=float) / max(1, N)
        st.session_state["sim_events"] = []
        st.session_state["sim_running"] = False
        st.session_state["sim_last_target"] = None
        st.session_state["sim_log"].append("[RESET] state cleared, events cleared.")
        st.rerun()

    if play_pause_clicked:
        st.session_state["sim_running"] = not st.session_state["sim_running"]
        st.session_state["sim_log"].append("[PLAY]" if st.session_state["sim_running"] else "[PAUSE]")
        st.rerun()

    if add_clicked:
        query = (text or "").strip()
        if not query:
            st.warning("입력 문장을 넣어줘.")
        else:
            st.session_state["sim_last_input"] = query

            results = score_query_against_emotions(
                data=data,
                model=model,
                query=query,
                use_multi_prototype=False,
                top_m=1,
                use_centering=use_centering,
                use_topk_axes=use_topk,
                topk=int(topk) if use_topk else 0,
                temperature=float(temperature),
            )

            if not results:
                st.error("감정 비교 결과가 없어. prototype 생성 상태를 확인해줘.")
            else:
                target = _dist_from_results(emos, results)
                st.session_state["sim_last_target"] = target

                steps_total = max(1, int(round(float(duration_sec) / float(TICK_SEC_FIXED))))
                ev = SimEvent(
                    text=query,
                    target=target,
                    strength=float(strength),
                    duration_sec=float(duration_sec),
                    steps_total=steps_total,
                    steps_done=0,
                )
                st.session_state["sim_events"].append(ev)

                perc = target * 100.0
                top_idx = np.argsort(-perc)[:5]
                top5 = ", ".join([f"{labels[i]} {perc[i]:.1f}%" for i in top_idx])

                st.session_state["sim_log"].append(
                    f"[ADD] '{query[:60]}' | dur={duration_sec:.2f}s tick={TICK_SEC_FIXED:.2f}s "
                    f"| strength={strength:.2f} resist={resistance:.2f} "
                    f"| cent={use_centering} topk={'on' if use_topk else 'off'} "
                    f"(K={int(topk) if use_topk else 'NA'}) T={temperature:.2f} | top5: {top5}"
                )

                st.success("Event added.")
                st.rerun()

    # =========================================================
    # Auto-play tick
    # =========================================================
    if st.session_state["sim_running"]:
        evs: List[SimEvent] = st.session_state["sim_events"]
        if not evs:
            st.session_state["sim_running"] = False
            st.session_state["sim_log"].append("[AUTO STOP] queue empty")
            st.rerun()
        else:
            _do_one_tick(resistance=float(resistance))
            time.sleep(TICK_SEC_FIXED)
            st.rerun()
