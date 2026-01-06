from __future__ import annotations

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from emo_proto.emotion_store import (
    load_emotions,
    list_emotions,
    l2_normalize,
    topk_axes_by_mean_abs,
)


def _row_l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """mat: [N, D] -> row-wise L2 normalize"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + eps)


def _build_matrix_from_emotions(data: dict, use_multi: bool) -> tuple[list[str], np.ndarray]:
    """
    emotions.json 기반으로 히트맵용 행렬(mat)을 만든다.

    use_multi=True  -> 각 감정의 text_vectors 전부를 row로 쌓음
    use_multi=False -> 각 감정의 prototype.vector를 row로 쌓음
    """
    labels: list[str] = []
    rows: list[np.ndarray] = []

    for e in list_emotions(data):
        emo_id = e.get("id", "?")
        name = e.get("name", emo_id)

        if use_multi:
            tv = e.get("text_vectors") or []
            for j, v in enumerate(tv):
                vv = l2_normalize(np.asarray(v, dtype=float))
                rows.append(vv)
                labels.append(f"{name} ({emo_id}) #{j+1}")
        else:
            pv = (e.get("prototype") or {}).get("vector")
            if pv is None:
                continue
            vv = l2_normalize(np.asarray(pv, dtype=float))
            rows.append(vv)
            labels.append(f"{name} ({emo_id})")

    if not rows:
        return [], np.zeros((0, 0), dtype=float)

    return labels, np.vstack(rows)


def _apply_centering(mat: np.ndarray) -> np.ndarray:
    """
    mat: [N, D] direction-only vectors
    centering:
      mu = mean(mat)
      mat' = row_norm(mat - mu)
    """
    mu = mat.mean(axis=0)
    centered = mat - mu
    centered = _row_l2_normalize(centered)
    return centered


def render(model, data_path: str):
    st.header("Heatmap (Emotion embedding axes)")

    data = load_emotions(data_path)
    emos = list_emotions(data)

    if not emos:
        st.info("emotions.json에 감정이 없어. Editor에서 감정부터 추가해줘.")
        return

    st.caption(
        "Editor에서 계산된 감정 벡터를 기반으로 Top-K axes를 뽑아 테이블/히트맵으로 보여줘.\n"
        "※ 먼저 Editor에서 'Compute ALL vectors'를 실행해서 prototype/text_vectors를 만들어야 해."
    )

    col_l, col_r = st.columns([1, 2])

    with col_l:
        # ✅ 기본 OFF 유지
        use_multi = st.toggle(
            "Use multi-prototype (text_vectors)",
            value=False,
            help="감정 texts 벡터까지 row로 펼쳐서 보고 싶을 때만 켜",
        )

        # ✅ centering 옵션 추가 (기본 ON 추천)
        use_centering = st.toggle(
            "Use centering (subtract global mean μ)",
            value=True,
            help="감정 벡터들의 공통 성분을 제거하고, 감정 간 차이를 만드는 방향을 강조",
        )

        topk = st.number_input("Top-K axes by mean(|value|)", min_value=4, max_value=512, value=32, step=4)

        refresh = st.button("Refresh from emotions.json", use_container_width=True)

    # 세션 캐시
    if "emo_heat_labels" not in st.session_state:
        st.session_state["emo_heat_labels"] = []
    if "emo_heat_mat_raw" not in st.session_state:
        st.session_state["emo_heat_mat_raw"] = None
    if "emo_heat_mat_used" not in st.session_state:
        st.session_state["emo_heat_mat_used"] = None
    if "emo_heat_use_multi" not in st.session_state:
        st.session_state["emo_heat_use_multi"] = None
    if "emo_heat_use_centering" not in st.session_state:
        st.session_state["emo_heat_use_centering"] = None

    # refresh 조건: 버튼 누르거나 옵션이 바뀌면
    need_reload = (
        refresh
        or (st.session_state["emo_heat_use_multi"] is None)
        or (st.session_state["emo_heat_use_multi"] != use_multi)
        or (st.session_state["emo_heat_use_centering"] is None)
        or (st.session_state["emo_heat_use_centering"] != use_centering)
    )

    if need_reload:
        labels, mat_raw = _build_matrix_from_emotions(data, use_multi=use_multi)
        st.session_state["emo_heat_labels"] = labels
        st.session_state["emo_heat_mat_raw"] = mat_raw

        if mat_raw is not None and mat_raw.size > 0 and use_centering:
            mat_used = _apply_centering(mat_raw)
        else:
            mat_used = mat_raw

        st.session_state["emo_heat_mat_used"] = mat_used
        st.session_state["emo_heat_use_multi"] = use_multi
        st.session_state["emo_heat_use_centering"] = use_centering

    labels = st.session_state.get("emo_heat_labels", [])
    mat_raw = st.session_state.get("emo_heat_mat_raw", None)
    mat = st.session_state.get("emo_heat_mat_used", None)

    if mat is None or mat.size == 0:
        st.warning(
            "히트맵을 만들 벡터가 없어.\n"
            "- Editor에서 감정(name/texts)을 채우고\n"
            "- 'Compute ALL vectors'를 실행해서 prototype/text_vectors를 생성한 다음\n"
            "- 다시 Heatmap에서 Refresh 눌러줘."
        )
        return

    mode = "multi(text_vectors)" if use_multi else "single(prototype)"
    cent = "centered(μ removed)" if use_centering else "raw"
    st.caption(f"Loaded {len(labels)} vectors, D={mat.shape[1]} · mode={mode} · {cent}")

    # ✅ axes는 '사용 mat'(centering 반영된 mat)에서 뽑는다
    axes = topk_axes_by_mean_abs(mat, int(topk))
    sub = mat[:, axes]
    axis_headers = [f"ax_{int(a)}" for a in axes.tolist()]

    # ---- table ----
    st.subheader("Top-K axes table")
    rows = []
    for i, lab in enumerate(labels):
        row = {"row": lab}
        for j, h in enumerate(axis_headers):
            row[h] = float(sub[i, j])
        rows.append(row)
    st.dataframe(rows, use_container_width=True)

    # ---- heatmap ----
    st.subheader("Embedding axes heatmap (Top-K axes by mean absolute value)")
    fig_h = max(4, 0.35 * len(labels))
    fig_w = min(14, 8 + 0.12 * int(topk))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(sub, aspect="auto")

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    ax.set_xticks(np.arange(len(axis_headers)))
    ax.set_xticklabels([h.replace("ax_", "") for h in axis_headers], rotation=90)

    ax.set_xlabel("Axis")
    ax.set_ylabel("Emotion vectors")
    fig.colorbar(im, ax=ax)

    st.pyplot(fig)

    if use_centering and (mat_raw is not None and mat_raw.size > 0):
        st.caption("centering: v' = normalize(v - μ), where μ is mean over all emotion vectors used in this page.")
