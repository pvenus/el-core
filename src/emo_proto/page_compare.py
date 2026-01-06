from __future__ import annotations

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from emotion_store import (
    load_emotions,
    score_query_against_emotions,
)


def render(model, data_path: str):
    st.header("Compare (Event → Emotion similarity)")

    data = load_emotions(data_path)

    st.subheader("Event word/phrase")
    query = st.text_input(" ", value="oh my god?!?")

    st.subheader("Options")
    col1, col2 = st.columns([1, 1])

    with col1:
        temperature = st.slider("Softmax temperature", 0.05, 2.0, 0.35, 0.05)

        # ✅ 기본 OFF
        use_multi = st.checkbox("Use multi-prototype (texts vectors, Top-M pooling)", value=False)
        top_m = st.number_input("Top-M (only when multi-prototype)", 1, 20, 3, 1, disabled=(not use_multi))

        use_centering = st.checkbox("Use centering (subtract global mean μ)", value=True)

    with col2:
        use_subspace = st.checkbox("Use emotion-only subspace (Top-K axes)", value=True)
        topk = st.number_input("Top-K axes", 4, 512, 32, 4)
        st.caption("subspace: compute Top-K axes by mean(|value|) and re-normalize within subspace")

    run = st.button("Compute similarity")

    if run:
        try:
            results = score_query_against_emotions(
                data=data,
                model=model,
                query=query,
                use_multi_prototype=use_multi,
                top_m=int(top_m),
                use_centering=use_centering,
                use_topk_axes=use_subspace,
                topk=int(topk),
                temperature=float(temperature),
            )
        except Exception as e:
            st.error(f"Failed: {e}")
            return

        if not results:
            st.warning("No results. 먼저 Editor에서 'Compute ALL vectors'를 실행해줘.")
            return

        st.caption(
            f"Computed in {'emotion subspace' if use_subspace else 'full space'} "
            f"(Top-K={topk if use_subspace else 'N/A'}), temperature={temperature}."
        )

        st.subheader("Similarity table")
        st.dataframe(
            [
                {
                    "emotion": f'{r["name"]} ({r["id"]})',
                    "cosine": round(r["cosine"], 4),
                    "percent": round(r["percent"], 4),
                    "n_candidates": r.get("n_candidates", 1),
                }
                for r in results
            ],
            use_container_width=True,
        )

        st.subheader("Distribution (softmax over cosine)")

        # percent 기준 내림차순 정렬
        labels = [r["name"] for r in results]
        values = [float(r["percent"]) for r in results]

        order = np.argsort(values)  # 오름차순
        labels_sorted = [labels[i] for i in order]
        values_sorted = [values[i] for i in order]

        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(labels_sorted))))

        ax.barh(labels_sorted, values_sorted)

        ax.set_xlabel("Percent (%)")
        ax.set_xlim(0, max(values_sorted) * 1.1)

        for i, v in enumerate(values_sorted):
            ax.text(v + 0.1, i, f"{v:.2f}%", va="center", fontsize=9)

        st.pyplot(fig)
