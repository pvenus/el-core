from __future__ import annotations

import streamlit as st

from emo_proto.emotion_store import (
    load_emotions,
    save_emotions,
    list_emotions,
    upsert_emotion,
    delete_emotion,
    set_texts_from_multiline,
    batch_compute_all_vectors,
)


def render(model, data_path: str):
    st.header("Emotion Editor")

    data = load_emotions(data_path)
    emos = list_emotions(data)

    left, right = st.columns([1, 2])

    # ---------------------------
    # Left: list + batch compute
    # ---------------------------
    with left:
        st.subheader("Emotions")

        selected_id = None
        if emos:
            labels = [f'{e["name"]} ({e["id"]})' for e in emos]
            selected_label = st.selectbox("Select emotion", labels, index=0)
            selected_id = emos[labels.index(selected_label)]["id"]
        else:
            st.info("No emotions yet. Add one on the right.")

        st.divider()
        st.subheader("Batch compute vectors (ALL emotions)")

        algo = st.selectbox("prototype algorithm", ["mean_dir", "medoid"], index=0)

        # ✅ texts vs name-only 선택 옵션
        vec_source = st.radio(
            "Vector source",
            options=["texts (fallback to name if empty)", "name only"],
            index=0,
            help="texts가 있으면 texts를 쓰거나, 아예 항상 name만 임베딩할지 선택",
        )
        vector_source = "name_only" if vec_source == "name only" else "texts_then_name"

        # ✅ 'text_vectors 저장'만 옵션으로 분리 (기본 OFF)
        store_text_vectors = st.toggle(
            "Store text_vectors cache (for multi-prototype / heatmap)",
            value=False,
            help="ON이면 text_vectors를 emotions.json에 저장(멀티프로토타입/상세 히트맵용). OFF면 prototype만 남김.",
        )

        if st.button("Compute ALL vectors", type="primary", use_container_width=True):
            try:
                # ✅ 핵심: 누를 때마다 무조건 재계산되도록 compute_text_vectors=True로 고정
                batch_compute_all_vectors(
                    data=data,
                    model=model,
                    algo=algo,
                    compute_text_vectors=True,      # <- 무조건 재계산
                    vector_source=vector_source,    # <- name_only/texts_then_name 반영
                )

                # ✅ 저장 옵션이 OFF면 text_vectors는 지움(= 멀티 기능 비활성 기본 정책)
                if not store_text_vectors:
                    for e in data.get("emotions", []):
                        e["text_vectors"] = None
                        e["text_vectors_updated_at"] = None

                save_emotions(data_path, data)
                st.success("Batch computation completed.")
                st.rerun()
            except Exception as e:
                st.error(f"Batch compute failed: {e}")

        st.caption("※ Vector source를 바꿨으면 반드시 Compute ALL vectors를 눌러서 반영해줘.")

    # ---------------------------
    # Right: editor (Save 옆에 Delete)
    # ---------------------------
    with right:
        st.subheader("Add / Update Emotion")

        existing_name = ""
        existing_texts = ""
        if selected_id:
            for e in emos:
                if e["id"] == selected_id:
                    existing_name = e.get("name", "")
                    existing_texts = "\n".join(e.get("texts") or [])
                    break

        emo_id = st.text_input("id (e.g., joy, sadness)", value=selected_id or "")
        emo_name = st.text_input("name (e.g., Joy, Sadness)", value=existing_name)

        texts_multiline = st.text_area(
            "texts (one sentence per line, unlimited)",
            value=existing_texts,
            height=260,
        )

        c1, c2, c3 = st.columns([1, 1, 3])

        if c1.button("Save emotion", type="primary"):
            if not emo_id.strip() or not emo_name.strip():
                st.error("id/name are required.")
            else:
                upsert_emotion(data, emo_id.strip(), emo_name.strip())
                set_texts_from_multiline(data, emo_id.strip(), texts_multiline)
                save_emotions(data_path, data)
                st.success("Saved.")
                st.rerun()

        # Delete는 선택된 감정 기준(안전하게)
        existing_ids = [e["id"] for e in emos]
        delete_target = selected_id or (emo_id.strip() if emo_id.strip() in existing_ids else None)

        if c2.button("Delete", type="secondary", disabled=(delete_target is None)):
            if delete_target is None:
                st.warning("삭제할 감정을 선택해줘.")
            else:
                delete_emotion(data, delete_target)
                save_emotions(data_path, data)
                st.success(f"Deleted: {delete_target}")
                st.rerun()

        st.caption("Delete는 현재 선택된 감정을 삭제해.")
