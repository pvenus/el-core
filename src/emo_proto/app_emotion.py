from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emo_proto.page_editor import render as render_editor
from emo_proto.page_heatmap import render as render_heatmap
from emo_proto.page_compare import render as render_compare
from emo_proto.page_simulation import render as render_simulation  # ✅ 추가

try:
    from llm_emb.emb_model import LLMEmbeddingModel
except Exception:
    LLMEmbeddingModel = None

import hashlib
import numpy as np


class DummyEmbeddingModel:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_text(self, text: str):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(self.dim,)).astype(float)
        return {"pooled": v.tolist(), "dim": self.dim}


@st.cache_resource
def get_model(use_dummy: bool, model_path: str, dummy_dim: int):
    if use_dummy or LLMEmbeddingModel is None:
        return DummyEmbeddingModel(dim=int(dummy_dim))
    return LLMEmbeddingModel(model_path)


def main():
    st.set_page_config(page_title="EL-Core · Emotion", layout="wide")

    st.sidebar.title("EL-Core")
    st.sidebar.caption("Emotion module (emo_proto)")

    with st.sidebar.expander("Global Filters", expanded=False):
        data_path = st.text_input("emotions.json path", value="data/emotions.json")
        use_dummy = st.toggle("Use dummy embedding", value=False)
        dummy_dim = st.number_input("dummy dim", min_value=64, max_value=4096, value=384, step=64)
        model_path = st.text_input("embedding model path", value="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")

    model = get_model(use_dummy=use_dummy, model_path=model_path, dummy_dim=int(dummy_dim))

    st.sidebar.divider()

    pages = [
        ("Editor", "editor", render_editor),
        ("Heatmap", "heatmap", render_heatmap),
        ("Compare", "compare", render_compare),
        ("Simulation", "simulation", render_simulation),  # ✅ 추가
    ]

    if "emotion_page" not in st.session_state:
        st.session_state["emotion_page"] = "editor"

    for title, key, _render in pages:
        active = (st.session_state["emotion_page"] == key)
        if st.sidebar.button(title, use_container_width=True, type=("primary" if active else "secondary")):
            st.session_state["emotion_page"] = key

    page_key = st.session_state["emotion_page"]
    for _, key, _render in pages:
        if key == page_key:
            _render(model=model, data_path=data_path)
            break


if __name__ == "__main__":
    main()
