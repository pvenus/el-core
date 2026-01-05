import streamlit as st
import numpy as np
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from llama_cpp import Llama

llm = Llama(
    model_path="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    embedding=True,
)


def render_embedding():
    st.title("Embedding")

    default_words = "\n".join([
        "Joy",
        "Elation",
        "Calmness",
        "Pride",
        "Hope",
        "Sadness",
        "Anger",
        "Fear",
        "Disgust",
        "Resignation",
    ])

    words_text = st.text_area("Words (one per line)", value=default_words, height=220)
    words = [w.strip() for w in words_text.splitlines() if w.strip()]

    # Session cache keys
    cache_key_words = "emb_words"
    cache_key_mat = "emb_mat"

    # If words changed, invalidate cached embeddings
    if cache_key_words in st.session_state and st.session_state.get(cache_key_words) != words:
        st.session_state.pop(cache_key_words, None)
        st.session_state.pop(cache_key_mat, None)

    if not words:
        st.warning("Please enter at least one word.")
        return

    compute = st.button("Compute embeddings")

    # Compute (or reuse) embeddings
    if compute:
        vectors = []
        for w in words:
            vectors.append(embed_word_normalized(w, llm))

        mat = np.asarray(vectors, dtype=np.float32)  # [N, D]
        st.session_state[cache_key_words] = words
        st.session_state[cache_key_mat] = mat

    # Load cached embeddings (if available)
    mat = st.session_state.get(cache_key_mat, None)
    if mat is None:
        st.info("Click 'Compute embeddings' to generate embeddings. (Changing inputs causes a rerun, so we cache results.)")
        return

    n, d = mat.shape

    st.caption(f"Computed {n} embeddings with dimension D={d}. Values are L2-normalized (direction-only).")

    # Compute axis importance by mean absolute value
    axis_importance = np.mean(np.abs(mat), axis=0)  # [D]
    sorted_axes = np.argsort(axis_importance)[::-1]  # descending

    # Top-K axis selection by importance
    count = st.number_input(
        "Top-K axes by mean(|value|)",
        min_value=1,
        max_value=d,
        value=min(32, d),
        step=1,
    )

    count = int(count)
    top_axes = sorted_axes[:count]

    slice_mat = mat[:, top_axes]  # [N, count]

    # Table view
    df = pd.DataFrame(
        slice_mat,
        index=words,
        columns=[f"ax_{i}" for i in top_axes],
    )
    # Append mean absolute value row for readability
    mean_abs_row = np.mean(np.abs(slice_mat), axis=0)
    df.loc["MEAN_ABS"] = mean_abs_row
    st.dataframe(df, width="stretch")

    # Heatmap plot
    fig, ax = plt.subplots()
    im = ax.imshow(slice_mat, aspect="auto", interpolation="nearest")
    ax.set_title("Embedding axes heatmap (Top-K axes by mean absolute value)")
    ax.set_xlabel("Axis")
    ax.set_ylabel("Word")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)

    # X ticks: show actual axis indices (top-K)
    max_xticks = 16
    if count <= max_xticks:
        xticks = list(range(count))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(i) for i in top_axes], rotation=0)
    else:
        step = max(1, count // max_xticks)
        xticks = list(range(0, count, step))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(top_axes[i]) for i in xticks], rotation=0)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

    # ---------------------------------------------------------
    # Event -> Emotion similarity (percent)
    # ---------------------------------------------------------
    st.subheader("Event → Emotion similarity")

    event_text = st.text_input("Event word/phrase", value="Breakup")
    temperature = st.slider("Softmax temperature", min_value=0.05, max_value=2.0, value=0.5, step=0.05)
    use_emotion_subspace = st.checkbox(
        "Use emotion-only subspace (remove low-contribution axes)",
        value=True,
        help="If enabled, similarity is computed only on the selected Top-K axes (by mean(|value|) across emotion words).",
    )

    if event_text.strip():
        event_vec_full = np.asarray(embed_word_normalized(event_text.strip(), llm), dtype=np.float32)  # [D]

        if use_emotion_subspace:
            # Keep only Top-K axes (emotion-contributing axes) and re-normalize in the subspace
            event_vec = event_vec_full[top_axes].astype(np.float32)
            mat_use = slice_mat.astype(np.float32)

            ev_norm = float(np.linalg.norm(event_vec))
            if ev_norm > 0:
                event_vec = event_vec / ev_norm

            # Re-normalize each emotion vector in the subspace (important: norms change after slicing)
            mat_norms = np.linalg.norm(mat_use, axis=1, keepdims=True)
            mat_norms = np.where(mat_norms > 0, mat_norms, 1.0)
            mat_use = mat_use / mat_norms

            st.caption(f"Similarity computed in emotion subspace using Top-K axes: K={count} (re-normalized within subspace).")
        else:
            # Use full embedding space
            event_vec = event_vec_full
            mat_use = mat
            st.caption("Similarity computed in full embedding space (all axes).")

        # Cosine similarity
        cos = np.asarray([float(np.dot(event_vec, mat_use[i])) for i in range(n)], dtype=np.float32)  # [N]

        # Softmax -> percent
        t = float(temperature)
        logits = cos / max(t, 1e-6)
        logits = logits - float(np.max(logits))  # stability
        expv = np.exp(logits)
        probs = expv / (np.sum(expv) + 1e-9)
        perc = probs * 100.0

        sim_df = pd.DataFrame({
            "emotion": words,
            "cosine": cos,
            "percent": perc,
        }).sort_values("percent", ascending=False).reset_index(drop=True)

        st.dataframe(sim_df, width="stretch")

        # Bar chart (percent)
        fig2, ax2 = plt.subplots()
        ax2.barh(sim_df["emotion"].tolist()[::-1], sim_df["percent"].tolist()[::-1])
        ax2.set_xlabel("Percent (%)")
        ax2.set_title("Event → Emotion distribution (softmax over cosine)")
        st.pyplot(fig2)
    else:
        st.info("Enter an event word/phrase to compute emotion similarity.")

# ---------------------------------------------------------
# Single word embedding (normalized)
# ---------------------------------------------------------

def embed_word_normalized(word: str, llm) -> List[float]:
    """
    Embed a single word and L2-normalize it so each value lies in [-1, 1].
    - If token-level embeddings are returned, apply mean pooling.
    - Output vector represents direction only.
    """
    emb = llm.embed(word)

    # Token-level embeddings -> mean pooling
    if isinstance(emb, list) and emb and isinstance(emb[0], (list, tuple)):
        vec = np.mean(np.asarray(emb, dtype=np.float32), axis=0)
    else:
        vec = np.asarray(emb, dtype=np.float32)

    # L2 normalization (direction only)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()
