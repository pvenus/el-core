import streamlit as st
import numpy as np
from typing import List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
from llama_cpp import Llama

import ast
import operator
import time

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
    # Δ Transition (Delta Ontology) demo
    # ---------------------------------------------------------
    st.subheader("Δ Transition (Delta Ontology) demo")

    # TransitionType catalog (ontology-like metadata)
    TRANSITIONS = {
        "Recovering": {
            "allowedWhen": "energy > 0.3",
            "blockedBy": None,
            "cost": {"stamina": -5},
            "cooldown_s": 10,
            "base_speed": 1.0,
        },
        "Escalating": {
            "allowedWhen": "threat_level > 0.5",
            "blockedBy": "self_control > 0.8",
            "cost": {"stamina": -2},
            "cooldown_s": 5,
            "base_speed": 1.2,
        },
        "Catharsis": {
            "allowedWhen": "energy > 0.2",
            "blockedBy": None,
            "cost": {"stamina": -3},
            "cooldown_s": 20,
            "base_speed": 1.1,
        },
        "Determination": {
            "allowedWhen": "energy > 0.4",
            "blockedBy": "overwhelm > 0.7",
            "cost": {"stamina": -10},
            "cooldown_s": 30,
            "base_speed": 0.9,
        },
        "Breakdown": {
            "allowedWhen": "overwhelm > 0.5",
            "blockedBy": None,
            "cost": {"stamina": -4},
            "cooldown_s": 8,
            "base_speed": 1.3,
        },
        "ContemptShift": {
            "allowedWhen": "threat_level > 0.4",
            "blockedBy": "empathy > 0.7",
            "cost": {"stamina": -1},
            "cooldown_s": 15,
            "base_speed": 1.0,
        },
        "Numbing": {
            "allowedWhen": "stamina < 0.4",
            "blockedBy": None,
            "cost": {"stamina": 0},
            "cooldown_s": 30,
            "base_speed": 0.8,
        },
        "Rebound": {
            "allowedWhen": "suppression > 0.6",
            "blockedBy": "self_control > 0.85",
            "cost": {"stamina": -6},
            "cooldown_s": 15,
            "base_speed": 1.25,
        },
        "Uplift": {
            "allowedWhen": "energy > 0.25",
            "blockedBy": None,
            "cost": {"stamina": +2},
            "cooldown_s": 12,
            "base_speed": 1.0,
        },
        "ThreatFreeze": {
            "allowedWhen": "threat_level > 0.7",
            "blockedBy": None,
            "cost": {"stamina": -2},
            "cooldown_s": 8,
            "base_speed": 0.6,
        },
    }

    # Example pair -> TransitionType mapping (uses only the 10 default emotions)
    PAIR_TYPE = {
        ("Sadness", "Calmness"): "Recovering",
        ("Fear", "Calmness"): "Recovering",
        ("Anger", "Calmness"): "Recovering",

        ("Calmness", "Fear"): "Escalating",
        ("Hope", "Fear"): "Escalating",
        ("Calmness", "Anger"): "Escalating",

        ("Anger", "Calmness"): "Catharsis",
        ("Fear", "Hope"): "Catharsis",
        ("Sadness", "Hope"): "Catharsis",

        ("Fear", "Pride"): "Determination",
        ("Sadness", "Pride"): "Determination",
        ("Resignation", "Pride"): "Determination",

        ("Hope", "Sadness"): "Breakdown",
        ("Calmness", "Sadness"): "Breakdown",
        ("Pride", "Resignation"): "Breakdown",

        ("Anger", "Disgust"): "ContemptShift",
        ("Fear", "Disgust"): "ContemptShift",
        ("Pride", "Disgust"): "ContemptShift",

        ("Sadness", "Resignation"): "Numbing",
        ("Fear", "Resignation"): "Numbing",
        ("Anger", "Resignation"): "Numbing",

        ("Resignation", "Anger"): "Rebound",
        ("Sadness", "Anger"): "Rebound",
        ("Fear", "Anger"): "Rebound",

        ("Calmness", "Joy"): "Uplift",
        ("Hope", "Joy"): "Uplift",
        ("Sadness", "Joy"): "Uplift",

        ("Joy", "Fear"): "ThreatFreeze",
        ("Calmness", "Fear"): "ThreatFreeze",
        ("Pride", "Fear"): "ThreatFreeze",
    }

    # State sliders (for allowedWhen/blockedBy demo)
    # Use Streamlit session_state so we can apply a transition and keep the new values.
    st.session_state.setdefault("energy", 0.6)
    st.session_state.setdefault("stamina", 0.6)
    st.session_state.setdefault("self_control", 0.5)
    st.session_state.setdefault("empathy", 0.5)
    st.session_state.setdefault("threat_level", 0.3)
    st.session_state.setdefault("overwhelm", 0.3)
    st.session_state.setdefault("suppression", 0.2)

    colA, colB, colC = st.columns(3)
    with colA:
        energy = st.slider("energy", 0.0, 1.0, float(st.session_state["energy"]), 0.01, key="energy")
        stamina = st.slider("stamina", 0.0, 1.0, float(st.session_state["stamina"]), 0.01, key="stamina")
    with colB:
        self_control = st.slider("self_control", 0.0, 1.0, float(st.session_state["self_control"]), 0.01, key="self_control")
        empathy = st.slider("empathy", 0.0, 1.0, float(st.session_state["empathy"]), 0.01, key="empathy")
    with colC:
        threat_level = st.slider("threat_level", 0.0, 1.0, float(st.session_state["threat_level"]), 0.01, key="threat_level")
        overwhelm = st.slider("overwhelm", 0.0, 1.0, float(st.session_state["overwhelm"]), 0.01, key="overwhelm")
        suppression = st.slider("suppression", 0.0, 1.0, float(st.session_state["suppression"]), 0.01, key="suppression")

    # Select current/target emotions
    c1, c2 = st.columns(2)
    with c1:
        current_emotion = st.selectbox("current emotion", words, index=words.index("Sadness") if "Sadness" in words else 0)
    with c2:
        target_emotion = st.selectbox("target emotion", words, index=words.index("Calmness") if "Calmness" in words else 0)

    # Choose space: emotion subspace (Top-K) or full space
    use_subspace_for_delta = st.checkbox(
        "Compute Δ in emotion-only subspace (Top-K axes)",
        value=True,
        help="Recommended: Δ is computed on the same Top-K axes used for the heatmap / emotion contribution ranking.",
    )

    # Build vectors for current/target in the chosen space and compute delta
    idx_map = {w: i for i, w in enumerate(words)}
    v_cur_full = mat[idx_map[current_emotion]]
    v_tgt_full = mat[idx_map[target_emotion]]

    if use_subspace_for_delta:
        v_cur = slice_mat[idx_map[current_emotion]].astype(np.float32)
        v_tgt = slice_mat[idx_map[target_emotion]].astype(np.float32)

        # Re-normalize in subspace (norms change after slicing)
        v_cur_n = np.linalg.norm(v_cur)
        v_tgt_n = np.linalg.norm(v_tgt)
        if v_cur_n > 0:
            v_cur = v_cur / v_cur_n
        if v_tgt_n > 0:
            v_tgt = v_tgt / v_tgt_n
    else:
        v_cur = v_cur_full.astype(np.float32)
        v_tgt = v_tgt_full.astype(np.float32)

    delta = (v_tgt - v_cur).astype(np.float32)
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > 0:
        delta_hat = delta / delta_norm
    else:
        delta_hat = delta

    # Build prototype deltas per TransitionType by averaging mapped pair deltas
    # Prototypes are computed in the same space as delta_hat.
    type_to_deltas = {k: [] for k in TRANSITIONS.keys()}
    for (src, dst), tname in PAIR_TYPE.items():
        if src not in idx_map or dst not in idx_map:
            continue

        if use_subspace_for_delta:
            a = slice_mat[idx_map[src]].astype(np.float32)
            b = slice_mat[idx_map[dst]].astype(np.float32)
            an = np.linalg.norm(a)
            bn = np.linalg.norm(b)
            if an > 0:
                a = a / an
            if bn > 0:
                b = b / bn
        else:
            a = mat[idx_map[src]].astype(np.float32)
            b = mat[idx_map[dst]].astype(np.float32)

        dlt = (b - a).astype(np.float32)
        dn = float(np.linalg.norm(dlt))
        if dn > 0:
            dlt = dlt / dn
        type_to_deltas[tname].append(dlt)

    proto = {}
    for tname, dlts in type_to_deltas.items():
        if not dlts:
            continue
        p = np.mean(np.stack(dlts, axis=0), axis=0)
        pn = float(np.linalg.norm(p))
        if pn > 0:
            p = p / pn
        proto[tname] = p.astype(np.float32)

    # Score delta_hat against prototypes and pick best type
    rows = []
    best_type = None
    best_score = -1e9
    for tname, p in proto.items():
        score = float(np.dot(delta_hat, p))
        rows.append((tname, score))
        if score > best_score:
            best_score = score
            best_type = tname

    score_df = pd.DataFrame(rows, columns=["TransitionType", "cosine(Δ, proto)"]).sort_values("cosine(Δ, proto)", ascending=False)
    st.dataframe(score_df, width="stretch")

    # Ontology rule evaluation (safe, Python-3.9-compatible)
    # Supports: numbers, variables, comparisons, and/or/not, + - * /, parentheses.
    # Examples: "energy > 0.3", "threat_level > 0.5 and self_control <= 0.8", "not (stamina < 0.4)".
    def _eval_condition(expr: Optional[str], env: dict) -> bool:
        if expr is None:
            return True

        OPS = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        CMPS = {
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
        }

        def _eval(node: ast.AST) -> Any:
            if isinstance(node, ast.Expression):
                return _eval(node.body)

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float, bool)):
                    return node.value
                raise ValueError("Unsupported constant")

            if isinstance(node, ast.Name):
                if node.id in env:
                    return env[node.id]
                raise ValueError(f"Unknown variable: {node.id}")

            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    return all(bool(_eval(v)) for v in node.values)
                if isinstance(node.op, ast.Or):
                    return any(bool(_eval(v)) for v in node.values)
                raise ValueError("Unsupported boolean operator")

            if isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.Not):
                    return not bool(_eval(node.operand))
                op_fn = OPS.get(type(node.op))
                if op_fn is None:
                    raise ValueError("Unsupported unary operator")
                return op_fn(float(_eval(node.operand)))

            if isinstance(node, ast.BinOp):
                op_fn = OPS.get(type(node.op))
                if op_fn is None:
                    raise ValueError("Unsupported binary operator")
                return op_fn(float(_eval(node.left)), float(_eval(node.right)))

            if isinstance(node, ast.Compare):
                left = _eval(node.left)
                for op, comp in zip(node.ops, node.comparators):
                    right = _eval(comp)
                    cmp_fn = CMPS.get(type(op))
                    if cmp_fn is None:
                        raise ValueError("Unsupported comparator")
                    if not cmp_fn(float(left), float(right)):
                        return False
                    left = right
                return True

            raise ValueError(f"Unsupported expression node: {type(node).__name__}")

        try:
            tree = ast.parse(expr, mode="eval")
            return bool(_eval(tree))
        except Exception:
            return False

    if best_type is None:
        st.warning("No TransitionType prototype could be computed (check PAIR_TYPE mapping).")
        return

    meta = TRANSITIONS[best_type]

    env = {
        "energy": float(st.session_state["energy"]),
        "stamina": float(st.session_state["stamina"]),
        "self_control": float(st.session_state["self_control"]),
        "empathy": float(st.session_state["empathy"]),
        "threat_level": float(st.session_state["threat_level"]),
        "overwhelm": float(st.session_state["overwhelm"]),
        "suppression": float(st.session_state["suppression"]),
    }

    allowed = _eval_condition(meta.get("allowedWhen"), env)
    blocked = (meta.get("blockedBy") is not None) and _eval_condition(meta.get("blockedBy"), env)
    is_allowed = allowed and (not blocked)

    # Cooldown tracking (per TransitionType)
    st.session_state.setdefault("transition_cooldowns", {})
    cd_map = st.session_state["transition_cooldowns"]
    now_s = time.time()
    last_s = float(cd_map.get(best_type, -1e9))
    cooldown_s = float(meta.get("cooldown_s", 0.0) or 0.0)
    remaining_cd = max(0.0, (last_s + cooldown_s) - now_s)
    cooldown_ready = remaining_cd <= 0.0

    can_apply = is_allowed and cooldown_ready

    # Simple speed model: base_speed * confidence (mapped from cosine)
    confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))  # [-1,1] -> [0,1]
    speed = float(meta.get("base_speed", 1.0)) * (0.5 + 0.5 * confidence)

    st.caption(f"Cooldown: {cooldown_s:.1f}s | remaining: {remaining_cd:.1f}s | ready: {cooldown_ready}")

    apply_clicked = st.button(
        "Apply transition (apply cost + start cooldown)",
        disabled=not can_apply,
        help="Applies TRANSITIONS[best_type]['cost'] into the current state and starts cooldown.",
    )

    if apply_clicked:
        cost = meta.get("cost") or {}
        for k, dv in cost.items():
            if k in st.session_state and isinstance(st.session_state[k], (int, float)):
                st.session_state[k] = float(st.session_state[k]) + float(dv)
                st.session_state[k] = max(0.0, min(1.0, float(st.session_state[k])))

        cd_map[best_type] = now_s
        st.success(f"Applied {best_type}. Cost={cost}. Cooldown started ({cooldown_s:.1f}s).")
        st.rerun()

    st.markdown(
        "**Current state (session-backed):** "
        f"energy={float(st.session_state['energy']):.2f}, "
        f"stamina={float(st.session_state['stamina']):.2f}, "
        f"self_control={float(st.session_state['self_control']):.2f}, "
        f"empathy={float(st.session_state['empathy']):.2f}, "
        f"threat_level={float(st.session_state['threat_level']):.2f}, "
        f"overwhelm={float(st.session_state['overwhelm']):.2f}, "
        f"suppression={float(st.session_state['suppression']):.2f}"
    )

    st.markdown(
        f"""\n**Selected:** `{current_emotion} → {target_emotion}`  \n**Δ-norm:** `{delta_norm:.4f}`  \n**Best TransitionType:** `{best_type}` (cosine `{best_score:.3f}`)  \n**AllowedWhen:** `{meta.get('allowedWhen')}`  \n**BlockedBy:** `{meta.get('blockedBy')}`  \n**Cost:** `{meta.get('cost')}`  \n**Cooldown:** `{meta.get('cooldown_s')}s`  \n**Allowed?** `{is_allowed}`  \n**Suggested speed multiplier:** `{speed:.3f}`\n"""
    )

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
