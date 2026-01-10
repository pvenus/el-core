#
# Shared embedding utility (single source of truth)
from src.llm.embedding import embed_text_normalized
import streamlit as st
import numpy as np
from typing import List, Optional, Any, Dict
#
from llama_cpp import Llama, llama_print_system_info

import os
import json





# Llama loader (cached)
@st.cache_resource
def get_llm() -> Llama:
    #return Llama(
    #    model_path="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    #    embedding=True,
    #    n_gpu_layers=-1,
    #)

    return Llama(
            model_path="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            n_ctx=int(2048),
            n_threads=1,
            n_gpu_layers=0,
            embedding=True,
            verbose=False,
        )


#
# --------------------------
# File-backed emotion/category helpers
# --------------------------
# File-backed stores
DEFAULT_ANCHOR_STORE = "data/ontology_anchors.json"  # 7 anchor emotions
DEFAULT_EMOTION_STORE = DEFAULT_ANCHOR_STORE  # backward compat
DEFAULT_CATEGORY_DIR = "data/ontology_categories"     # per-category accepted words

#
# File-backed emotion helpers (anchors)

def _load_emotions(path: str) -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            return []
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and isinstance(raw.get("items"), list):
            raw = raw.get("items")
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for it in raw:
            if isinstance(it, str):
                out.append({"word": it, "enabled": True, "dim": None, "embed_vec": None})
            elif isinstance(it, dict):
                w = str(it.get("word") or it.get("emotion") or "").strip()
                if not w:
                    continue
                enabled = bool(it.get("enabled", True))
                embed_vec = it.get("embed_vec")
                dim = it.get("dim")
                if not isinstance(embed_vec, list):
                    embed_vec = None
                if not isinstance(dim, int):
                    dim = None
                # If embed_vec is present but dim is not, infer dim
                if embed_vec is not None and dim is None and isinstance(embed_vec, list):
                    dim = len(embed_vec)
                out.append({"word": w, "enabled": enabled, "embed_vec": embed_vec, "dim": dim})
        return out
    except Exception:
        return []

def _save_emotions(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    normalized: List[Dict[str, Any]] = []
    for it in items:
        w = str(it.get("word", "")).strip()
        if not w:
            continue
        row: Dict[str, Any] = {"word": w, "enabled": bool(it.get("enabled", True))}
        ev = it.get("embed_vec")
        dim = it.get("dim")
        if isinstance(ev, list):
            # Always slice to dim if dim is set and valid
            if isinstance(dim, int) and dim > 0:
                ev = ev[:dim]
            row["embed_vec"] = ev
            row["dim"] = len(ev)
        elif isinstance(dim, int):
            row["dim"] = dim
        normalized.append(row)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"items": normalized}, f, ensure_ascii=False, indent=2)


# --------------------------
# Category store helpers
# --------------------------

def _safe_category_name(name: str) -> str:
    # keep it filesystem-safe
    n = str(name or "").strip()
    if not n:
        return "untitled"
    keep = []
    for ch in n:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "untitled"


def _category_path(category_name: str) -> str:
    os.makedirs(DEFAULT_CATEGORY_DIR, exist_ok=True)
    safe = _safe_category_name(category_name)
    return os.path.join(DEFAULT_CATEGORY_DIR, f"{safe}.json")


def _load_category(path: str) -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            return []
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and isinstance(raw.get("items"), list):
            raw = raw.get("items")
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for it in raw:
            if isinstance(it, str):
                out.append({"word": it, "enabled": True, "dim": None, "embed_vec": None})
            elif isinstance(it, dict):
                w = str(it.get("word") or "").strip()
                if not w:
                    continue
                enabled = bool(it.get("enabled", True))
                embed_vec = it.get("embed_vec")
                dim = it.get("dim")
                sims = it.get("sims")
                if not isinstance(embed_vec, list):
                    embed_vec = None
                if not isinstance(dim, int):
                    dim = None
                if embed_vec is not None and dim is None:
                    dim = len(embed_vec)
                if not isinstance(sims, dict):
                    sims = None
                out.append({"word": w, "enabled": enabled, "embed_vec": embed_vec, "dim": dim, "sims": sims})
        return out
    except Exception:
        return []


def _save_category(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    normalized: List[Dict[str, Any]] = []
    for it in items:
        w = str(it.get("word", "")).strip()
        if not w:
            continue
        row: Dict[str, Any] = {"word": w, "enabled": bool(it.get("enabled", True))}
        ev = it.get("embed_vec")
        dim = it.get("dim")
        sims = it.get("sims")
        if isinstance(ev, list):
            if isinstance(dim, int) and dim > 0:
                ev = ev[:dim]
            row["embed_vec"] = ev
            row["dim"] = len(ev)
        elif isinstance(dim, int):
            row["dim"] = dim
        if isinstance(sims, dict):
            row["sims"] = sims
        normalized.append(row)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"items": normalized}, f, ensure_ascii=False, indent=2)


# --------------------------
# Embedding helpers for ontology
# --------------------------
def _preview_vec(vec: Optional[List[float]], k: int = 8) -> str:
    if not isinstance(vec, list) or len(vec) == 0:
        return ""
    head = vec[:k]
    try:
        return "[" + ", ".join(f"{float(x):.4f}" for x in head) + (", ...]" if len(vec) > k else "]")
    except Exception:
        return "[... ]"


def _ensure_item_embeddings(items: List[Dict[str, Any]], llm: Llama) -> bool:
    """Ensure each item with a non-empty word has embed_vec. Returns True if any item was updated."""
    updated = False
    for it in items:
        w = str(it.get("word", "")).strip()
        if not w:
            continue
        ev = it.get("embed_vec")
        if not isinstance(ev, list) or len(ev) == 0:
            dim = it.get("dim")
            if not isinstance(dim, int) or dim <= 0:
                dim = _get_embed_dim(llm)
            vec = _embed_word_vec(w, llm, dim=dim)
            it["embed_vec"] = vec
            it["dim"] = len(vec) if isinstance(vec, list) else dim
            updated = True
    return updated


def render_ontology_lab():
    st.title("Ontology lab")

    st.caption("7대 감정(Anchor)과 카테고리 후보 단어를 파일(JSON)로 관리합니다. Anchor는 별도 파일, 카테고리별 후보는 별도 파일로 저장합니다.")

    store_path = st.text_input("Anchor emotion store path", value=DEFAULT_ANCHOR_STORE)

    # Always load on start (or when path changes)
    if st.session_state.get("emo_store_path") != store_path:
        st.session_state["emo_store_path"] = store_path
        st.session_state.pop("emo_items", None)
        st.session_state.pop("emb_words", None)
        st.session_state.pop("emb_mat", None)

    if "emo_items" not in st.session_state:
        st.session_state["emo_items"] = _load_emotions(store_path)

    items: List[Dict[str, Any]] = list(st.session_state.get("emo_items", []))

    # Seed defaults if empty
    if not items:
        seed = [
            # Tier 0
            "joy",
            "anger",
            "sad",
            "fear",
            "love",
            "aversion",
            "desire",
        ]
        items = [{"word": w, "enabled": True} for w in seed]
        _save_emotions(store_path, items)
        st.session_state["emo_items"] = items

    # Normalize
    for it in items:
        it["word"] = str(it.get("word", "")).strip()
        it["enabled"] = bool(it.get("enabled", True))

    # Ensure embeddings exist and persist once (no rerun needed)
    try:
        llm = get_llm()
        print(llama_print_system_info())
        if _ensure_item_embeddings(items, llm):
            _save_emotions(store_path, items)
            st.session_state["emo_items"] = _load_emotions(store_path)
            # invalidate matrix cache to align with persisted vectors
            st.session_state.pop("emb_words", None)
            st.session_state.pop("emb_mat", None)
    except Exception as e:
        st.warning(f"Embedding init failed: {e}")

    # -----------------------------
    # 1) Anchor emotions (7)
    # -----------------------------
    st.subheader("1) Anchor emotions (7)")

    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        add_clicked = st.button("Add", use_container_width=True)
    with c2:
        delete_clicked = st.button("Delete selected", use_container_width=True, disabled=(len(items) == 0))

    if add_clicked:
        st.session_state["emo_show_add"] = True

    # Add form (popup-like)
    if st.session_state.get("emo_show_add", False):
        with st.container(border=True):
            st.markdown("**Add Emotion Word**")
            with st.form("emo_add_form", clear_on_submit=False):
                new_word = st.text_input("word", value="")
                new_enabled = st.checkbox("enabled", value=True)
                # Input for dim
                new_dim = st.number_input("embedding dimension (dim)", min_value=1, max_value=4096, value=2048, step=1)
                ok = st.form_submit_button("OK")
                cancel = st.form_submit_button("Cancel")

            if cancel:
                st.session_state["emo_show_add"] = False
                st.rerun()

            if ok:
                w = str(new_word).strip()
                if not w:
                    st.warning("word is empty")
                else:
                    llm = get_llm()
                    ev_full = _embed_word_vec(w, llm, dim=_get_embed_dim(llm))
                    # Slice to dim before saving
                    dim = int(new_dim)
                    ev = ev_full[:dim]
                    items.append({"word": w, "enabled": bool(new_enabled), "embed_vec": ev, "dim": dim})
                    _save_emotions(store_path, items)
                    st.session_state["emo_items"] = _load_emotions(store_path)
                    st.session_state["emo_show_add"] = False
                    # invalidate embeddings cache
                    st.session_state.pop("emb_words", None)
                    st.session_state.pop("emb_mat", None)
                    st.rerun()

    # Build rows for editor
    rows = []
    for i, it in enumerate(items):
        ev = it.get("embed_vec")
        dim = it.get("dim")
        # If embed_vec exists and dim is set, slice to dim for display
        ev_disp = ev
        if isinstance(ev, list) and isinstance(dim, int) and dim > 0:
            ev_disp = ev[:dim]
        rows.append({
            "idx": i,
            "enabled": bool(it.get("enabled", True)),
            "word": str(it.get("word", "")),
            "embed_dim": (len(ev_disp) if isinstance(ev_disp, list) else 0),
            "embed_vec": _preview_vec(ev_disp),
        })

    # Explicit selector (single source of truth)
    st.session_state.setdefault("emo_selected_idx", 0)

    options = list(range(len(items)))
    if not options:
        st.info("No emotion items.")
        sel_idx = 0
    else:
        # Clamp BEFORE widget instantiation (safe to write to session_state here)
        cur = int(st.session_state.get("emo_selected_idx", 0))
        cur = max(0, min(cur, len(items) - 1))
        st.session_state["emo_selected_idx"] = cur

        sel_idx = st.selectbox(
            "Selected emotion",
            options=options,
            index=cur,
            format_func=lambda i: f"{i}: {str(items[i].get('word',''))}",
            key="emo_selected_idx",
        )

    st.caption("Selection is controlled by the 'Selected emotion' dropdown (this Streamlit version doesn't support click-to-select). Changes apply immediately.")
    edited = st.data_editor(
        rows,
        key="emo_table",
        hide_index=True,
        num_rows="fixed",
        column_config={
            "idx": st.column_config.NumberColumn("idx", disabled=True, width="small"),
            "enabled": st.column_config.CheckboxColumn("enabled", width="small"),
            "word": st.column_config.TextColumn("word", width="large"),
            "embed_dim": st.column_config.NumberColumn("embed_dim", disabled=True, width="small"),
            "embed_vec": st.column_config.TextColumn("embed_vec", disabled=True, width="large"),
        },
        use_container_width=True,
    )


    # Immediately update sel_idx after edits for downstream logic
    sel_idx = int(st.session_state.get("emo_selected_idx", 0))
    # Apply edits -> autosave & reload
    changed = False
    for i in range(min(len(items), len(edited))):
        new_enabled = bool(edited[i].get("enabled", True))
        new_word = str(edited[i].get("word", "")).strip()
        old_word = str(items[i].get("word", "")).strip()
        old_dim = items[i].get("dim")
        # Only allow editing enabled/word for now; dim is not user-editable in the table
        if new_enabled != bool(items[i].get("enabled", True)) or new_word != old_word:
            items[i]["enabled"] = new_enabled
            items[i]["word"] = new_word
            # If the word changed, refresh embedding and update dim
            if new_word and new_word != old_word:
                try:
                    llm = get_llm()
                    # Use previous dim if set, else use model default
                    dim = old_dim if isinstance(old_dim, int) and old_dim > 0 else _get_embed_dim(llm)
                    ev_full = _embed_word_vec(new_word, llm, dim=dim)
                    ev = ev_full[:dim]
                    items[i]["embed_vec"] = ev
                    items[i]["dim"] = dim
                except Exception as e:
                    st.warning(f"Embed failed for '{new_word}': {e}")
                    items[i]["embed_vec"] = None
                    items[i]["dim"] = None
            changed = True

    # Delete selected
    if delete_clicked:
        idx = int(st.session_state.get("emo_selected_idx", 0))
        if 0 <= idx < len(items):
            items.pop(idx)
            changed = True

    if changed:
        _save_emotions(store_path, items)
        st.session_state["emo_items"] = _load_emotions(store_path)
        st.session_state.pop("emb_words", None)
        st.session_state.pop("emb_mat", None)
        st.rerun()

    # ---------------------------------
    # 2) Similarity table vs. selection
    # ---------------------------------
    st.subheader("2) Similarity vs selected")

    if len(items) <= 1:
        st.info("No comparison available (need at least 2 items).")
    else:
        sel_idx = int(st.session_state.get("emo_selected_idx", 0))
        sel_idx = max(0, min(sel_idx, len(items) - 1))
        sel_item = items[sel_idx]
        sel_word = str(sel_item.get("word", "")).strip()
        sel_vec = sel_item.get("embed_vec")

        if not isinstance(sel_vec, list) or len(sel_vec) == 0:
            st.warning(f"Selected item '{sel_word or sel_idx}' has no embedding vector.")
        else:
            # Compute cosine similarity against all others.
            # (Vectors are expected to be L2-normalized already, so dot ~= cosine.)
            def _cos_sim(a: List[float], b: List[float]) -> float:
                d = min(len(a), len(b))
                if d <= 0:
                    return 0.0
                # dot product on shared prefix
                s = 0.0
                for i in range(d):
                    try:
                        s += float(a[i]) * float(b[i])
                    except Exception:
                        return 0.0
                return float(s)

            # Helper: slice and re-norm candidate vectors for category batch builder
            def _slice_and_renorm(vec: List[float], dim: int) -> List[float]:
                try:
                    arr = np.asarray(vec[:dim], dtype=np.float32)
                    n = float(np.linalg.norm(arr))
                    if n > 0:
                        arr = arr / n
                    return arr.tolist()
                except Exception:
                    return (vec[:dim] if isinstance(vec, list) else [])

            sim_rows: List[Dict[str, Any]] = []
            for j, it in enumerate(items):
                if j == sel_idx:
                    continue
                w = str(it.get("word", "")).strip()
                ev = it.get("embed_vec")
                if not isinstance(ev, list) or len(ev) == 0:
                    sim = None
                    dist = None
                    comp_dim = 0
                else:
                    comp_dim = min(len(sel_vec), len(ev))
                    sim = _cos_sim(sel_vec, ev)
                    dist = 1.0 - sim

                sim_rows.append({
                    "idx": j,
                    "enabled": bool(it.get("enabled", True)),
                    "word": w,
                    "cos_sim": (float(sim) if sim is not None else None),
                    "cos_dist": (float(dist) if dist is not None else None),
                    "comp_dim": int(comp_dim),
                    "same_dim": (isinstance(ev, list) and len(ev) == len(sel_vec)),
                })

            # Sort by similarity (desc), missing vectors last
            sim_rows.sort(key=lambda r: (-1e9 if r["cos_sim"] is None else -r["cos_sim"]))

            st.caption(f"Selected: idx={sel_idx}, word='{sel_word}', dim={len(sel_vec)}")

            st.dataframe(
                sim_rows,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "idx": st.column_config.NumberColumn("idx", width="small"),
                    "enabled": st.column_config.CheckboxColumn("enabled", width="small"),
                    "word": st.column_config.TextColumn("word", width="large"),
                    "cos_sim": st.column_config.NumberColumn("cos_sim", format="%.4f", width="small"),
                    "cos_dist": st.column_config.NumberColumn("cos_dist", format="%.4f", width="small"),
                    "comp_dim": st.column_config.NumberColumn("comp_dim", width="small"),
                    "same_dim": st.column_config.CheckboxColumn("same_dim", width="small"),
                },
            )

    # ---------------------------------
    # 3) Pairwise similarity (all A-B)
    # ---------------------------------
    st.subheader("3) Pairwise similarity (all pairs)")

    st.caption("All pairwise cosine similarities across the current emotion list (higher = more similar).")

    # Options
    only_enabled = st.checkbox("Only enabled items", value=True)
    top_k_pairs = st.number_input("Show top K pairs", min_value=10, max_value=5000, value=100, step=10)

    # Build candidate indices
    cand_idxs = []
    for i, it in enumerate(items):
        if only_enabled and not bool(it.get("enabled", True)):
            continue
        ev = it.get("embed_vec")
        if isinstance(ev, list) and len(ev) > 0:
            cand_idxs.append(i)

    if len(cand_idxs) < 2:
        st.info("Not enough items with vectors to compute pairwise similarities.")
    else:
        pair_rows: List[Dict[str, Any]] = []
        for a_pos in range(len(cand_idxs)):
            a = cand_idxs[a_pos]
            a_it = items[a]
            a_word = str(a_it.get("word", "")).strip()
            a_vec = a_it.get("embed_vec")
            if not isinstance(a_vec, list) or len(a_vec) == 0:
                continue

            for b_pos in range(a_pos + 1, len(cand_idxs)):
                b = cand_idxs[b_pos]
                b_it = items[b]
                b_word = str(b_it.get("word", "")).strip()
                b_vec = b_it.get("embed_vec")
                if not isinstance(b_vec, list) or len(b_vec) == 0:
                    continue

                comp_dim = min(len(a_vec), len(b_vec))
                sim = _cos_sim(a_vec, b_vec)
                pair_rows.append({
                    "idx_a": a,
                    "word_a": a_word,
                    "idx_b": b,
                    "word_b": b_word,
                    "cos_sim": float(sim),
                    "cos_dist": float(1.0 - float(sim)),
                    "comp_dim": int(comp_dim),
                    "same_dim": (len(a_vec) == len(b_vec)),
                })

        # Sort by similarity desc
        pair_rows.sort(key=lambda r: -r["cos_sim"])

        # Trim
        k = int(top_k_pairs)
        if k > 0 and len(pair_rows) > k:
            pair_rows = pair_rows[:k]

        st.dataframe(
            pair_rows,
            hide_index=True,
            use_container_width=True,
            column_config={
                "idx_a": st.column_config.NumberColumn("A idx", width="small"),
                "word_a": st.column_config.TextColumn("A word", width="large"),
                "idx_b": st.column_config.NumberColumn("B idx", width="small"),
                "word_b": st.column_config.TextColumn("B word", width="large"),
                "cos_sim": st.column_config.NumberColumn("cos_sim", format="%.4f", width="small"),
                "cos_dist": st.column_config.NumberColumn("cos_dist", format="%.4f", width="small"),
                "comp_dim": st.column_config.NumberColumn("comp_dim", width="small"),
                "same_dim": st.column_config.CheckboxColumn("same_dim", width="small"),
            },
        )

    # ---------------------------------------------
    # 4) Difference-vector similarity (A->B vs C->D)
    # ---------------------------------------------
    st.subheader("4) Difference-vector similarity (A→B vs C→D)")

    st.caption(
        "We treat a relation as a direction vector (A→B) = vec(B) − vec(A) and compare relations by cosine similarity. "
        "This helps find repeated/consistent transitions that can be named as a new ontology relation."
    )

    diff_only_enabled = st.checkbox("Only enabled items (diff)", value=True, key="diff_only_enabled")
    diff_sim_threshold = st.slider("Similarity threshold", min_value=-1.0, max_value=1.0, value=0.80, step=0.01)

    # Candidate items for relation vectors
    diff_idxs: List[int] = []
    for i, it in enumerate(items):
        if diff_only_enabled and not bool(it.get("enabled", True)):
            continue
        ev = it.get("embed_vec")
        if isinstance(ev, list) and len(ev) > 0:
            diff_idxs.append(i)

    if len(diff_idxs) < 2:
        st.info("Not enough items with vectors to compute relation vectors.")
    else:
        def _l2_normalize_list(v: List[float]) -> List[float]:
            try:
                arr = np.asarray(v, dtype=np.float32)
                n = float(np.linalg.norm(arr))
                if n > 0:
                    arr = arr / n
                return arr.tolist()
            except Exception:
                return v

        def _sub_vec(b: List[float], a: List[float]) -> List[float]:
            d = min(len(a), len(b))
            out = [0.0] * d
            for i in range(d):
                try:
                    out[i] = float(b[i]) - float(a[i])
                except Exception:
                    out[i] = 0.0
            return out

        # Build ordered relation vectors: (A->B) = vec(B) - vec(A)
        rels: List[Dict[str, Any]] = []
        for ai in diff_idxs:
            a_it = items[ai]
            a_word = str(a_it.get("word", "")).strip()
            a_vec = a_it.get("embed_vec")
            if not isinstance(a_vec, list) or len(a_vec) == 0:
                continue
            for bi in diff_idxs:
                if bi == ai:
                    continue
                b_it = items[bi]
                b_word = str(b_it.get("word", "")).strip()
                b_vec = b_it.get("embed_vec")
                if not isinstance(b_vec, list) or len(b_vec) == 0:
                    continue
                dv = _sub_vec(b_vec, a_vec)
                # Normalize relation direction so cosine compares direction
                dvn = _l2_normalize_list(dv)
                rels.append({
                    "a_idx": ai,
                    "a_word": a_word,
                    "b_idx": bi,
                    "b_word": b_word,
                    "rel_key": f"{a_word}→{b_word}",
                    "vec": dvn,
                    "dim": len(dvn),
                })

        if len(rels) < 2:
            st.info("Not enough relation vectors to compare.")
        else:
            # For each relation vector, count how many other relations are similar above threshold.
            rel_strength: List[Dict[str, Any]] = []
            best_matches: List[Dict[str, Any]] = []

            for i in range(len(rels)):
                vi = rels[i]["vec"]
                key_i = rels[i]["rel_key"]
                a_i = rels[i]["a_word"]
                b_i = rels[i]["b_word"]

                sims: List[float] = []
                # Track the single best match (excluding itself)
                best_j = None
                best_sim = -2.0

                for j in range(len(rels)):
                    if j == i:
                        continue
                    vj = rels[j]["vec"]
                    sim = _cos_sim(vi, vj)
                    if sim >= float(diff_sim_threshold):
                        sims.append(float(sim))
                    if sim > best_sim:
                        best_sim = float(sim)
                        best_j = j

                rel_strength.append({
                    "relation": key_i,
                    "A": a_i,
                    "B": b_i,
                    "support_cnt(>=thr)": int(len(sims)),
                    "support_mean_sim": (float(np.mean(sims)) if sims else None),
                    "best_match": (rels[best_j]["rel_key"] if best_j is not None else None),
                    "best_match_sim": float(best_sim) if best_j is not None else None,
                })

                if best_j is not None:
                    best_matches.append({
                        "rel_1": key_i,
                        "rel_2": rels[best_j]["rel_key"],
                        "cos_sim": float(best_sim),
                    })

            # Show relations that have lots of high-similarity siblings (good candidates for naming a relation)
            rel_strength.sort(key=lambda r: (-r["support_cnt(>=thr)"], -(r["support_mean_sim"] or -1e9)))

            st.markdown("**4.1 Relation candidates (many similar A→B directions)**")
            st.dataframe(
                rel_strength,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "relation": st.column_config.TextColumn("relation", width="large"),
                    "A": st.column_config.TextColumn("A", width="small"),
                    "B": st.column_config.TextColumn("B", width="small"),
                    "support_cnt(>=thr)": st.column_config.NumberColumn("support_cnt(>=thr)", width="small"),
                    "support_mean_sim": st.column_config.NumberColumn("support_mean_sim", format="%.4f", width="small"),
                    "best_match": st.column_config.TextColumn("best_match", width="large"),
                    "best_match_sim": st.column_config.NumberColumn("best_match_sim", format="%.4f", width="small"),
                },
            )

            # Show top relation-pair similarities across all relations (A→B vs C→D)
            st.markdown("**4.2 Top relation-pair similarities (A→B vs C→D)**")

            # Compute all unique relation-pair comparisons (upper triangle)
            rel_pair_rows: List[Dict[str, Any]] = []
            for i in range(len(rels)):
                vi = rels[i]["vec"]
                for j in range(i + 1, len(rels)):
                    vj = rels[j]["vec"]
                    sim = _cos_sim(vi, vj)
                    if sim < float(diff_sim_threshold):
                        continue
                    rel_pair_rows.append({
                        "rel_1": rels[i]["rel_key"],
                        "rel_2": rels[j]["rel_key"],
                        "cos_sim": float(sim),
                    })

            rel_pair_rows.sort(key=lambda r: -r["cos_sim"])

            st.dataframe(
                rel_pair_rows,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "rel_1": st.column_config.TextColumn("relation 1", width="large"),
                    "rel_2": st.column_config.TextColumn("relation 2", width="large"),
                    "cos_sim": st.column_config.NumberColumn("cos_sim", format="%.4f", width="small"),
                },
            )

            # Optional: inspect one relation and compute an average relation vector from its supports
            st.markdown("**4.3 Inspect one relation & build an averaged relation vector**")

            # Only show relations that actually have at least 1 supporter above threshold
            cand_rows = [r for r in rel_strength if int(r.get("support_cnt(>=thr)") or 0) >= 1]
            if not cand_rows:
                st.info("No relations have supporters above the current threshold. Lower the threshold (e.g., 0.75→0.70) to discover clusters.")
            else:
                # Sort by (support count desc, mean sim desc)
                cand_rows.sort(key=lambda r: (-int(r.get("support_cnt(>=thr)") or 0), -(r.get("support_mean_sim") or -1e9)))

                # Build display options with counts for easier scanning
                cand_keys = [r["relation"] for r in cand_rows]
                sel_rel = st.selectbox(
                    "Pick a relation to inspect",
                    options=cand_keys,
                    index=0,
                    key="sel_rel_inspect",
                    format_func=lambda k: next(
                        (f"{k}  (support={int(r.get('support_cnt(>=thr)') or 0)}, mean={float(r.get('support_mean_sim') or 0.0):.3f})" for r in cand_rows if r.get("relation") == k),
                        k,
                    ),
                )

                # Find the relation vector
                base = None
                for r in rels:
                    if r["rel_key"] == sel_rel:
                        base = r
                        break

                if base is None:
                    st.info("Selected relation not found.")
                else:
                    base_vec = base["vec"]
                    # Gather supporters above threshold
                    supporters: List[Dict[str, Any]] = []
                    for r in rels:
                        if r["rel_key"] == sel_rel:
                            continue
                        sim = _cos_sim(base_vec, r["vec"])
                        if sim >= float(diff_sim_threshold):
                            supporters.append({"relation": r["rel_key"], "cos_sim": float(sim)})

                    supporters.sort(key=lambda x: -x["cos_sim"])

                    if not supporters:
                        st.info("No supporting relations above threshold for this relation.")
                    else:
                        # Average relation vector (base + supporters)
                        vecs = [np.asarray(base_vec, dtype=np.float32)]
                        for s in supporters:
                            # find supporter vec
                            for r in rels:
                                if r["rel_key"] == s["relation"]:
                                    vecs.append(np.asarray(r["vec"], dtype=np.float32))
                                    break
                        avg = np.mean(np.stack(vecs, axis=0), axis=0)
                        n = float(np.linalg.norm(avg))
                        if n > 0:
                            avg = avg / n

                        st.caption(f"Supporters count (>=thr): {len(supporters)}")
                        st.dataframe(
                            supporters[: min(200, len(supporters))],
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "relation": st.column_config.TextColumn("relation", width="large"),
                                "cos_sim": st.column_config.NumberColumn("cos_sim", format="%.4f", width="small"),
                            },
                        )

                        # Show avg vector preview (head)
                        avg_list = avg.astype(np.float32).tolist()
                        st.text(f"Averaged relation vector head: {_preview_vec(avg_list, k=12)}")

    # ---------------------------------
    # 5) Category builder (batch input)
    # ---------------------------------
    st.subheader("5) Category builder (batch input)")

    st.caption(
        "여러 단어 리스트를 한 번에 입력하고, 각 단어가 7대 Anchor 감정 모두와 유사도 임계치 이상일 때만 채택하여 카테고리 파일로 저장합니다.\n"
        "입력 포맷 예시:\n"
        "[exploration]\ncuriosity\ninterest\n...\n\n"
        "[social]\nrespect\ntrust\n...\n"
    )

    # Build anchor set (enabled only)
    anchors = [(str(it.get("word", "")).strip(), it.get("embed_vec")) for it in items if bool(it.get("enabled", True))]
    anchors = [(w, v) for (w, v) in anchors if w and isinstance(v, list) and len(v) > 0]

    if len(anchors) < 1:
        st.info("No enabled anchors with vectors. Enable anchors first.")
        return

    # Use a shared comparison dimension across anchors
    anchor_dim = min(len(v) for _, v in anchors)

    cA, cB, cC = st.columns([2, 2, 6])
    with cA:
        default_cat = st.text_input("Default category name", value="default")
    with cB:
        thr_all = st.slider("All-anchor similarity threshold", min_value=-1.0, max_value=1.0, value=0.70, step=0.01)
    with cC:
        st.write(f"Anchor count: {len(anchors)} / comp_dim: {anchor_dim}")

    raw_batch = st.text_area(
        "Paste candidate lists (multiple categories allowed)",
        value="[default]\n\n",
        height=200,
        help="Use [category] headers. Words can be separated by newlines, commas, or semicolons.",
    )

    eval_clicked = st.button("Evaluate & Save", use_container_width=True)

    def _parse_batch(text: str, fallback_cat: str) -> Dict[str, List[str]]:
        lines = [ln.rstrip() for ln in (text or "").splitlines()]
        cat = str(fallback_cat or "default").strip() or "default"
        out: Dict[str, List[str]] = {cat: []}

        def _push_words(target_cat: str, chunk: str) -> None:
            if not chunk:
                return
            # split by comma/semicolon as well
            parts: List[str] = []
            for p in chunk.replace(";", ",").split(","):
                w = p.strip()
                if w:
                    parts.append(w)
            if not parts:
                return
            out.setdefault(target_cat, [])
            out[target_cat].extend(parts)

        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            # category header: [name]
            if s.startswith("[") and s.endswith("]") and len(s) >= 3:
                cat = s[1:-1].strip() or cat
                out.setdefault(cat, [])
                continue
            # allow: category: name
            if s.lower().startswith("category:"):
                cat = s.split(":", 1)[1].strip() or cat
                out.setdefault(cat, [])
                continue
            _push_words(cat, s)

        # de-dup preserving order
        for k in list(out.keys()):
            seen = set()
            dedup = []
            for w in out[k]:
                ww = str(w).strip()
                if not ww or ww in seen:
                    continue
                seen.add(ww)
                dedup.append(ww)
            out[k] = dedup
        # remove empty categories
        out = {k: v for k, v in out.items() if v}
        return out

    if eval_clicked:
        llm = get_llm()

        cat_map = _parse_batch(raw_batch, default_cat)
        if not cat_map:
            st.warning("No candidate words found in the input.")
        else:
            # Build a cache of embeddings to avoid repeated calls
            unique_words: List[str] = []
            seen = set()
            for ws in cat_map.values():
                for w in ws:
                    if w not in seen:
                        seen.add(w)
                        unique_words.append(w)

            emb_cache: Dict[str, List[float]] = {}
            for w in unique_words:
                try:
                    ev_full = _embed_word_vec(w, llm, dim=_get_embed_dim(llm))
                    ev = _slice_and_renorm(ev_full, anchor_dim)
                    emb_cache[w] = ev
                except Exception:
                    emb_cache[w] = []

            # Evaluate and save per category
            summary_rows = []
            for cat_name, words in cat_map.items():
                accepted: List[Dict[str, Any]] = []
                rejected: List[Dict[str, Any]] = []

                for w in words:
                    v = emb_cache.get(w) or []
                    if not isinstance(v, list) or len(v) == 0:
                        rejected.append({"word": w, "reason": "no_embedding"})
                        continue

                    sims: Dict[str, float] = {}
                    ok_all = True
                    min_sim = 2.0
                    for a_name, a_vec in anchors:
                        sim = _cos_sim(a_vec[:anchor_dim], v)
                        sims[a_name] = float(sim)
                        if sim < float(thr_all):
                            ok_all = False
                        if sim < min_sim:
                            min_sim = float(sim)

                    if ok_all:
                        accepted.append({"word": w, "enabled": True, "embed_vec": v, "dim": len(v), "sims": sims, "min_sim": float(min_sim)})
                    else:
                        rejected.append({"word": w, "min_sim": float(min_sim), "sims": sims})

                # Sort accepted by min_sim desc (stronger all-around match first)
                accepted.sort(key=lambda r: -float(r.get("min_sim") or -1e9))
                rejected.sort(key=lambda r: -float(r.get("min_sim") or -1e9))

                path = _category_path(cat_name)
                _save_category(path, accepted)

                summary_rows.append({
                    "category": cat_name,
                    "file": path,
                    "input_cnt": len(words),
                    "accepted_cnt": len(accepted),
                    "rejected_cnt": len(rejected),
                })

                with st.expander(f"{cat_name} (accepted {len(accepted)}/{len(words)})", expanded=(len(cat_map) == 1)):
                    st.caption(f"Saved: {path}")
                    if accepted:
                        st.dataframe(
                            [{"word": r["word"], "min_sim": r.get("min_sim")} for r in accepted],
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.info("No accepted words for this category.")

                    if rejected:
                        st.markdown("**Rejected (top few)**")
                        st.dataframe(
                            [{"word": r["word"], "min_sim": r.get("min_sim")} for r in rejected[: min(50, len(rejected))]],
                            hide_index=True,
                            use_container_width=True,
                        )

            st.markdown("**Batch save summary**")
            st.dataframe(summary_rows, hide_index=True, use_container_width=True)

    return



# --------------------------
# Embedding helpers for ontology
# --------------------------

def _get_embed_dim(llm: Llama, fallback: int = 2048) -> int:
    # Prefer model metadata if available
    try:
        meta = getattr(llm, "metadata", None)
        if isinstance(meta, dict):
            v = meta.get("llama.embedding_length")
            if v is not None:
                return int(v)
    except Exception:
        pass
    return int(fallback)


def _embed_word_vec(word: str, llm: Llama, dim: Optional[int] = None) -> List[float]:
    """Embed using the shared util (single source of truth). Returns an L2-normalized vector."""
    w = str(word).strip()
    if not w:
        return []

    if dim is None:
        dim = _get_embed_dim(llm)

    vec = embed_text_normalized(w, int(dim), backend="llama_cpp", llm=llm)
    # Ensure plain JSON-serializable floats
    return [float(x) for x in vec]
