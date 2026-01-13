import re
from typing import Dict

# Helper: Parse scored concepts from string like "warmth(1.000), hostility(0.841)"
def parse_scored_concepts(text: str) -> Dict[str, float]:
    """
    Parse strings like:
      warmth(1.000), hostility(0.841), friendliness(0.835)
    into a {concept: score} dict.

    Rules:
    - 1.0 is treated as 100% similarity baseline
    - Scores are clamped to [0.0, 1.0]
    - Whitespace tolerant
    """
    out: Dict[str, float] = {}
    if not text:
        return out
    for name, val in re.findall(r"([a-zA-Z0-9_\-]+)\s*\(\s*([0-9]*\.?[0-9]+)\s*\)", text):
        try:
            f = float(val)
            if f < 0.0:
                f = 0.0
            if f > 1.0:
                f = 1.0
            out[name] = f
        except Exception:
            continue
    return out
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH so `import src....` works when Streamlit changes CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../el-core
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

import json
import os
from typing import Any, Dict, List

# -----------------------------
# Ontology/Concepts Helpers
# -----------------------------
DEFAULT_CATEGORY_DIR = "data/ontology_categories"

def _load_total_categories(path: str = "") -> List[Dict[str, Any]]:
    """Load items from data/ontology_categories/total.json (or given path)."""
    p = path or os.path.join(DEFAULT_CATEGORY_DIR, "total.json")
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        raw = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for it in raw:
            if isinstance(it, dict):
                w = str(it.get("word") or "").strip()
                if not w:
                    continue
                out.append({
                    "word": w,
                    "enabled": bool(it.get("enabled", True)),
                    "dim": it.get("dim"),
                })
            elif isinstance(it, str):
                w = it.strip()
                if not w:
                    continue
                out.append({"word": w, "enabled": True, "dim": None})
        return out
    except Exception:
        return []

def _dedup_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

from src.scenario.maker import ChoiceMakerPipeline

# -----------------------------
# File-backed store (JSON)
# -----------------------------
def _load_items(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        return []

    items: List[Dict[str, Any]] = []
    for it in data:
        if not isinstance(it, dict):
            continue

        # New format: full choice payload already stored
        # NOTE: OFF-mode entries may not have `choice_id`/`action` but DO have `concepts`/`embed_vec`.
        if any(k in it for k in ("choice_id", "action", "concepts", "embed_vec", "impact", "direction")):
            display_text = str(it.get("display_text", ""))
            embed_text = str(it.get("action", {}).get("embed_text", it.get("embed_text", "")))
            items.append(
                {
                    "display_text": display_text,
                    "embed_text": embed_text,
                    "enabled": bool(it.get("enabled", True)),
                    "artifact": it,
                    "round_id": int(it.get("round_id", 1) or 1),
                }
            )
            continue

        # Legacy/simple format
        items.append(
            {
                "display_text": str(it.get("display_text", "")),
                "embed_text": str(it.get("embed_text", "")),
                "enabled": bool(it.get("enabled", True)),
                "artifact": None,
                "round_id": int(it.get("round_id", 1) or 1),
            }
        )

    return items


def _save_items(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    normalized: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue

        enabled = bool(it.get("enabled", True))
        art = it.get("artifact")

        if isinstance(art, dict):
            out = dict(art)
            out["enabled"] = enabled
            # Ensure top-level display/embed are present for quick inspection
            out.setdefault("display_text", str(it.get("display_text", "")))
            if "action" in out and isinstance(out.get("action"), dict):
                out["action"].setdefault("embed_text", str(it.get("embed_text", "")))
            else:
                out.setdefault("embed_text", str(it.get("embed_text", "")))
            out["round_id"] = int(it.get("round_id", out.get("round_id", 1)) or 1)
            normalized.append(out)
        else:
            # Draft row (not yet built)
            normalized.append(
                {
                    "enabled": enabled,
                    "display_text": str(it.get("display_text", "")),
                    "embed_text": str(it.get("embed_text", "")),
                    "round_id": int(it.get("round_id", 1) or 1),
                }
            )

    payload = {"items": normalized}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -----------------------------
# Helper: persist and reload
# -----------------------------
def _persist_and_reload(path: str) -> None:
    """Save current session items to disk, then reload from disk for a clean, canonical state."""
    if not path:
        return
    _save_items(path, st.session_state.get("scenario_items", []))
    st.session_state["scenario_loaded_path"] = path
    st.session_state["scenario_items"] = _load_items(path)

def render_scenario_lab():
    st.title("Scenario Lab")

    st.caption(
        "파일(JSON)로 display_text / embed_text 목록을 관리하고, 선택한 항목으로 ChoiceMakerPipeline 결과를 확인합니다."
    )

    import io
    import contextlib
    import traceback

    def _get_pipe() -> ChoiceMakerPipeline:
        cache = st.session_state.setdefault("scenario_pipe_cache", {})
        key = f"dim"
        if key not in cache:
            cache[key] = ChoiceMakerPipeline(model_path="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf", dim=2048)
        return cache[key]

    # -----------------------------
    # Session state
    # -----------------------------
    if "scenario_items" not in st.session_state:
        st.session_state["scenario_items"] = []
    if "scenario_loaded_path" not in st.session_state:
        st.session_state["scenario_loaded_path"] = ""
    if "scenario_selected_idx" not in st.session_state:
        st.session_state["scenario_selected_idx"] = 0
    if "scenario_selected_idx_pending" not in st.session_state:
        st.session_state["scenario_selected_idx_pending"] = None
    if "scenario_add_open" not in st.session_state:
        st.session_state["scenario_add_open"] = False

    st.divider()

    # -----------------------------
    # File controls
    # -----------------------------
    st.subheader("1) 파일(자동 로드/자동 저장)")

    default_path = st.session_state["scenario_loaded_path"] or "data/scenario_items.json"
    store_path = st.text_input("Scenario JSON Path", value=default_path)

    # Auto-load on first render or when path changes
    if st.session_state.get("scenario_loaded_path") != store_path:
        st.session_state["scenario_loaded_path"] = store_path
        st.session_state["scenario_items"] = _load_items(store_path)
        st.session_state["scenario_selected_idx_pending"] = 0
        st.rerun()

    st.caption("이 페이지는 시작 시 자동으로 파일을 로드하며, 변경 사항은 자동 저장/리로드됩니다.")

    st.divider()

    # -----------------------------
    # List view / add / delete (table) + embedding compute
    # -----------------------------
    st.subheader("2) 리스트 관리 (표 기반: 출력 / 추가 / 삭제 / 사용 여부)")

    import json

    def _safe_json(v, max_len: int = 500) -> str:
        """Serialize nested/list values into a compact, readable string for table display."""
        try:
            s = json.dumps(v, ensure_ascii=False)
        except TypeError:
            s = str(v)
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        return s

    def _flatten_choice_item(item: dict) -> dict:
        # top-level
        out = {
            "choice_id": item.get("choice_id"),
            "display_text": item.get("display_text"),
            "enabled": item.get("enabled"),
            "round_id": item.get("round_id"),
            "embed_text": item.get("embed_text"),
        }

        # impact
        impact = item.get("impact") or {}
        out["impact_mag"] = impact.get("magnitude")
        out["impact_dur"] = impact.get("duration")
        out["delta_vars"] = _safe_json(impact.get("delta_vars") or {})
        out["direction"] = _safe_json(impact.get("direction") or [])

        # concepts: support concept_scores(dict), list[dict], or list[str]
        concept_scores = item.get("concept_scores")
        if isinstance(concept_scores, dict) and concept_scores:
            # Render in a stable, ranked order: follow `concepts` list if present, else sort by score desc.
            ordered_names: List[str] = []
            concepts = item.get("concepts") or []
            if isinstance(concepts, list) and concepts:
                ordered_names = [str(c).strip() for c in concepts if str(c).strip()]
            if not ordered_names:
                ordered_names = [k for k, _v in sorted(concept_scores.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)]

            parts: List[str] = []
            for name in ordered_names:
                try:
                    sc = float(concept_scores.get(name, 0.0))
                except Exception:
                    sc = 0.0
                parts.append(f"{name}({sc:.3f})")
            out["concepts"] = ", ".join(parts)

        else:
            concepts = item.get("concepts") or []
            if isinstance(concepts, list) and concepts and isinstance(concepts[0], dict):
                out["concepts"] = ", ".join(
                    [
                        f"{c.get('id')}({(c.get('score') or 0):.3f})"
                        for c in concepts
                        if isinstance(c, dict)
                    ]
                )
            elif isinstance(concepts, list):
                out["concepts"] = ", ".join([str(c) for c in concepts if str(c).strip()])
            else:
                out["concepts"] = str(concepts)

        # embed vec can be huge: show length + preview
        ev = item.get("embed_vec") or []
        if isinstance(ev, list):
            out["embed_vec_len"] = len(ev)
            out["embed_vec_head"] = _safe_json(ev[:12])
        else:
            out["embed_vec_len"] = None
            out["embed_vec_head"] = _safe_json(ev)

        return out

    items: List[dict] = st.session_state["scenario_items"]

    if not items:
        st.info("로드된 항목이 없습니다. Add로 추가하거나 Load 후 사용하세요.")
        items.append({"display_text": "", "embed_text": "", "enabled": True, "round_id": 1, "artifact": None})

    # Normalize schema
    for it in items:
        it["display_text"] = str(it.get("display_text", ""))
        it["embed_text"] = str(it.get("embed_text", ""))
        it["enabled"] = bool(it.get("enabled", True))
        it["artifact"] = it.get("artifact") if isinstance(it.get("artifact"), dict) else None
        it["round_id"] = int(it.get("round_id", 1) or 1)

    # Build flattened rows for display
    # If artifact exists, flatten it; otherwise flatten the item itself
    rows = []
    for it in items:
        art = it.get("artifact")
        if isinstance(art, dict):
            row = _flatten_choice_item(art)
            # fallback for display_text/embed_text/enabled/round_id if missing
            for k in ("display_text", "embed_text", "enabled", "round_id"):
                if row.get(k) is None:
                    row[k] = it.get(k)
            rows.append(row)
        else:
            rows.append(_flatten_choice_item(it))

    import pandas as pd
    df = pd.DataFrame(rows)
    if not df.empty:
        preferred_cols = [
            "choice_id",
            "display_text",
            "enabled",
            "round_id",
            "impact_mag",
            "impact_dur",
            "delta_vars",
            "concepts",
            "direction",
            "embed_text",
            "embed_vec_len",
            "embed_vec_head",
        ]
        cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
        df = df[cols]

    st.dataframe(df, width="stretch")

    # NOTE: Table is rendered with st.dataframe for read-only stability.
    # Inline edit persistence via `edited` (from st.data_editor) is intentionally disabled.

    # Explicit row selector
    # Apply pending selected index (must happen before the widget with key `scenario_selected_idx` is instantiated)
    pending = st.session_state.get("scenario_selected_idx_pending")
    if isinstance(pending, int):
        st.session_state["scenario_selected_idx"] = pending
    st.session_state["scenario_selected_idx_pending"] = None

    st.markdown("#### 선택 행")
    if len(items) == 0:
        cur_for_widget = 0
    else:
        cur_for_widget = int(st.session_state.get("scenario_selected_idx", 0))
        cur_for_widget = max(0, min(cur_for_widget, len(items) - 1))

    def _row_label(i: int) -> str:
        dt = (items[i].get("display_text") or "").strip()
        et = (items[i].get("embed_text") or "").strip()
        head = dt if dt else et
        head = head.replace("\n", " ")
        if len(head) > 60:
            head = head[:60] + "…"
        return f"{i}: {head}"

    selected_idx = st.selectbox(
        "실행/삭제 대상 행",
        options=list(range(len(items))),
        index=cur_for_widget,
        format_func=_row_label,
        key="scenario_selected_idx",
        disabled=(len(items) == 0),
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("Add", width="stretch"):
            st.session_state["scenario_add_open"] = True
            st.rerun()

    if st.session_state.get("scenario_add_open"):
        def _add_body() -> None:
            # Tag pool from total.json (used only for concept selection mode)
            total_items = _load_total_categories()
            enabled_words = [
                it["word"]
                for it in total_items
                if bool(it.get("enabled", True)) and str(it.get("word", "")).strip()
            ]
            enabled_words = _dedup_preserve_order(enabled_words)

            st.caption(
                "텍스트는 직접 입력합니다. Embed mode ON이면 임베딩을 생성하고, OFF이면 1~3순위 concept만 저장합니다."
            )

            embed_mode = st.toggle("Embed mode", value=False)

            # User-entered text (always present)
            src_text = st.text_area("텍스트", value="", height=120, placeholder="여기에 지문/대사를 입력하세요")

            col_round, _col_spacer = st.columns([1, 1])
            with col_round:
                new_round = st.number_input("round_id", min_value=1, value=1, step=1)

            # OFF mode: 3 ranked concept selections + numeric scores
            concepts: List[str] = []
            if not embed_mode:
                all_concepts = ["(none)"] + sorted(enabled_words)

                st.caption("OFF mode: 1~3순위 concept를 선택하고, 오른쪽에 유사도(0.0~1.0)를 입력하세요. 1.0 = 100%")

                cA, sA = st.columns([3, 1])
                with cA:
                    c1_val = st.selectbox("Concept #1", options=all_concepts, index=0, key="concept_1")
                with sA:
                    s1_val = st.number_input("Score #1", min_value=0.0, max_value=1.0, value=1.0, step=0.001, format="%.3f", key="concept_score_1")

                cB, sB = st.columns([3, 1])
                with cB:
                    c2_val = st.selectbox("Concept #2", options=all_concepts, index=0, key="concept_2")
                with sB:
                    s2_val = st.number_input("Score #2", min_value=0.0, max_value=1.0, value=0.85, step=0.001, format="%.3f", key="concept_score_2")

                cC, sC = st.columns([3, 1])
                with cC:
                    c3_val = st.selectbox("Concept #3", options=all_concepts, index=0, key="concept_3")
                with sC:
                    s3_val = st.number_input("Score #3", min_value=0.0, max_value=1.0, value=0.70, step=0.001, format="%.3f", key="concept_score_3")

                # Build a canonical scored string for downstream parsing/saving.
                scored_parts: List[str] = []
                for name, sc in [(c1_val, s1_val), (c2_val, s2_val), (c3_val, s3_val)]:
                    n = str(name).strip()
                    if not n or n == "(none)":
                        continue
                    try:
                        f = float(sc)
                    except Exception:
                        f = 0.0
                    if f < 0.0:
                        f = 0.0
                    if f > 1.0:
                        f = 1.0
                    scored_parts.append(f"{n}({f:.3f})")

                # Keep compatibility with existing payload logic
                st.session_state["concepts_scored_text"] = ", ".join(scored_parts)
                st.session_state["concepts_list"] = [p.split("(", 1)[0] for p in scored_parts]

            d_ok, d_cancel = st.columns([1, 1])
            with d_ok:
                ok = st.button("OK", type="primary", width="stretch")
            with d_cancel:
                cancel = st.button("Cancel", width="stretch")

            if cancel:
                st.session_state["scenario_add_open"] = False
                st.rerun()

            if ok:
                if not str(src_text).strip():
                    st.warning("텍스트를 입력해 주세요.")
                    return

                items = st.session_state.get("scenario_items", [])

                # Embed mode ON: generate embedding via pipeline
                if embed_mode:
                    try:
                        pipe = _get_pipe()
                        art = pipe.make_choice(
                            str(src_text).strip(),
                            round_id=int(new_round),
                            overrides=None,
                            debug=False,
                        )

                        payload = art.to_choice_payload()

                        # Remove best fields source (ontology_best) entirely.
                        if isinstance(payload, dict):
                            payload.pop("ontology_best", None)
                            # In embed mode, concepts can be empty by default
                            payload.setdefault("concepts", [])

                        items.append(
                            {
                                "display_text": str(src_text).strip(),
                                "embed_text": str(art.embed_text),
                                "enabled": True,
                                "round_id": int(new_round),
                                "artifact": payload,
                            }
                        )
                    except Exception:
                        # fallback: add without artifact
                        items.append(
                            {
                                "display_text": str(src_text).strip(),
                                "embed_text": str(src_text).strip(),
                                "enabled": True,
                                "round_id": int(new_round),
                                "artifact": {
                                    "display_text": str(src_text).strip(),
                                    "embed_text": str(src_text).strip(),
                                    "concepts": [],
                                },
                            }
                        )

                # Embed mode OFF: build ChoiceArtifact via make_choice_off()
                else:
                    pipe = _get_pipe()

                    # Ranked concepts (already limited to 3 by UI)
                    ranked_concepts = list(st.session_state.get("concepts_list", []))

                    # Parse scored text into {concept: score}
                    scored_text = st.session_state.get("concepts_scored_text", "")
                    concept_scores = parse_scored_concepts(scored_text)

                    art = pipe.make_choice_off(
                        str(src_text).strip(),
                        concepts_ranked=ranked_concepts,
                        concept_scores=concept_scores,
                        round_id=int(new_round),
                        debug=False,
                    )

                    payload = art.to_choice_payload()

                    # Ensure OFF-mode payload has no ontology_best duplication at top-level
                    if isinstance(payload, dict):
                        payload.pop("ontology_best", None)

                    items.append(
                        {
                            "display_text": str(src_text).strip(),
                            "embed_text": str(art.embed_text),
                            "enabled": True,
                            "round_id": int(new_round),
                            "artifact": payload,
                        }
                    )

                st.session_state["scenario_items"] = items

                # Auto save + reload
                _persist_and_reload(store_path)

                st.session_state["scenario_add_open"] = False
                st.session_state["scenario_selected_idx_pending"] = max(
                    0, len(st.session_state.get("scenario_items", [])) - 1
                )
                st.rerun()

        # Streamlit dialog API differs by version.
        # Newer: st.dialog is a decorator. Older: st.experimental_dialog.
        if hasattr(st, "dialog"):
            @st.dialog("Add Item")
            def _dlg():
                _add_body()

            _dlg()
        elif hasattr(st, "experimental_dialog"):
            @st.experimental_dialog("Add Item")
            def _dlg2():
                _add_body()

            _dlg2()
        else:
            with st.expander("Add Item", expanded=True):
                _add_body()

    with c2:
        if st.button("Delete Selected Row", width="stretch"):
            if not items:
                st.warning("삭제할 행이 없습니다.")
            else:
                idx = int(st.session_state.get("scenario_selected_idx", 0))
                if 0 <= idx < len(items):
                    del items[idx]

                # Ensure at least one row exists
                if not items:
                    items.append({"display_text": "", "embed_text": "", "enabled": True, "round_id": 1, "artifact": None})
                    next_idx = 0
                else:
                    next_idx = max(0, min(idx, len(items) - 1))

                st.session_state["scenario_items"] = items

                # Auto save + reload
                _persist_and_reload(store_path)

                # Defer updating the selectbox key until the next rerun
                st.session_state["scenario_selected_idx_pending"] = int(next_idx)
                st.rerun()

    st.caption("표 변경/추가/삭제/실행 결과는 자동으로 저장되고 즉시 리로드됩니다. 실행/삭제는 위 선택 행 기준입니다.")

    st.divider()

    st.subheader("3) 선택 항목 결과")
    items_preview = st.session_state.get("scenario_items", [])
    idx_preview = int(st.session_state.get("scenario_selected_idx", 0)) if items_preview else 0
    if items_preview and 0 <= idx_preview < len(items_preview) and isinstance(items_preview[idx_preview].get("artifact"), dict):
        st.json(items_preview[idx_preview]["artifact"])
    else:
        st.info("선택한 항목에 저장된 결과(artifact)가 없습니다. Add 하거나 표를 수정하면 자동으로 생성됩니다.")
