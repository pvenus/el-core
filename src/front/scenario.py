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
        if ("choice_id" in it) or ("action" in it) or ("action_id" in it):
            display_text = str(it.get("display_text", ""))
            embed_text = str(it.get("action", {}).get("embed_text", it.get("embed_text", "")))
            items.append(
                {
                    "display_text": display_text,
                    "embed_text": embed_text,
                    "enabled": bool(it.get("enabled", True)),
                    "artifact": it,
                    "dim": int(it.get("dim", 6) or 6),
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
                "dim": int(it.get("dim", 6) or 6),
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
            out["dim"] = int(it.get("dim", out.get("dim", 6)) or 6)
            out["round_id"] = int(it.get("round_id", out.get("round_id", 1)) or 1)
            normalized.append(out)
        else:
            # Draft row (not yet built)
            normalized.append(
                {
                    "enabled": enabled,
                    "display_text": str(it.get("display_text", "")),
                    "embed_text": str(it.get("embed_text", "")),
                    "dim": int(it.get("dim", 6) or 6),
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

    def _get_pipe(dim: int) -> ChoiceMakerPipeline:
        cache = st.session_state.setdefault("scenario_pipe_cache", {})
        key = f"dim:{int(dim)}"
        if key not in cache:
            cache[key] = ChoiceMakerPipeline(dim=int(dim))
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

    items: List[Dict[str, Any]] = st.session_state["scenario_items"]

    if not items:
        st.info("로드된 항목이 없습니다. Add로 추가하거나 Load 후 사용하세요.")
        items.append({"display_text": "", "embed_text": "", "enabled": True, "dim": 6, "round_id": 1, "artifact": None})

    # Normalize schema
    for it in items:
        it["display_text"] = str(it.get("display_text", ""))
        it["embed_text"] = str(it.get("embed_text", ""))
        it["enabled"] = bool(it.get("enabled", True))
        it["artifact"] = it.get("artifact") if isinstance(it.get("artifact"), dict) else None
        it["dim"] = int(it.get("dim", 6) or 6)
        it["round_id"] = int(it.get("round_id", 1) or 1)

    rows_for_editor: List[Dict[str, Any]] = []
    for it in items:
        art = it.get("artifact") if isinstance(it, dict) else None
        choice_id = ""
        action_id = ""
        tags_preview = ""
        if isinstance(art, dict):
            choice_id = str(art.get("choice_id", ""))
            action_id = str(art.get("action", {}).get("action_id", art.get("action_id", "")))
            tags = art.get("tags", [])
            if isinstance(tags, list):
                tags_preview = ",".join([str(x) for x in tags])

        rows_for_editor.append(
            {
                "enabled": it["enabled"],
                "display_text": it["display_text"],
                "embed_text": it["embed_text"],
                "dim": int(it.get("dim", 6) or 6),
                "round_id": int(it.get("round_id", 1) or 1),
                "choice_id": choice_id,
                "action_id": action_id,
                "tags": tags_preview,
            }
        )

    edited = st.data_editor(
        rows_for_editor,
        key="scenario_table",
        num_rows="fixed",
        width="stretch",
        column_config={
            "enabled": st.column_config.CheckboxColumn("enabled"),
            "display_text": st.column_config.TextColumn("display_text", width="large"),
            "embed_text": st.column_config.TextColumn("embed_text", width="large"),
            "dim": st.column_config.NumberColumn("dim", min_value=2, step=1, width="small"),
            "round_id": st.column_config.NumberColumn("round_id", min_value=1, step=1, width="small"),
            "choice_id": st.column_config.TextColumn("choice_id", disabled=True),
            "action_id": st.column_config.TextColumn("action_id", disabled=True),
            "tags": st.column_config.TextColumn("tags", disabled=True, width="medium"),
        },
        hide_index=False,
    )

    # Persist edits (auto-save + reload)
    if isinstance(edited, list) and len(edited) == len(items):
        changed = False
        for i in range(len(items)):
            new_enabled = bool(edited[i].get("enabled", True))
            new_display = str(edited[i].get("display_text", ""))
            new_embed = str(edited[i].get("embed_text", ""))
            new_dim = int(edited[i].get("dim", items[i].get("dim", 6)) or 6)
            new_round = int(edited[i].get("round_id", items[i].get("round_id", 1)) or 1)

            if (
                (new_enabled != items[i].get("enabled"))
                or (new_display != items[i].get("display_text"))
                or (new_embed != items[i].get("embed_text"))
                or (new_dim != int(items[i].get("dim", 6) or 6))
                or (new_round != int(items[i].get("round_id", 1) or 1))
            ):
                changed = True

            # If text or dim changed, invalidate previously built artifact
            if (
                (new_display != items[i].get("display_text"))
                or (new_embed != items[i].get("embed_text"))
                or (new_dim != int(items[i].get("dim", 6) or 6))
            ):
                items[i]["artifact"] = None

            items[i]["enabled"] = new_enabled
            items[i]["display_text"] = new_display
            items[i]["embed_text"] = new_embed
            items[i]["dim"] = int(new_dim)
            items[i]["round_id"] = int(new_round)

        st.session_state["scenario_items"] = items

        if changed:
            # Auto-run make_choice for enabled rows whose artifact is missing
            for j in range(len(items)):
                row = items[j]
                if not bool(row.get("enabled", True)):
                    continue
                if not str(row.get("display_text", "")).strip():
                    continue
                if row.get("artifact") is not None:
                    continue
                try:
                    pipe = _get_pipe(int(row.get("dim", 6) or 6))
                    art = pipe.make_choice(
                        str(row.get("display_text", "")),
                        round_id=int(row.get("round_id", 1) or 1),
                        overrides={"embed_text": str(row.get("embed_text", ""))} if str(row.get("embed_text", "")).strip() else None,
                        debug=False,
                    )
                    row["artifact"] = art.to_choice_payload()
                    row["embed_text"] = str(art.embed_text)
                except Exception:
                    pass

            st.session_state["scenario_items"] = items
            _persist_and_reload(store_path)
            # Keep selection stable
            st.session_state["scenario_selected_idx_pending"] = int(st.session_state.get("scenario_selected_idx", 0))
            st.rerun()

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
            st.caption("display_text / embed_text 두 값을 입력한 뒤 OK를 누르면 즉시 저장되고 화면이 갱신됩니다.")
            new_display = st.text_area("display_text", value="", height=120)
            new_embed = st.text_area("embed_text", value="", height=120)

            a1, a2 = st.columns([1, 1])
            with a1:
                new_dim = st.number_input("dim", min_value=2, value=6, step=1)
            with a2:
                new_round = st.number_input("round_id", min_value=1, value=1, step=1)

            d_ok, d_cancel = st.columns([1, 1])
            with d_ok:
                ok = st.button("OK", type="primary", width="stretch")
            with d_cancel:
                cancel = st.button("Cancel", width="stretch")

            if cancel:
                st.session_state["scenario_add_open"] = False
                st.rerun()

            if ok:
                items = st.session_state.get("scenario_items", [])
                try:
                    pipe = _get_pipe(int(new_dim))
                    art = pipe.make_choice(
                        str(new_display),
                        round_id=int(new_round),
                        overrides={"embed_text": str(new_embed)} if str(new_embed).strip() else None,
                        debug=False,
                    )
                    payload = art.to_choice_payload()
                    items.append(
                        {
                            "display_text": str(new_display),
                            "embed_text": str(art.embed_text),
                            "enabled": True,
                            "dim": int(new_dim),
                            "round_id": int(new_round),
                            "artifact": payload,
                        }
                    )
                except Exception:
                    # fallback: add empty artifact if make_choice fails
                    items.append(
                        {
                            "display_text": str(new_display),
                            "embed_text": str(new_embed),
                            "enabled": True,
                            "dim": int(new_dim),
                            "round_id": int(new_round),
                            "artifact": None,
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
                    items.append({"display_text": "", "embed_text": "", "enabled": True, "dim": 6, "round_id": 1, "artifact": None})
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
