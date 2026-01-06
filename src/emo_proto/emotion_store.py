from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -------------------------
# 기본 유틸
# -------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + eps)


def softmax_np(x: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    z = x / t
    z = z - float(np.max(z))  # stability
    e = np.exp(z)
    return e / (float(np.sum(e)) + 1e-12)


def topk_axes_by_mean_abs(mat: np.ndarray, k: int) -> np.ndarray:
    """
    mat: [N, D] (row-normalized recommended)
    axis_importance = mean(|value|) over rows
    """
    if mat.ndim != 2:
        raise ValueError("mat must be 2D")
    imp = np.mean(np.abs(mat), axis=0)  # [D]
    idx = np.argsort(imp)[::-1]
    k = int(max(1, min(k, mat.shape[1])))
    return idx[:k]


def slice_and_renorm(v: np.ndarray, axes: np.ndarray) -> np.ndarray:
    vv = v[axes].astype(float)
    n = float(np.linalg.norm(vv))
    if n > 0:
        vv = vv / n
    return vv


def cosine_dot(a: np.ndarray, b: np.ndarray) -> float:
    # a, b are assumed L2-normalized
    return float(np.dot(a, b))


# -------------------------
# emotions.json I/O
# -------------------------
def load_emotions(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"version": 1, "emotions": []}
    return json.loads(p.read_text(encoding="utf-8"))


def save_emotions(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# CRUD
# -------------------------
def ensure_schema(e: Dict[str, Any]) -> None:
    e.setdefault("texts", [])
    e.setdefault("prototype", {"algo": "mean_dir", "vector": None, "dim": None, "updated_at": None})
    # 다중 프로토타입용 캐시(선택)
    e.setdefault("text_vectors", None)         # list[list[float]] or None
    e.setdefault("text_vectors_updated_at", None)


def list_emotions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    emos = data.get("emotions", [])
    for e in emos:
        ensure_schema(e)
    return emos


def upsert_emotion(data: Dict[str, Any], emo_id: str, name: str, texts: Optional[List[str]] = None) -> None:
    emos = data.setdefault("emotions", [])
    for e in emos:
        if e.get("id") == emo_id:
            e["name"] = name
            ensure_schema(e)
            if texts is not None:
                e["texts"] = texts
            return
    new_e = {
        "id": emo_id,
        "name": name,
        "texts": texts or [],
        "prototype": {"algo": "mean_dir", "vector": None, "dim": None, "updated_at": None},
        "text_vectors": None,
        "text_vectors_updated_at": None,
    }
    emos.append(new_e)


def delete_emotion(data: Dict[str, Any], emo_id: str) -> None:
    data["emotions"] = [e for e in data.get("emotions", []) if e.get("id") != emo_id]


def set_texts_from_multiline(data: Dict[str, Any], emo_id: str, multiline: str) -> None:
    lines = [(ln or "").strip() for ln in (multiline or "").splitlines()]
    lines = [ln for ln in lines if ln]
    for e in data.get("emotions", []):
        if e.get("id") == emo_id:
            ensure_schema(e)
            e["texts"] = lines
            # texts가 바뀌면 캐시 무효화
            e["text_vectors"] = None
            e["text_vectors_updated_at"] = None
            # prototype도 무효화(선택적으로)
            e["prototype"]["vector"] = None
            e["prototype"]["dim"] = None
            e["prototype"]["updated_at"] = None
            return
    raise KeyError(f"Emotion not found: {emo_id}")


# -------------------------
# Embedding
# -------------------------
def embed_text(model, text: str) -> Tuple[np.ndarray, int]:
    out = model.embed_text(text)
    v = np.asarray(out["pooled"], dtype=float)
    dim = int(out.get("dim", v.shape[0]))
    return v, dim


def embed_texts(model, texts: List[str]) -> Tuple[np.ndarray, int]:
    vecs: List[np.ndarray] = []
    dim: Optional[int] = None
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        v, dim_ = embed_text(model, t)
        dim = dim_
        vecs.append(v)
    if not vecs:
        raise ValueError("No valid texts.")
    X = np.vstack(vecs)
    return X, int(dim if dim is not None else X.shape[1])


# 1) compute_text_vectors_for_emotion 시그니처 변경 + vector_source 적용

def compute_text_vectors_for_emotion(e: Dict[str, Any], model, vector_source: str = "texts_then_name") -> None:
    """
    vector_source:
      - "texts_then_name": texts가 있으면 texts, 비면 name 1개로 fallback
      - "name_only": 항상 name 1개만 사용 (texts 무시)
    """
    ensure_schema(e)

    name = (e.get("name") or "").strip()
    texts = e.get("texts") or []
    texts = [(t or "").strip() for t in texts if (t or "").strip()]

    if vector_source == "name_only":
        if not name:
            e["text_vectors"] = None
            e["text_vectors_updated_at"] = None
            return
        use_texts = [name]
    else:
        # texts_then_name
        if texts:
            use_texts = texts
        else:
            if not name:
                e["text_vectors"] = None
                e["text_vectors_updated_at"] = None
                return
            use_texts = [name]

    X, _ = embed_texts(model, use_texts)
    Xn = np.vstack([l2_normalize(x) for x in X])
    e["text_vectors"] = Xn.astype(float).tolist()
    e["text_vectors_updated_at"] = now_iso()


def build_prototype_from_text_vectors(e: Dict[str, Any], algo: str = "mean_dir") -> Tuple[List[float], int]:
    """
    e["text_vectors"] must exist
    """
    tv = e.get("text_vectors")
    if not tv:
        raise ValueError("text_vectors missing. Compute them first.")
    Xn = np.asarray(tv, dtype=float)  # already normalized (n, dim)
    dim = int(Xn.shape[1])

    if algo == "mean_dir":
        m = l2_normalize(Xn.mean(axis=0))
        return m.astype(float).tolist(), dim

    if algo == "medoid":
        sims = Xn @ Xn.T
        score = sims.mean(axis=1)
        idx = int(np.argmax(score))
        v = l2_normalize(Xn[idx])
        return v.astype(float).tolist(), dim

    raise ValueError(f"Unsupported algo: {algo}")


# 2) batch_compute_all_vectors에 vector_source 파라미터 추가해서 전달

def batch_compute_all_vectors(
    data: Dict[str, Any],
    model,
    algo: str = "mean_dir",
    compute_text_vectors: bool = True,
    vector_source: str = "texts_then_name",  # ✅ 추가
) -> Dict[str, Any]:
    emos = list_emotions(data)

    for e in emos:
        if compute_text_vectors:
            compute_text_vectors_for_emotion(e, model, vector_source=vector_source)

        tv = e.get("text_vectors")
        if not tv:
            ensure_schema(e)
            e["prototype"] = {"algo": algo, "vector": None, "dim": None, "updated_at": None}
            continue

        vec, dim = build_prototype_from_text_vectors(e, algo=algo)
        e["prototype"] = {"algo": algo, "vector": vec, "dim": dim, "updated_at": now_iso()}

    return data


# -------------------------
# Compare 옵션을 위한 준비
# -------------------------
def collect_vectors_for_axes(
    data: Dict[str, Any],
    use_multi_prototype: bool,
) -> Tuple[np.ndarray, List[str]]:
    """
    Top-K axes 계산을 위한 mat 구성.
    - multi-prototype면: 모든 감정의 text_vectors를 쌓음
    - 아니면: 각 감정 prototype만 쌓음
    """
    mat_rows: List[np.ndarray] = []
    labels: List[str] = []

    for e in list_emotions(data):
        emo_id = e.get("id", "?")
        name = e.get("name", emo_id)

        if use_multi_prototype:
            tv = e.get("text_vectors") or []
            for j, v in enumerate(tv):
                vv = l2_normalize(np.asarray(v, dtype=float))
                mat_rows.append(vv)
                labels.append(f"{name}:{emo_id}#{j+1}")
        else:
            pv = (e.get("prototype") or {}).get("vector")
            if pv is None:
                continue
            vv = l2_normalize(np.asarray(pv, dtype=float))
            mat_rows.append(vv)
            labels.append(f"{name}:{emo_id}")

    if not mat_rows:
        raise ValueError("No vectors available to build axes matrix. Compute vectors first.")
    mat = np.vstack(mat_rows)
    return mat, labels


def compute_global_center(
    data: Dict[str, Any],
    use_multi_prototype: bool,
) -> np.ndarray:
    """
    centering에 쓰는 global mean μ.
    - multi-prototype면: 모든 text_vectors 평균
    - 아니면: 모든 prototype 평균
    """
    vecs: List[np.ndarray] = []
    for e in list_emotions(data):
        if use_multi_prototype:
            tv = e.get("text_vectors") or []
            for v in tv:
                vecs.append(np.asarray(v, dtype=float))
        else:
            pv = (e.get("prototype") or {}).get("vector")
            if pv is not None:
                vecs.append(np.asarray(pv, dtype=float))
    if not vecs:
        raise ValueError("No vectors to compute global center.")
    mu = np.mean(np.vstack(vecs), axis=0)
    return mu.astype(float)


# -------------------------
# Similarity (main)
# -------------------------
def score_query_against_emotions(
    data: Dict[str, Any],
    model,
    query: str,
    *,
    use_multi_prototype: bool = True,
    top_m: int = 3,
    use_centering: bool = True,
    use_topk_axes: bool = True,
    topk: int = 32,
    temperature: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    입력 문장 -> 감정별 score -> softmax percent

    - multi-prototype:
        score_i = mean(top_m cos(q, tv_ij))
      else:
        score_i = cos(q, proto_i)

    - centering:
        q := norm(q - μ), p := norm(p - μ)

    - top-k axes:
        axes from emotion matrix by mean(|value|)
        q := renorm(q[axes]), p := renorm(p[axes])
    """
    query = (query or "").strip()
    if not query:
        return []

    q_full, _ = embed_text(model, query)
    q_full = l2_normalize(q_full)

    mu = None
    if use_centering:
        mu = compute_global_center(data, use_multi_prototype=use_multi_prototype)

    axes = None
    if use_topk_axes:
        mat, _ = collect_vectors_for_axes(data, use_multi_prototype=use_multi_prototype)
        axes = topk_axes_by_mean_abs(mat, int(topk))

    rows: List[Dict[str, Any]] = []

    for e in list_emotions(data):
        emo_id = e.get("id")
        name = e.get("name")

        candidates: List[np.ndarray] = []
        if use_multi_prototype:
            tv = e.get("text_vectors") or []
            for v in tv:
                candidates.append(l2_normalize(np.asarray(v, dtype=float)))
        else:
            pv = (e.get("prototype") or {}).get("vector")
            if pv is not None:
                candidates.append(l2_normalize(np.asarray(pv, dtype=float)))

        if not candidates:
            continue

        q = q_full.copy()
        if mu is not None:
            q = l2_normalize(q - mu)
        if axes is not None:
            q = slice_and_renorm(q, axes)

        sims: List[float] = []
        for p_full in candidates:
            p = p_full.copy()
            if mu is not None:
                p = l2_normalize(p - mu)
            if axes is not None:
                p = slice_and_renorm(p, axes)
            sims.append(cosine_dot(q, p))

        sims_arr = np.array(sims, dtype=float)
        if use_multi_prototype:
            m = int(max(1, min(top_m, sims_arr.size)))
            top_vals = np.sort(sims_arr)[::-1][:m]
            score = float(np.mean(top_vals))
        else:
            score = float(sims_arr[0])

        rows.append({"id": emo_id, "name": name, "cosine": score, "n_candidates": len(candidates)})

    if not rows:
        return []

    scores = np.array([r["cosine"] for r in rows], dtype=float)
    probs = softmax_np(scores, temperature=float(temperature)) * 100.0
    for i, r in enumerate(rows):
        r["percent"] = float(probs[i])

    rows.sort(key=lambda r: -r["percent"])
    return rows
