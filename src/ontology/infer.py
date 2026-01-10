"""Ontology inference helpers.

This module provides a small "reasoner"-like inference layer on top of the
ontology objects produced by `src/ontology/builder.py`.

Goals
- Given a concept word/text, return its nearest anchors (already available via sims/top-anchors).
- Given an embedding vector, infer top anchors and nearest concepts.
- Provide simple rule-based reasoning hooks (thresholds, allow/deny categories).

This is intentionally dependency-light and works with the in-repo dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import logging

# NOTE: User requested always-on verbose output (no debug toggle).
# We use prints for deterministic visibility in IDEs and scripts.
LOG = logging.getLogger("ontology.infer")

_VERBOSE = True

def _p(msg: str) -> None:
    if _VERBOSE:
        print(msg)


# -----------------------------
# Utilities
# -----------------------------

def _as_list_floats(x: Any) -> List[float]:
    """Best-effort conversion to List[float]."""
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    # numpy / torch tensors, etc.
    try:
        return [float(v) for v in list(x)]
    except Exception as e:
        raise TypeError(f"Cannot convert to float list: {type(x)}") from e


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity for two vectors."""
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"Vector dim mismatch: {len(a)} != {len(b)}")

    # local imports to keep module light
    import math

    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


# -----------------------------
# Public result types
# -----------------------------


@dataclass(frozen=True)
class ScoredItem:
    id: str
    label: str
    score: float
    meta: Dict[str, Any]

    def to_dict(self, *, include_meta: bool = False) -> Dict[str, Any]:
        """Serialize to a plain dict.

        Note: scenario/maker often expects {id,label,score}. Some call sites also
        check `similarity`, so we provide it for compatibility.
        """
        d: Dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "score": float(self.score),
            "similarity": float(self.score),
        }
        if include_meta:
            d["meta"] = dict(self.meta or {})
        return d

    def to_tuple(self) -> Tuple[str, str, float]:
        """(id, label, score) convenience."""
        return (self.id, self.label, float(self.score))


@dataclass(frozen=True)
class InferenceResult:
    """Unified inference result."""

    query: str
    category_id: Optional[str]
    top_anchors: List[ScoredItem]
    nearest_concepts: List[ScoredItem]

    def to_dict(self, *, include_meta: bool = False) -> Dict[str, Any]:
        return {
            "query": self.query,
            "category_id": self.category_id,
            "anchors": [a.to_dict(include_meta=include_meta) for a in self.top_anchors],
            "concepts": [c.to_dict(include_meta=include_meta) for c in self.nearest_concepts],
        }


# -----------------------------
# Ontology adapters
# -----------------------------


def _get_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _concept_text(concept: Any) -> str:
    return str(_get_attr(concept, "text", "word", "label", "name", default=""))


def _concept_id(concept: Any) -> str:
    return str(_get_attr(concept, "id", default=""))


def _concept_category_id(concept: Any) -> Optional[str]:
    cid = _get_attr(concept, "category_id", "category", default=None)
    if cid is None:
        return None
    # If category is a dataclass/object, try common id/name fields.
    if not isinstance(cid, (str, int)):
        cid2 = _get_attr(cid, "id", "name", "key", default=None)
        if cid2 is not None:
            return str(cid2)
    return str(cid)


def _concept_vector(concept: Any) -> List[float]:
    vec = _get_attr(concept, "vector", "embed_vec", "embedding", default=None)
    if vec is None:
        return []
    # vector may be a dataclass-like with `.values`
    values = _get_attr(vec, "values", default=None)
    if values is not None:
        return _as_list_floats(values)
    return _as_list_floats(vec)


def _anchor_id(anchor: Any) -> str:
    return str(_get_attr(anchor, "id", default=""))


def _anchor_name(anchor: Any) -> str:
    return str(_get_attr(anchor, "name", "text", "label", default=""))


def _anchor_vector(anchor: Any) -> List[float]:
    vec = _get_attr(anchor, "vector", "embed_vec", "embedding", default=None)
    if vec is None:
        return []
    values = _get_attr(vec, "values", default=None)
    if values is not None:
        return _as_list_floats(values)
    return _as_list_floats(vec)



# -----------------------------
# Reasoner
# -----------------------------


# Helper: Map cosine similarity [-1, 1] to [0, 1]
def _cosine_to_01(s: float) -> float:
    """Map cosine similarity from [-1, 1] to [0, 1]."""
    # Clamp for numerical safety
    if s < -1.0:
        s = -1.0
    elif s > 1.0:
        s = 1.0
    return 0.5 * (s + 1.0)


class Reasoner:
    """A lightweight reasoner for the ontology produced by builder.

    Expected ontology shape (best-effort):
      - ontology.categories: Dict[str, Category]
      - ontology.anchors: Dict[str, AnchorAxis]  OR List[AnchorAxis]
      - ontology.concepts: Dict[str, Concept]    OR List[Concept]

    If your actual object uses different field names, adapters above attempt
    to handle common variants.
    """

    def __init__(
        self,
        ontology: Any,
        cosine_fn: Callable[[Sequence[float], Sequence[float]], float] = _cosine_sim,
        similarity_transform: Callable[[float], float] | None = None,
    ) -> None:
        self.ontology = ontology
        self.cosine_fn = cosine_fn
        # Similarity post-processing.
        # Many UIs/consumers expect a 0..1 score; cosine similarity is -1..1.
        # If no transform is provided, we keep raw cosine.
        self.similarity_transform = similarity_transform or (lambda s: s)

        # Always-on diagnostics: dump ontology structure to locate anchors/fields.
        _p("[Reasoner] ---- ontology diagnostics (always-on) ----")
        try:
            public_attrs = [
                a for a in dir(ontology)
                if not a.startswith("_")
                and a not in {"__dict__", "__weakref__", "__annotations__"}
            ]
            _p(f"[Reasoner] ontology type={type(ontology).__name__} public_attrs({len(public_attrs)}): {public_attrs}")
        except Exception as e:
            _p(f"[Reasoner] ontology attr dump failed: {type(e).__name__}: {e}")

        # Probe common container fields and print summary.
        probe_names = [
            "anchors",
            "anchor_axes",
            "axes",
            "anchorAxis",
            "anchors_by_id",
            "anchor_map",
            "anchor",
            "anchor_list",
            "anchors_list",
            "categories",
            "concepts",
            "relations",
        ]
        for n in probe_names:
            try:
                v = getattr(ontology, n, None)
                if v is None:
                    continue
                # Determine container shape without materializing huge iterables.
                shape = None
                try:
                    if isinstance(v, dict):
                        shape = f"dict(len={len(v)})"
                    elif isinstance(v, list):
                        shape = f"list(len={len(v)})"
                    else:
                        # Fallback for iterables/objects
                        shape = f"{type(v).__name__}"
                except Exception:
                    shape = f"{type(v).__name__}"

                # Sample a first element (safe)
                sample = None
                try:
                    if isinstance(v, dict) and v:
                        sample = next(iter(v.values()))
                    elif isinstance(v, list) and v:
                        sample = v[0]
                    else:
                        # try iterator
                        it = iter(v)
                        sample = next(it, None)
                except Exception:
                    sample = None

                if sample is not None:
                    _p(
                        f"[Reasoner] probe {n}: {shape} sample_type={type(sample).__name__} sample_repr={repr(sample)[:200]}"
                    )
                    # If sample looks like an anchor/concept, print key fields
                    try:
                        sid = _get_attr(sample, "id", default=None)
                        sname = _get_attr(sample, "name", "text", "label", default=None)
                        svec = _get_attr(sample, "vector", "embed_vec", "embedding", default=None)
                        sdim = None
                        try:
                            if svec is not None:
                                svals = _get_attr(svec, "values", default=svec)
                                sdim = len(list(svals)) if svals is not None else None
                        except Exception:
                            sdim = None
                        _p(f"[Reasoner]   sample fields: id={sid!r} name/text={sname!r} vec_dim={sdim!r}")
                    except Exception:
                        pass
                else:
                    _p(f"[Reasoner] probe {n}: {shape} (no sample)")
            except Exception as e:
                _p(f"[Reasoner] probe {n} failed: {type(e).__name__}: {e}")
        _p("[Reasoner] ---- end diagnostics ----")

        self._anchors: List[Any] = self._collect_anchors(ontology)
        self._concepts: List[Any] = self._collect_concepts(ontology)

        anchors_with_vec = sum(1 for a in self._anchors if _anchor_vector(a))
        concepts_with_vec = sum(1 for c in self._concepts if _concept_vector(c))
        _p(f"[Reasoner] Loaded ontology: anchors={len(self._anchors)} (with_vec={anchors_with_vec}), concepts={len(self._concepts)} (with_vec={concepts_with_vec})")
        if self._anchors:
            a0 = self._anchors[0]
            _p(f"[Reasoner] Anchor[0]: id={_anchor_id(a0)!r} name={_anchor_name(a0)!r} vec_dim={len(_anchor_vector(a0))}")
        if self._concepts:
            c0 = self._concepts[0]
            _p(f"[Reasoner] Concept[0]: id={_concept_id(c0)!r} text={_concept_text(c0)!r} vec_dim={len(_concept_vector(c0))} category_id={_concept_category_id(c0)!r}")

        # Optional indexes for fast lookup
        self._concept_by_text: Dict[str, Any] = {}
        for c in self._concepts:
            t = _concept_text(c).strip().lower()
            if t:
                self._concept_by_text.setdefault(t, c)

    @staticmethod
    def _collect_anchors(ontology: Any) -> List[Any]:
        """Collect anchors from the ontology.

        Builder variants may store anchors under different field names.
        We probe common candidates and return the first non-empty container.
        """

        candidates = [
            "anchors",
            "anchor_axes",
            "axes",
            "anchorAxis",
            "anchors_by_id",
            "anchor_map",
            "anchor_list",
            "anchors_list",
        ]

        for name in candidates:
            anchors = _get_attr(ontology, name, default=None)
            if anchors is None:
                continue

            # Normalize to list
            try:
                if isinstance(anchors, dict):
                    vals = list(anchors.values())
                elif isinstance(anchors, list):
                    vals = anchors
                else:
                    vals = list(anchors)
            except Exception:
                continue

            if vals:
                _p(f"[Reasoner] _collect_anchors picked field {name!r} -> {len(vals)} anchors")
                return vals

        _p("[Reasoner] _collect_anchors found no anchors in known fields")
        return []

    @staticmethod
    def _collect_concepts(ontology: Any) -> List[Any]:
        candidates = ["concepts", "concept_list", "nodes", "items"]
        for name in candidates:
            concepts = _get_attr(ontology, name, default=None)
            if concepts is None:
                continue
            try:
                if isinstance(concepts, dict):
                    vals = list(concepts.values())
                elif isinstance(concepts, list):
                    vals = concepts
                else:
                    vals = list(concepts)
            except Exception:
                continue
            if vals:
                _p(f"[Reasoner] _collect_concepts picked field {name!r} -> {len(vals)} concepts")
                return vals
        _p("[Reasoner] _collect_concepts found no concepts in known fields")
        return []

    # ----------
    # Public APIs
    # ----------

    def infer_from_text(
        self,
        text: str,
        *,
        category_id: Optional[str] = None,
        top_k_anchors: int = 5,
        top_k_concepts: int = 10,
        min_anchor_score: float = -1.0,
        min_concept_score: float = -1.0,
    ) -> InferenceResult:
        """Infer anchors/concepts given a concept text.

        Strategy:
          1) If the text matches an existing concept, use its precomputed sims (if any).
          2) Otherwise, fallback to vector similarity if possible (requires external embedder).

        Note: For (2), consider calling `infer_from_vector()` directly.
        """

        q = (text or "").strip()
        key = q.lower()
        concept = self._concept_by_text.get(key)

        _p(f"[infer_from_text] query={q!r} matched_concept={'yes' if concept is not None else 'no'}")

        top_anchors: List[ScoredItem] = []
        nearest_concepts: List[ScoredItem] = []

        if concept is not None:
            # Prefer precomputed sims dict if present; otherwise compute from anchor vectors.
            sims = _get_attr(concept, "sims", default=None)
            _p(f"[infer_from_text] concept={_concept_text(concept)!r} sims_type={type(sims).__name__} sims_len={(len(sims) if isinstance(sims, dict) else None)} vec_dim={len(_concept_vector(concept))}")
            if isinstance(sims, dict) and sims:
                items = []
                for a_name, score in sims.items():
                    if score is None:
                        continue
                    s = float(score)
                    if s < min_anchor_score:
                        continue
                    items.append(
                        ScoredItem(
                            id=str(a_name),
                            label=str(a_name),
                            score=s,
                            meta={"source": "concept.sims"},
                        )
                    )
                items.sort(key=lambda x: x.score, reverse=True)
                top_anchors = items[: max(0, int(top_k_anchors))]
            else:
                # Fallback: compute anchor similarity from the concept vector.
                vec0 = _concept_vector(concept)
                if vec0:
                    top_anchors = self._nearest_anchors_by_vector(
                        vec0, top_k=top_k_anchors, min_score=min_anchor_score
                    )

            # Nearest concepts by vector sim within category (optional)
            vec = _concept_vector(concept)
            if vec:
                nearest_concepts = self._nearest_concepts_by_vector(
                    vec,
                    category_id=category_id,
                    top_k=top_k_concepts,
                    min_score=min_concept_score,
                )
            else:
                # At least return the exact match as #1
                nearest_concepts = [
                    ScoredItem(
                        id=_concept_id(concept),
                        label=_concept_text(concept),
                        score=1.0,
                        meta={"source": "exact_match"},
                    )
                ]
        else:
            # No concept match; return empty anchors/concepts.
            # Users can call infer_from_vector after embedding.
            nearest_concepts = []
            top_anchors = []

        return InferenceResult(
            query=q,
            category_id=category_id,
            top_anchors=top_anchors,
            nearest_concepts=nearest_concepts,
        )

    def infer_from_vector(
        self,
        vec: Sequence[float],
        *,
        category_id: Optional[str] = None,
        top_k_anchors: int = 5,
        top_k_concepts: int = 10,
        min_anchor_score: float = -1.0,
        min_concept_score: float = -1.0,
        query_label: str = "<vector>",
    ) -> InferenceResult:
        """Infer anchors and nearest concepts from an embedding vector."""

        v = _as_list_floats(vec)
        top_anchors = self._nearest_anchors_by_vector(
            v, top_k=top_k_anchors, min_score=min_anchor_score
        )
        nearest_concepts = self._nearest_concepts_by_vector(
            v,
            category_id=category_id,
            top_k=top_k_concepts,
            min_score=min_concept_score,
        )
        return InferenceResult(
            query=query_label,
            category_id=category_id,
            top_anchors=top_anchors,
            nearest_concepts=nearest_concepts,
        )

    # ----------
    # Internal ranking helpers
    # ----------

    def _nearest_anchors_by_vector(
        self,
        vec: Sequence[float],
        *,
        top_k: int,
        min_score: float,
    ) -> List[ScoredItem]:
        qv = list(vec)
        scored: List[ScoredItem] = []
        _p(f"[_nearest_anchors_by_vector] query_dim={len(qv)} anchors={len(self._anchors)} min_score={min_score}")
        missing_vec = 0
        dim_mismatch = 0
        cosine_err = 0
        kept = 0
        for a in self._anchors:
            aid = _anchor_id(a) or _anchor_name(a)
            aname = _anchor_name(a) or _anchor_id(a)
            av = _anchor_vector(a)
            if not av:
                missing_vec += 1
                _p(f"  - anchor {aname!r} ({aid}): SKIP (no vector)")
                continue
            if len(av) != len(qv):
                dim_mismatch += 1
                _p(f"  - anchor {aname!r} ({aid}): SKIP (dim mismatch {len(av)} != {len(qv)})")
                continue
            try:
                raw = float(self.cosine_fn(qv, av))
                s = float(self.similarity_transform(raw))
            except Exception as e:
                cosine_err += 1
                _p(f"  - anchor {aname!r} ({aid}): SKIP (cosine error: {type(e).__name__}: {e})")
                continue
            if s < min_score:
                _p(f"  - anchor {aname!r} ({aid}): score={s:.6f} (raw={raw:.6f}) -> filtered (< min_score)")
                continue
            kept += 1
            _p(f"  - anchor {aname!r} ({aid}): score={s:.6f} (raw={raw:.6f}) -> kept")
            scored.append(
                ScoredItem(
                    id=aid,
                    label=aname,
                    score=s,
                    meta={"source": "cosine(anchor.vector)", "raw_cosine": raw},
                )
            )
        _p(f"[_nearest_anchors_by_vector] summary: missing_vec={missing_vec} dim_mismatch={dim_mismatch} cosine_err={cosine_err} kept={kept}")
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: max(0, int(top_k))]

    def _nearest_concepts_by_vector(
        self,
        vec: Sequence[float],
        *,
        category_id: Optional[str],
        top_k: int,
        min_score: float,
    ) -> List[ScoredItem]:
        qv = list(vec)
        scored: List[ScoredItem] = []
        _p(f"[_nearest_concepts_by_vector] query_dim={len(qv)} concepts={len(self._concepts)} category_id={category_id!r} min_score={min_score}")
        missing_vec = 0
        dim_mismatch = 0
        cosine_err = 0
        kept = 0
        filtered_by_category = 0
        for c in self._concepts:
            cid = _concept_id(c)
            ctext = _concept_text(c)
            ccid = _concept_category_id(c)

            if category_id is not None:
                # If concept has no category recorded, don't exclude it.
                if ccid is not None and ccid != str(category_id):
                    filtered_by_category += 1
                    _p(f"  - concept {ctext!r} ({cid}) cat={ccid!r}: SKIP (category filter)")
                    continue

            cv = _concept_vector(c)
            if not cv:
                missing_vec += 1
                _p(f"  - concept {ctext!r} ({cid}) cat={ccid!r}: SKIP (no vector)")
                continue
            if len(cv) != len(qv):
                dim_mismatch += 1
                _p(f"  - concept {ctext!r} ({cid}) cat={ccid!r}: SKIP (dim mismatch {len(cv)} != {len(qv)})")
                continue
            try:
                raw = float(self.cosine_fn(qv, cv))
                s = float(self.similarity_transform(raw))
            except Exception as e:
                cosine_err += 1
                _p(f"  - concept {ctext!r} ({cid}) cat={ccid!r}: SKIP (cosine error: {type(e).__name__}: {e})")
                continue
            if s < min_score:
                _p(f"  - concept {ctext!r} ({cid}) cat={ccid!r}: score={s:.6f} (raw={raw:.6f}) -> filtered (< min_score)")
                continue

            kept += 1
            _p(f"  - concept {ctext!r} ({cid}) cat={ccid!r}: score={s:.6f} (raw={raw:.6f}) -> kept")
            scored.append(
                ScoredItem(
                    id=cid,
                    label=ctext,
                    score=s,
                    meta={"source": "cosine(concept.vector)", "category_id": ccid, "raw_cosine": raw},
                )
            )

        _p(f"[_nearest_concepts_by_vector] summary: filtered_by_category={filtered_by_category} missing_vec={missing_vec} dim_mismatch={dim_mismatch} cosine_err={cosine_err} kept={kept}")
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: max(0, int(top_k))]


# -----------------------------
# Optional CLI quick-test
# -----------------------------


def _try_build_default_ontology() -> Any:
    """Try to build ontology via builder in a script-friendly way.

    This file is often executed as:
      python src/ontology/infer.py
    In that case, relative imports like `from .builder import ...` fail because
    `__package__` is empty. We therefore try a few import strategies.
    """

    # 1) Package-relative import (works when run with -m, e.g. `python -m ontology.infer`)
    try:
        from .builder import build_ontology_default_paths  # type: ignore

        return build_ontology_default_paths()
    except Exception:
        pass

    # 2) Absolute import (works when `src` is on PYTHONPATH)
    try:
        from ontology.builder import build_ontology_default_paths  # type: ignore

        return build_ontology_default_paths()
    except Exception:
        pass

    # 3) Script execution fallback: add the `src` directory to sys.path
    try:
        import sys
        from pathlib import Path

        src_dir = Path(__file__).resolve().parents[1]  # .../src
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from ontology.builder import build_ontology_default_paths  # type: ignore

        return build_ontology_default_paths()
    except Exception:
        return None


if __name__ == "__main__":
    onto = _try_build_default_ontology()
    if onto is None:
        raise SystemExit(
            "Failed to build ontology. Try running from project root with: "
            "`PYTHONPATH=src python src/ontology/infer.py` or `python -m ontology.infer`."
        )

    # Use a 0..1 similarity transform for friendlier scores in quick-tests.
    r = Reasoner(onto, similarity_transform=_cosine_to_01)

    # Example: exact concept lookup
    res1 = r.infer_from_text("rage", category_id=None, top_k_anchors=3, top_k_concepts=5)
    print("[infer_from_text]", res1.query)
    print("  anchors:")
    for it in res1.top_anchors:
        print(f"    - {it.label}: {it.score:.3f} ({it.meta.get('source')})")
    print("  concepts:")
    for it in res1.nearest_concepts:
        print(f"    - {it.label}: {it.score:.3f}")

    # Example: vector-based inference (use a concept's vector as the query)
    c = r._concept_by_text.get("rage")
    if c is not None:
        v = _concept_vector(c)
        res2 = r.infer_from_vector(v, category_id=None, top_k_anchors=3, top_k_concepts=5, query_label="rage.vector")
        print("\n[infer_from_vector]", res2.query)
        print("  anchors:")
        for it in res2.top_anchors:
            print(f"    - {it.label}: {it.score:.3f}")
        print("  concepts:")
        for it in res2.nearest_concepts:
            print(f"    - {it.label}: {it.score:.3f}")
    def infer_concepts_as_dicts(
        self,
        vec: Sequence[float],
        *,
        category_id: Optional[str] = None,
        top_k: int = 10,
        min_score: float = -1.0,
        query_label: str = "<vector>",
    ) -> List[Dict[str, Any]]:
        """Return nearest concepts as plain dicts including score.

        This is primarily for integration points that expect a list of
        {id,label,score} objects.
        """
        res = self.infer_from_vector(
            vec,
            category_id=category_id,
            top_k_anchors=0,
            top_k_concepts=top_k,
            min_anchor_score=-1.0,
            min_concept_score=min_score,
            query_label=query_label,
        )
        return [it.to_dict(include_meta=False) for it in res.nearest_concepts]