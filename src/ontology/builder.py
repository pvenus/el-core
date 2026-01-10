"""Ontology builder.

This module loads one or more concept JSON files (e.g., `total.json`) plus an optional
anchor definition JSON (e.g., `emotion_anchor.json`), then builds in-memory DTO objects.

Design goals
- Be tolerant to slightly different JSON shapes (list vs dict; different key names).
- Keep the builder independent from the rest of the app; it only assembles DTOs.
- Provide a single entrypoint: `build_ontology(...)`.

Expected JSON (flexible)

Anchors file (examples)
- List form:
  [
    {"id": "joy", "name": "joy", "category": "emotion", "seed_terms": ["joy", "delight"]},
    ...
  ]
- Dict form:
  {"category": "emotion", "anchors": [ ...same as above... ]}

Concepts file (examples)
- List form:
  [
    {"id": "relief", "text": "relief", "category": "emotion" , "anchor_scores": {"joy": 0.12, ...}},
    ...
  ]
- Dict form:
  {"concepts": [...], "relations": [...], "categories": [...]}

Relations (optional)
- {"src": "conceptA", "dst": "conceptB", "type": "similar_to", "weight": 0.83}

NOTE: The concrete DTO class names are assumed to be:
- Category (src/ontology/dto/category.py)
- AnchorAxis (src/ontology/dto/anchor_axis.py)
- Concept (src/ontology/dto/concept.py)
- Relation (src/ontology/dto/relation.py)
- Vector (src/ontology/dto/vector.py)

If your DTO field names differ, adjust the `_new_*` helper constructors below.
"""

from __future__ import annotations

import json
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys

# Allow running this module directly: `python src/ontology/builder.py`
# When executed as a script, relative imports (from .dto...) fail because there's no package context.
# We patch sys.path to include the project `src` directory, then re-import DTOs via absolute paths.
if __package__ in (None, ""):
    _this = Path(__file__).resolve()
    _src_root = _this.parents[1]  # .../src
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))

try:
    # Normal package import (e.g., `python -m src.front.main`)
    from .dto.category import Category
    from .dto.anchor_axis import AnchorAxis
    from .dto.concept import Concept
    from .dto.relation import Relation
    from .dto.vector import Vector
except ImportError:  # pragma: no cover
    # Script mode (e.g., `python src/ontology/builder.py`)
    from ontology.dto.category import Category
    from ontology.dto.anchor_axis import AnchorAxis
    from ontology.dto.concept import Concept
    from ontology.dto.relation import Relation
    from ontology.dto.vector import Vector


# ----------------------------
# Public result container
# ----------------------------


@dataclass
class OntologyBuildResult:
    """Built ontology objects (in-memory)."""

    categories: Dict[str, Category]
    anchor_axes: Dict[str, AnchorAxis]
    concepts: Dict[str, Concept]
    relations: List[Relation]


# ----------------------------
# Public API
# ----------------------------


def build_ontology(
    concept_json_paths: List[str | Path],
    anchor_json_path: str | Path | None = None,
    default_category_id: str = "emotion",
    anchor_category_name: str | None = None,
) -> OntologyBuildResult:
    """Build ontology from JSON files.

    Args:
        concept_json_paths: One or more JSON files containing concepts (e.g., total.json).
        anchor_json_path: Optional JSON file describing anchor axes (e.g., emotion_anchor.json).
        default_category_id: Used when a concept/anchor has no category.

    Returns:
        OntologyBuildResult with dicts keyed by ID.
    """

    categories: Dict[str, Category] = {}
    anchor_axes: Dict[str, AnchorAxis] = {}
    concepts: Dict[str, Concept] = {}
    relations: List[Relation] = []

    # 1) Load anchors (optional)
    if anchor_json_path is not None:
        anchor_data = _load_json(anchor_json_path)
        anchor_category_id, anchors = _normalize_anchor_payload(anchor_data, default_category_id)
        if anchor_category_name:
            anchor_category_id = anchor_category_name

        # Ensure category exists
        _ensure_category(categories, anchor_category_id)

        for a in anchors:
            axis = _new_anchor_axis(a, category_id=anchor_category_id)
            # last write wins (allows overriding)
            axis_key = (
                getattr(axis, "id", None)
                or getattr(axis, "axis_id", None)
                or getattr(axis, "name", None)
                or a.get("id")
                or a.get("axis_id")
                or a.get("name")
                or a.get("word")
            )
            if not axis_key:
                raise ValueError(f"Anchor axis missing id/name: {a}")
            anchor_axes[str(axis_key)] = axis

    # 2) Load concept files (can contain categories/relations too)
    for p in concept_json_paths:
        payload = _load_json(p)
        cats, cons, rels = _normalize_concept_payload(payload, default_category_id)

        for c in cats:
            cid = c.get("id") or c.get("category_id") or c.get("name")
            if cid:
                categories[cid] = _new_category(c, category_id=cid)

        # ensure default exists
        _ensure_category(categories, default_category_id)

        for c in cons:
            category_id = c.get("category") or c.get("category_id") or default_category_id
            _ensure_category(categories, category_id)

            concept = _new_concept(c, category_id=category_id)
            concept_id = getattr(concept, "id", None) or c.get("id") or c.get("concept_id") or c.get("text")
            if not concept_id:
                raise ValueError(f"Concept missing id/text in {p}: {c}")
            concepts[str(concept_id)] = concept

        for r in rels:
            relations.append(_new_relation(r))

    return OntologyBuildResult(
        categories=categories,
        anchor_axes=anchor_axes,
        concepts=concepts,
        relations=relations,
    )


# ----------------------------
# JSON loading + normalization
# ----------------------------


def _load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _normalize_anchor_payload(data: Any, default_category_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (category_id, anchors_list)."""
    if isinstance(data, list):
        anchors = [x for x in data if isinstance(x, dict)]
        category_id = _most_common(
            [a.get("category") or a.get("category_id") for a in anchors if (a.get("category") or a.get("category_id"))]
        ) or default_category_id
        return category_id, anchors

    if isinstance(data, dict):
        category_id = data.get("category") or data.get("category_id") or default_category_id
        anchors_raw = data.get("anchors") or data.get("anchor_axes") or data.get("items") or []
        if isinstance(anchors_raw, dict):
            anchors = [anchors_raw]
        else:
            anchors = [x for x in anchors_raw if isinstance(x, dict)]
        return category_id, anchors

    return default_category_id, []


def _normalize_concept_payload(data: Any, default_category_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (categories, concepts, relations)."""
    if isinstance(data, list):
        # list of concepts
        return [], [x for x in data if isinstance(x, dict)], []

    if isinstance(data, dict):
        categories = data.get("categories") or data.get("category") or []
        concepts = data.get("concepts") or data.get("items") or data.get("nodes") or []
        relations = data.get("relations") or data.get("edges") or []

        # If dict looks like a single concept
        if not concepts and ("text" in data or "word" in data or "id" in data or "concept_id" in data):
            concepts = [data]

        categories_list = categories if isinstance(categories, list) else [categories]
        concepts_list = concepts if isinstance(concepts, list) else [concepts]
        relations_list = relations if isinstance(relations, list) else [relations]

        return (
            [x for x in categories_list if isinstance(x, dict)],
            [x for x in concepts_list if isinstance(x, dict)],
            [x for x in relations_list if isinstance(x, dict)],
        )

    return [], [], []


def _most_common(values: List[str]) -> Optional[str]:
    if not values:
        return None
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]



# ----------------------------
# DTO constructors (adjust here if your field names differ)
# ----------------------------


# Helper: best-effort DTO constructor
def _construct_dto(cls, **kwargs):
    """Best-effort DTO constructor.

    We filter kwargs by the __init__ signature (when available) to avoid passing
    unexpected keyword arguments (e.g., DTOs that use `category_id` instead of `id`).

    Falls back to a few common patterns if signature inspection is not possible.
    """
    try:
        sig = inspect.signature(cls)
        allowed = set(sig.parameters.keys())
        # Remove `self` if present (it usually isn't for class callables, but be safe)
        allowed.discard("self")
        filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        return cls(**filtered)
    except Exception:
        # Fallback: try the most likely subsets
        for keys in (
            ("id", "name", "description"),
            ("category_id", "name", "description"),
            ("id", "name"),
            ("category_id", "name"),
            ("name", "description"),
            ("name",),
        ):
            attempt = {k: kwargs.get(k) for k in keys if kwargs.get(k) is not None}
            try:
                return cls(**attempt)
            except TypeError:
                continue
        # Last resort: positional with name or id
        for v in (kwargs.get("name"), kwargs.get("id"), kwargs.get("category_id")):
            if v is None:
                continue
            try:
                return cls(v)
            except TypeError:
                continue
        raise



def _ensure_category(categories: Dict[str, Category], category_id: str) -> None:
    if category_id in categories:
        return
    # Some DTOs use `id`, others use `category_id`, others only have `name`.
    categories[category_id] = _construct_dto(Category, id=category_id, category_id=category_id, name=category_id)



def _new_category(raw: Dict[str, Any], category_id: str) -> Category:
    name = raw.get("name") or raw.get("label") or category_id
    desc = raw.get("description") or raw.get("desc") or ""
    return _construct_dto(Category, id=category_id, category_id=category_id, name=name, description=desc)



def _new_anchor_axis(raw: Dict[str, Any], category_id: str) -> AnchorAxis:
    """Create an AnchorAxis DTO from a flexible anchor JSON record.

    We have seen these variants:
    - {"id": "joy", "name": "joy", "category": "emotion", "seed_terms": [...]}
    - {"name": "joy", ...}
    - {"word": "joy", "enabled": true, "embed_vec": [...], "dim": 2048}

    DTO field names differ across projects, so we always route construction through
    `_construct_dto(...)` which filters kwargs by the DTO's actual __init__ signature.
    """

    axis_id = (
        raw.get("id")
        or raw.get("axis_id")
        or raw.get("name")
        or raw.get("word")
        or raw.get("label")
    )
    if not axis_id:
        raise ValueError(f"Anchor axis missing id/name: {raw}")

    name = raw.get("name") or raw.get("label") or raw.get("word") or str(axis_id)

    seed_terms = raw.get("seed_terms") or raw.get("seeds") or raw.get("terms") or raw.get("seed") or []
    if isinstance(seed_terms, str):
        seed_terms = [seed_terms]
    if seed_terms is None:
        seed_terms = []

    enabled = raw.get("enabled")

    embedding = raw.get("embedding") or raw.get("embed_vec") or raw.get("vector") or raw.get("vec")

    return _construct_dto(
        AnchorAxis,
        # identity
        id=str(axis_id),
        axis_id=str(axis_id),
        name=str(name),
        label=str(name),
        # category
        category_id=str(category_id),
        category=str(category_id),
        category_name=str(category_id),
        # embedding (some DTOs require this)
        embedding=embedding,
        embed_vec=embedding,
        vector=embedding,
        # misc
        seed_terms=list(seed_terms),
        seeds=list(seed_terms),
        enabled=enabled,
    )


def _new_concept(raw: Dict[str, Any], category_id: str) -> Concept:
    # -----------------
    # Normalize flexible shapes
    # -----------------
    # Some generators emit: {"word": "rage", "enabled": true, "embed_vec": [...], "sims": {...}}
    # Map these into keys the builder already understands.
    if "text" not in raw and "word" in raw:
        raw = dict(raw)  # avoid mutating the caller's dict
        raw.setdefault("text", raw.get("word"))

    if "id" not in raw and "concept_id" not in raw:
        raw = dict(raw)
        raw.setdefault("id", raw.get("concept_id") or raw.get("key") or raw.get("text") or raw.get("word"))

    concept_id = raw.get("id") or raw.get("concept_id") or raw.get("key") or raw.get("text") or raw.get("word")
    text = raw.get("text") or raw.get("label") or raw.get("name") or raw.get("word") or concept_id
    if not concept_id or not text:
        raise ValueError(f"Concept missing id/text: {raw}")

    # anchor similarity / scores (optional)
    # Accept multiple aliases; `sims` is commonly emitted by embedding pipelines.
    anchor_scores = (
        raw.get("anchor_scores")
        or raw.get("scores")
        or raw.get("anchor_similarity")
        or raw.get("sims")
        or {}
    )
    if anchor_scores is None:
        anchor_scores = {}

    # vector / embedding (optional)
    # Accept `embed_vec` in addition to existing aliases.
    vec_raw = raw.get("vector") or raw.get("embedding") or raw.get("vec") or raw.get("embed_vec")
    vector_obj = None
    if isinstance(vec_raw, dict):
        values = vec_raw.get("values") or vec_raw.get("data") or vec_raw.get("v")
        if isinstance(values, list):
            vector_obj = _new_vector(values)
    elif isinstance(vec_raw, list):
        vector_obj = _new_vector(vec_raw)

    # Meta: keep useful flags without duplicating giant embedding arrays.
    meta: Dict[str, Any] = dict(raw.get("meta") or {})
    if "enabled" in raw and "enabled" not in meta:
        meta["enabled"] = raw.get("enabled")
    if "dim" in raw and "dim" not in meta:
        meta["dim"] = raw.get("dim")

    # -----------------
    # Construct DTO (signature-safe)
    # -----------------
    # Different projects name the concept text field differently (text/label/name/word/value).
    # Use `_construct_dto` so we only pass kwargs that exist in the Concept __init__.
    return _construct_dto(
        Concept,
        # identity
        id=str(concept_id),
        concept_id=str(concept_id),
        key=str(concept_id),
        # main string
        text=str(text),
        label=str(text),
        name=str(text),
        word=str(text),
        value=str(text),
        # category
        category_id=str(category_id),
        category=str(category_id),
        category_name=str(category_id),
        # optional attributes
        anchor_scores=dict(anchor_scores) if isinstance(anchor_scores, dict) else {},
        scores=dict(anchor_scores) if isinstance(anchor_scores, dict) else {},
        sims=dict(anchor_scores) if isinstance(anchor_scores, dict) else {},
        vector=vector_obj,
        embedding=vector_obj,
        embed_vec=vector_obj,
        meta=meta,
        enabled=meta.get("enabled"),
        dim=meta.get("dim"),
    )


def _new_vector(values: List[Any]):
    """Create a Vector DTO when Vector is a concrete class.

    Some codebases define `Vector` as a typing alias (e.g., `Vector = List[float]`).
    In that case, `Vector(...)` is not instantiable and we should just return the
    raw list of floats.
    """

    # Ensure float list
    vals = [float(x) for x in values]

    # If Vector is a typing alias (e.g., typing.List[float]) it won't be instantiable.
    # Detect non-class / typing constructs and return the raw list.
    if not isinstance(Vector, type):
        return vals

    # Otherwise, try to build the DTO class.
    try:
        return Vector(values=vals, dim=len(vals))  # type: ignore[arg-type]
    except TypeError:
        try:
            return Vector(values=vals)  # type: ignore[arg-type]
        except TypeError:
            # Last resort: some DTOs may accept positional values
            return Vector(vals)  # type: ignore[misc]


def _new_relation(raw: Dict[str, Any]) -> Relation:
    src = raw.get("src") or raw.get("source") or raw.get("from")
    dst = raw.get("dst") or raw.get("target") or raw.get("to")
    rtype = raw.get("type") or raw.get("relation") or raw.get("predicate") or "related_to"
    weight = raw.get("weight")

    if src is None or dst is None:
        raise ValueError(f"Relation missing src/dst: {raw}")

    try:
        return Relation(src=str(src), dst=str(dst), type=str(rtype), weight=float(weight) if weight is not None else None)  # type: ignore[arg-type]
    except TypeError:
        # Minimal fallback
        return Relation(src=str(src), dst=str(dst), type=str(rtype))  # type: ignore[arg-type]


def build_ontology_default_paths() -> OntologyBuildResult:
    """Build ontology using fixed default JSON paths.

    This is a convenience wrapper for local verification without CLI flags.

    Default files:
      - anchors : data/ontology_anchors.json
      - concepts: data/ontology_categories/total.json
    """
    # Resolve paths relative to project root (…/el-core)
    here = Path(__file__).resolve()
    project_root = here.parents[2]  # src/ontology/builder.py -> src -> project root

    anchor_path = project_root / "data" / "ontology_anchors.json"
    concept_path = project_root / "data" / "ontology_categories" / "total.json"

    return build_ontology(
        concept_json_paths=[concept_path],
        anchor_json_path=anchor_path,
        default_category_id="emotion",
        anchor_category_name="emotion",
    )


if __name__ == "__main__":
    res = build_ontology_default_paths()

    print("[OntologyBuilder] build ok")
    print(f"- categories : {len(res.categories)} -> {sorted(res.categories.keys())}")
    print(f"- anchors    : {len(res.anchor_axes)}")
    print(f"- concepts   : {len(res.concepts)}")
    print(f"- relations  : {len(res.relations)}")

    # Show a small sample
    sample_ids = list(res.concepts.keys())[:10]
    if sample_ids:
        print("- sample concepts:")
        for cid in sample_ids:
            c = res.concepts[cid]
            text = getattr(c, "text", None) or getattr(c, "label", None) or cid
            scores = getattr(c, "anchor_scores", None)
            if isinstance(scores, dict) and scores:
                top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
                top_str = ", ".join([f"{k}:{v:.3f}" for k, v in top])
                print(f"  * {cid} | {text} | top anchors: {top_str}")
            else:
                print(f"  * {cid} | {text}")
