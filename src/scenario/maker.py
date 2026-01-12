"""maker.py

Single-step pipeline to turn a display text into a fully-specified Choice payload.

Goal (MVP):
- Input: display_text (e.g., "I'm here with you... (나는...)" or plain English)
- Output: a dict payload that can be passed to Choice.create(...) later

Design principles:
- Keep the pipeline deterministic (same text -> same output) for reproducible tests.
- Make every stage pluggable so you can gradually swap in:
  - real embeddings
  - ontology-backed tag extraction
  - LLM classifier for action_id
  - richer effects/constraints

This module does NOT force a dependency on sim_scenario.py types.
If `Choice` is importable, you can opt-in via `to_choice_object()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
import hashlib
import numpy as np
from pathlib import Path
from src.llm.embedding import embed_text_normalized
from src.simulation.dto.impact import Impact
from src.scenario.dto.choice_artifact import ChoiceArtifact


# ---------------------------
# Pipeline interfaces
# ---------------------------

class Embedder(Protocol):
    def embed(self, text: str, dim: int) -> List[float]:
        ...



class ConceptInferer(Protocol):
    def infer_concepts(
        self,
        vec: List[float],
        *,
        category_id: Optional[str] = None,
        top_k: int = 5,
        min_score: float = -1.0,
    ) -> List[Dict[str, Any]]:
        ...






class OntologyResolver(Protocol):
    def best_match(self, vec: List[float], *, category_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        ...

    def infer_concepts(self, vec: List[float], *, category_id: Optional[str] = None, top_k: int = 5, min_score: float = -1.0) -> List[Dict[str, Any]]:
        ...


# ---------------------------
# Default implementations (MVP)
# ---------------------------

class DeterministicHashEmbedder:
    """Deterministic pseudo-embedding (hash backend).

    Scenario/maker 단계에서 모델 없이도 재현 가능한 임베딩을 제공한다.
    실제 모델로 교체하더라도 embedding.py 내부 backend 확장으로 흡수한다.
    """

    def embed(self, text: str, dim: int) -> List[float]:
        vec = embed_text_normalized(text, dim=dim, backend="hash")
        out = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        if len(out) != int(dim):
            raise ValueError(f"hash embed dim mismatch: expected {dim}, got {len(out)}")
        return out


class LlamaCppEmbedder:
    """llama_cpp 기반 임베더.

    - backend="llama_cpp"를 사용하여 embedding.py가 (pooling + normalize)까지 처리한다.
    - llama_cpp가 설치되어 있지 않으면 ImportError가 발생한다.
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 131072,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        embedding: bool = True,
        verbose: bool = False,
    ):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise ImportError(
                "llama_cpp가 설치되어 있지 않습니다. `pip install llama-cpp-python` 후 다시 시도하세요."
            ) from e

        if not model_path:
            raise ValueError("model_path is required for LlamaCppEmbedder")
        self.model_path = resolve_model_path(model_path)

        # llama_cpp 기본값은 환경/플랫폼에 따라 다를 수 있어 명시적으로 지정
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=int(n_ctx),
            n_threads=n_threads,
            n_gpu_layers=int(n_gpu_layers),
            embedding=bool(embedding),
            verbose=bool(verbose),
        )

    def embed(self, text: str, dim: int) -> List[float]:
        # dim은 hash backend에서만 강제용으로 쓰고, llama_cpp에서는 참고값/검증값 정도로 둔다.
        vec = embed_text_normalized(text, dim=dim, backend="llama_cpp", llm=self.llm)
        # Defensive check: ensure we always return the requested dim
        if hasattr(vec, "shape") and len(getattr(vec, "shape")) == 1:
            if int(vec.shape[0]) != int(dim):
                raise ValueError(f"llama_cpp embed dim mismatch: expected {dim}, got {int(vec.shape[0])}")
        out = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        if len(out) != int(dim):
            raise ValueError(f"llama_cpp embed dim mismatch: expected {dim}, got {len(out)}")
        return out






class DefaultOntologyResolver:
    def infer_anchors_direction(
        self,
        vec: List[float],
        *,
        category_id: Optional[str] = None,
        top_k_anchors: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Optional[List[float]]:
        """Return an *anchor-score* direction vector.

        NOTE:
        - This does NOT try to reconstruct an embedding-space direction.
        - Instead, it returns the anchor similarity scores as a vector:
            direction[i] = score(anchor_ids[i])
        - The corresponding anchor ids (same order) are stored in
          `self._last_anchor_ids` for downstream use/debug.

        This matches the user's intent:
          joy score, anger score, ... -> direction vector
        """
        r = self._get_reasoner()
        if r is None:
            self._last_anchor_ids = []
            return None

        cat = category_id if category_id is not None else self._category_id
        tka = self._top_k_anchors if top_k_anchors is None else int(top_k_anchors)
        ms = self._min_score if min_score is None else float(min_score)

        try:
            res = r.infer_from_vector(
                vec,
                category_id=cat,
                top_k_anchors=tka,
                top_k_concepts=1,
                min_score=ms,
                query_label="maker.embed_vec",
            )
        except TypeError:
            try:
                res = r.infer_from_vector(
                    vec,
                    category_name=cat,
                    top_k_anchors=tka,
                    top_k_concepts=1,
                    min_score=ms,
                    query_label="maker.embed_vec",
                )
            except TypeError:
                try:
                    res = r.infer_from_vector(
                        vec,
                        category=cat,
                        top_k_anchors=tka,
                        top_k_concepts=1,
                        min_score=ms,
                        query_label="maker.embed_vec",
                    )
                except TypeError:
                    res = r.infer_from_vector(vec)

        anchors = None
        for attr in ("top_anchors", "anchors", "nearest_anchors", "anchor_scores"):
            if hasattr(res, attr):
                anchors = getattr(res, attr)
                break
        if not anchors:
            self._last_anchor_ids = []
            return None

        def _get_anchor_id(a: Any) -> Optional[str]:
            if isinstance(a, dict):
                return a.get("id") or a.get("anchor_id") or a.get("name") or a.get("label")
            return (
                getattr(a, "id", None)
                or getattr(a, "anchor_id", None)
                or getattr(a, "name", None)
                or getattr(a, "label", None)
            )

        def _get_score(item: Any) -> Optional[float]:
            s_val = None
            if isinstance(item, dict):
                s_val = item.get("score")
                if s_val is None:
                    s_val = item.get("similarity")
                if s_val is None:
                    s_val = item.get("sim")
                if s_val is None:
                    s_val = item.get("cosine")
                if s_val is None:
                    s_val = item.get("cos")
                if s_val is None:
                    s_val = item.get("raw")
                if s_val is None:
                    s_val = item.get("raw_score")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                s_val = item[1]
            else:
                s_val = getattr(item, "score", None)
                if s_val is None:
                    s_val = getattr(item, "similarity", None)
                if s_val is None:
                    s_val = getattr(item, "sim", None)
                if s_val is None:
                    s_val = getattr(item, "cosine", None)
                if s_val is None:
                    s_val = getattr(item, "cos", None)
                if s_val is None:
                    s_val = getattr(item, "raw", None)
                if s_val is None:
                    s_val = getattr(item, "raw_score", None)

            if s_val is None:
                return None
            try:
                return float(s_val)
            except Exception:
                return None

        anchor_ids: List[str] = []
        scores: List[float] = []

        for item in anchors:
            a_obj = item
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                a_obj = item[0]

            aid = _get_anchor_id(a_obj)
            sc = _get_score(item)
            if aid is None or sc is None:
                continue
            anchor_ids.append(str(aid))
            scores.append(float(sc))

        # Persist for consumers/debug
        self._last_anchor_ids = anchor_ids

        return scores
    """Ontology-backed nearest-node resolver (best-effort).

    It will try to import your existing ontology builder/reasoner.
    If not available, it safely returns None.
    """

    def __init__(
        self,
        *,
        ontology_obj: Any = None,
        category_id: Optional[str] = "emotion",
        top_k_anchors: int = 7,
        min_score: float = -1.0,
    ):
        self._ontology_obj = ontology_obj
        self._reasoner = None
        self._category_id = category_id
        self._top_k_anchors = int(top_k_anchors)
        self._min_score = float(min_score)

    def _get_reasoner(self) -> Any:
        if self._reasoner is not None:
            return self._reasoner

        # Lazy import so maker.py can run without ontology deps.
        try:
            # Prefer local package path
            from src.ontology.infer import Reasoner  # type: ignore
        except Exception:
            try:
                from ontology.infer import Reasoner  # type: ignore
            except Exception:
                return None

        onto = self._ontology_obj
        if onto is None:
            # Build default ontology if possible
            try:
                from src.ontology.builder import build_ontology_default_paths  # type: ignore
            except Exception:
                try:
                    from ontology.builder import build_ontology_default_paths  # type: ignore
                except Exception:
                    return None

            try:
                onto = build_ontology_default_paths()
                self._ontology_obj = onto
            except Exception:
                return None

        self._ontology_obj = onto
        try:
            self._reasoner = Reasoner(onto)
        except Exception:
            return None

        return self._reasoner

    def best_match(self, vec: List[float], *, category_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        r = self._get_reasoner()
        if r is None:
            return None

        cat = category_id if category_id is not None else self._category_id

        # Reasoner API compatibility layer: different projects may name args differently.
        try:
            res = r.infer_from_vector(
                vec,
                category_id=cat,
                top_k_anchors=self._top_k_anchors,
                top_k_concepts=1,
                min_score=self._min_score,
                query_label="maker.embed_vec",
            )
        except TypeError:
            try:
                # Some versions use category_name instead of category_id
                res = r.infer_from_vector(
                    vec,
                    category_name=cat,
                    top_k_anchors=self._top_k_anchors,
                    top_k_concepts=1,
                    min_score=self._min_score,
                    query_label="maker.embed_vec",
                )
            except TypeError:
                try:
                    # Some versions use category instead of category_id
                    res = r.infer_from_vector(
                        vec,
                        category=cat,
                        top_k_anchors=self._top_k_anchors,
                        top_k_concepts=1,
                        min_score=self._min_score,
                        query_label="maker.embed_vec",
                    )
                except TypeError:
                    # Last resort
                    res = r.infer_from_vector(vec)

        # Try to normalize result shape
        top = None
        for attr in ("top_concepts", "concepts", "nearest_concepts"):
            if hasattr(res, attr):
                top = getattr(res, attr)
                break
        if not top:
            return None

        item = top[0]
        # Common shapes: (Concept, score) or {'id':..., 'score':...}
        if isinstance(item, dict):
            cid = item.get("id") or item.get("concept_id")
            label = item.get("label") or item.get("text") or item.get("name")
            score = item.get("score")
            if score is None:
                # common alternates
                score = item.get("similarity")
            if score is None:
                score = item.get("sim")
            if score is None:
                score = item.get("cosine")
            if score is None:
                score = item.get("cos")
            if score is None:
                score = item.get("raw")
            if score is None:
                score = item.get("raw_score")
            if score is None:
                score = item.get("distance")
            if score is None:
                score = item.get("dist")

            try:
                score_f = float(score) if score is not None else None
            except Exception:
                score_f = None

            return {"id": cid, "label": label, "score": score_f}

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            c, score = item[0], item[1]
            cid = getattr(c, "id", None)
            label = getattr(c, "label", None) or getattr(c, "text", None) or getattr(c, "name", None)
            try:
                score_f = float(score)
            except Exception:
                score_f = None
            return {"id": cid, "label": label, "score": score_f}

        # Fallback: item itself may be a Concept
        c = item
        cid = getattr(c, "id", None)
        label = getattr(c, "label", None) or getattr(c, "text", None) or getattr(c, "name", None)
        score = getattr(c, "score", None)
        if score is None:
            score = getattr(c, "similarity", None)
        if score is None:
            score = getattr(c, "sim", None)
        if score is None:
            score = getattr(c, "cosine", None)
        if score is None:
            score = getattr(c, "cos", None)
        if score is None:
            score = getattr(c, "raw", None)
        if score is None:
            score = getattr(c, "raw_score", None)
        if score is None:
            score = getattr(c, "distance", None)
        if score is None:
            score = getattr(c, "dist", None)

        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None

        return {"id": cid, "label": label, "score": score_f}

    def infer_concepts(
        self,
        vec: List[float],
        *,
        category_id: Optional[str] = None,
        top_k: int = 5,
        min_score: float = -1.0,
    ) -> List[Dict[str, Any]]:
        r = self._get_reasoner()
        if r is None:
            return []

        cat = category_id if category_id is not None else self._category_id

        try:
            res = r.infer_from_vector(
                vec,
                category_id=cat,
                top_k_anchors=self._top_k_anchors,
                top_k_concepts=int(top_k),
                min_score=float(min_score),
                query_label="maker.embed_vec",
            )
        except TypeError:
            try:
                res = r.infer_from_vector(
                    vec,
                    category_name=cat,
                    top_k_anchors=self._top_k_anchors,
                    top_k_concepts=int(top_k),
                    min_score=float(min_score),
                    query_label="maker.embed_vec",
                )
            except TypeError:
                try:
                    res = r.infer_from_vector(
                        vec,
                        category=cat,
                        top_k_anchors=self._top_k_anchors,
                        top_k_concepts=int(top_k),
                        min_score=float(min_score),
                        query_label="maker.embed_vec",
                    )
                except TypeError:
                    res = r.infer_from_vector(vec)

        top = None
        for attr in ("top_concepts", "concepts", "nearest_concepts"):
            if hasattr(res, attr):
                top = getattr(res, attr)
                break
        if not top:
            return []

        out: List[Dict[str, Any]] = []
        for item in top:
            if isinstance(item, dict):
                cid = item.get("id") or item.get("concept_id")
                label = item.get("label") or item.get("text") or item.get("name")
                score = item.get("score")
                if score is None:
                    score = item.get("similarity")
                if score is None:
                    score = item.get("sim")
                if score is None:
                    score = item.get("cosine")
                if score is None:
                    score = item.get("cos")
                if score is None:
                    score = item.get("raw")
                if score is None:
                    score = item.get("raw_score")
                if score is None:
                    score = item.get("distance")
                if score is None:
                    score = item.get("dist")

                try:
                    score_f = float(score) if score is not None else None
                except Exception:
                    score_f = None

                out.append({"id": cid, "label": label, "score": score_f})
                continue

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                c, score = item[0], item[1]
                cid = getattr(c, "id", None)
                label = getattr(c, "label", None) or getattr(c, "text", None) or getattr(c, "name", None)
                try:
                    score_f = float(score)
                except Exception:
                    score_f = None
                out.append({"id": cid, "label": label, "score": score_f})
                continue

            c = item
            cid = getattr(c, "id", None)
            label = getattr(c, "label", None) or getattr(c, "text", None) or getattr(c, "name", None)
            score = getattr(c, "score", None)
            if score is None:
                score = getattr(c, "similarity", None)
            if score is None:
                score = getattr(c, "sim", None)
            if score is None:
                score = getattr(c, "cosine", None)
            if score is None:
                score = getattr(c, "cos", None)
            if score is None:
                score = getattr(c, "raw", None)
            if score is None:
                score = getattr(c, "raw_score", None)
            if score is None:
                score = getattr(c, "distance", None)
            if score is None:
                score = getattr(c, "dist", None)

            try:
                score_f = float(score) if score is not None else None
            except Exception:
                score_f = None

            out.append({"id": cid, "label": label, "score": score_f})

        return out

# ---------------------------
# Text normalization
# ---------------------------

def _project_root() -> Path:
    # maker.py: <root>/src/scenario/maker.py
    return Path(__file__).resolve().parents[2]

def resolve_model_path(model_path: str) -> str:
    mp = str(model_path or "").strip()
    if not mp:
        return mp

    p = Path(mp)
    if not p.is_absolute():
        p = _project_root() / p

    return str(p)

def extract_embed_text(display_text: str) -> str:
    return str(display_text).strip()


# ---------------------------
# Main pipeline
# ---------------------------

class ChoiceMakerPipeline:
    def __init__(
        self,
        model_path: str,
        dim: int = 2048,
        embedder: Optional[Embedder] = None,
        concept_inferer: Optional[ConceptInferer] = None,
        ontology_resolver: Optional[OntologyResolver] = None,
    ):
        self.dim = int(dim)

        # Embedder selection:
        # - If embedder is provided, use it.
        # - Else if model_path is a non-empty string, use LlamaCppEmbedder.
        # - Else, use DeterministicHashEmbedder.
        if embedder is not None:
            self.embedder = embedder
        elif model_path:
            self.embedder = LlamaCppEmbedder(model_path=model_path)
        else:
            self.embedder = DeterministicHashEmbedder()

        self.ontology_resolver = ontology_resolver or DefaultOntologyResolver()
        self.concept_inferer = concept_inferer or self.ontology_resolver
        self.rng = np.random.default_rng(42)

    def make_choice(
        self,
        display_text: str,
        *,
        choice_id: Optional[str] = None,
        round_id: int = 1,
        overrides: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> ChoiceArtifact:
        """Create one ChoiceArtifact from display_text.

        `overrides` can partially override derived fields, e.g.:
          {"embed_text": "...", "embed_vec": [...], "concepts": [...], "ontology_category_id": "...", "top_k_concepts": 5, "min_score": -1.0, "impact": {...}}
        """
        overrides = overrides or {}

        embed_text = str(overrides.get("embed_text") or extract_embed_text(display_text))

        embed_vec = list(overrides.get("embed_vec") or self.embedder.embed(embed_text, self.dim))
        # Ensure correct length
        if len(embed_vec) != self.dim:
            raise ValueError(f"embed_vec dim mismatch: expected {self.dim}, got {len(embed_vec)}")

        # Concepts (ontology inference)
        concepts = list(overrides.get("concepts") or [])
        if not concepts:
            try:
                onto_cat = overrides.get("ontology_category_id", "emotion")
                top_k = int(overrides.get("top_k_concepts") or 5)
                min_score = float(overrides.get("min_score") or -1.0)
                if onto_cat is not False:
                    try:
                        concepts = self.concept_inferer.infer_concepts(
                            embed_vec,
                            category_id=(None if onto_cat is None else str(onto_cat)),
                            top_k=top_k,
                            min_score=min_score,
                        )
                    except TypeError:
                        concepts = self.concept_inferer.infer_concepts(
                            embed_vec,
                            category_id=(None if onto_cat is None else str(onto_cat)),
                            top_k=top_k,
                        )
            except Exception:
                concepts = []
        # Normalize concept dicts: ensure `score` exists (float) and preserve 0.0
        if concepts and isinstance(concepts, list):
            norm: List[Dict[str, Any]] = []
            for c in concepts:
                if not isinstance(c, dict):
                    continue
                s = c.get("score")
                if s is None:
                    s = c.get("similarity")
                if s is None:
                    s = c.get("sim")
                if s is None:
                    s = c.get("cosine")
                if s is None:
                    s = c.get("cos")
                if s is None:
                    s = c.get("raw")
                if s is None:
                    s = c.get("raw_score")
                if s is None:
                    s = c.get("distance")
                if s is None:
                    s = c.get("dist")

                if s is not None:
                    try:
                        c["score"] = float(s)
                    except Exception:
                        pass
                norm.append(c)
            concepts = norm

        ontology_best = None
        try:
            # Allow override of ontology category (or disable by setting None explicitly)
            onto_cat = overrides.get("ontology_category_id", "emotion")
            if onto_cat is not False:  # allow disabling via False
                ontology_best = self.ontology_resolver.best_match(embed_vec, category_id=(None if onto_cat is None else str(onto_cat)))
        except Exception:
            ontology_best = None

        # Construct Impact.direction as an *anchor-score vector* (NOT an embedding vector)
        direction_list: Optional[List[float]] = None
        anchor_ids: List[str] = []
        try:
            if hasattr(self.ontology_resolver, "infer_anchors_direction"):
                direction_list = self.ontology_resolver.infer_anchors_direction(
                    embed_vec,
                    category_id=(
                        None
                        if overrides.get("ontology_category_id", "emotion") is None
                        else str(overrides.get("ontology_category_id", "emotion"))
                    ),
                    top_k_anchors=int(overrides.get("top_k_anchors", 7)),
                    min_score=float(overrides.get("min_score") or -1.0),
                )
                # Best-effort: fetch anchor id ordering if resolver stored it
                anchor_ids = list(getattr(self.ontology_resolver, "_last_anchor_ids", []) or [])
        except Exception:
            direction_list = None
            anchor_ids = []

        if isinstance(direction_list, list) and len(direction_list) > 0:
            direction = np.asarray(direction_list, dtype=np.float32)
            duration = max(1, int(float(direction.sum())))
        else:
            direction = np.zeros(0, dtype=np.float32)
            duration = 1
        impact = Impact(
            direction=direction,
            magnitude=1.0,
            duration=duration,
            profile={"ontology_best": ontology_best, "anchor_ids": anchor_ids},
        )

        # Stable IDs if omitted
        if choice_id is None:
            # deterministic id based on embed_text only
            h = hashlib.sha1(embed_text.encode("utf-8")).hexdigest()[:8]
            choice_id = f"r{int(round_id)}_{h}"

        art = ChoiceArtifact(
            choice_id=str(choice_id),
            round_id=int(round_id),
            display_text=str(display_text),
            embed_text=str(embed_text),
            concepts=list(concepts),
            embed_vec=embed_vec,
            ontology_best=ontology_best,
            impact=impact,
        )

        if debug:
            print("[MAKER] display_text=", display_text)
            print("[MAKER] embed_text=", embed_text)
            print("[MAKER] concepts=", [{"id": c.get("id"), "label": c.get("label"), "score": c.get("score")} if isinstance(c, dict) else c for c in concepts])
            print("[MAKER] ontology_best=", ontology_best)
            dir_head = impact.direction[:8].tolist() if isinstance(impact.direction, np.ndarray) else []
            print(
                "[MAKER] impact=",
                {
                    "magnitude": impact.magnitude,
                    "duration": impact.duration,
                    "delta_vars": impact.delta_vars,
                    "profile": impact.profile,
                    "direction_head": [float(x) for x in dir_head],
                },
            )
            print("[MAKER] embed_vec=", [round(x, 4) for x in embed_vec])

        return art


# ---------------------------
# Quick manual test
# ---------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChoiceMakerPipeline quick test")
    parser.add_argument(
        "--model-path",
        default=str(_project_root() / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        help="Path to a llama.cpp (GGUF) model for embeddings.",
    )
    parser.add_argument("--dim", type=int, default=2048, help="Embedding dimension")
    args = parser.parse_args()

    # Fail-fast: always try to load the requested llama.cpp model path.
    # If the file path is wrong, llama_cpp will raise an error (that's intended).
    pipe = ChoiceMakerPipeline(model_path=args.model_path, dim=args.dim)
    print(f"[maker.py] Using llama.cpp model: {args.model_path}")

    demo = "I'm here with you. Let's take a breath together. (나는 네 곁에 있어. 같이 숨 쉬자.)"
    art = pipe.make_choice(
        demo,
        round_id=1,
        overrides={"embed_text": "111"},
        debug=True,
    )

    print("\nChoice payload:")
    print(art.to_choice_payload())
