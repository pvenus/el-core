from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.ontology.infer import (
    Reasoner,
    _cosine_to_01,
    _try_build_default_ontology,
)

from src.scenario.builder import build_simple_demo_scenario, build_scenario_artifact_from_items_json
from src.scenario.manager import ScenarioManager

LOG = logging.getLogger("ontology.pipeline")


@dataclass
class ConceptMatch:
    """Transport-friendly concept match."""

    concept_id: str
    label: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.concept_id, "label": self.label, "score": float(self.score)}


class OntologyConceptPipeline:
    """Small, reusable pipeline to infer ontology concepts.

    This is intended as the first half of a larger flow:

        text/embedding -> concepts -> (later) scenario choice selection

    The scenario-selection step is intentionally NOT implemented here.
    """

    def __init__(
        self,
        *,
        similarity_transform=_cosine_to_01,
    ):
        onto = _try_build_default_ontology()
        if onto is None:
            raise SystemExit(
                "Failed to build ontology. Try running from project root with: "
                "`PYTHONPATH=src python src/ontology/infer.py` or `python -m ontology.infer`."
            )
        self.reasoner = Reasoner(onto, similarity_transform=similarity_transform)

    def infer_concepts_from_text(
        self,
        text: str,
        *,
        category_id: Optional[str] = None,
        top_k: int = 10,
        min_score: float = -1.0,
    ) -> List[Dict[str, Any]]:
        """Infer nearest concepts from free text.

        Returns a list of plain dicts: {id,label,score}.
        """
        res = self.reasoner.infer_from_text(
            text,
            category_id=category_id,
            top_k_anchors=0,
            top_k_concepts=top_k,
            min_anchor_score=-1.0,
            min_concept_score=min_score,
        )
        # `to_dict(include_meta=False)` is already compatible with {id,label,score}
        return [it.to_dict(include_meta=False) for it in res.nearest_concepts]

    def infer_concepts_from_vector(
        self,
        vec: Sequence[float],
        *,
        category_id: Optional[str] = None,
        top_k: int = 10,
        min_score: float = -1.0,
        query_label: str = "<vector>",
    ) -> List[Dict[str, Any]]:
        """Infer nearest concepts from an embedding vector.

        Returns a list of plain dicts: {id,label,score}.
        """
        res = self.reasoner.infer_from_vector(
            vec,
            category_id=category_id,
            top_k_anchors=0,
            top_k_concepts=top_k,
            min_anchor_score=-1.0,
            min_concept_score=min_score,
            query_label=query_label,
        )
        return [it.to_dict(include_meta=False) for it in res.nearest_concepts]

    def infer_concepts_from_anchor_scores(
        self,
        anchor_scores: Dict[str, float] | Sequence[float],
        *,
        category_id: Optional[str] = None,
        top_k: int = 10,
        min_score: float = -1.0,
        normalize: bool = True,
        query_label: str = "<anchor_scores>",
    ) -> List[Dict[str, Any]]:
        """Infer nearest concepts from 7-axis anchor scores.

        `anchor_scores` may be either:
          - dict: {"joy": 0.1, "anger": 0.8, ...}
          - list/tuple: aligned to the ontology anchor order used by Reasoner

        Returns a list of plain dicts: {id,label,score}.
        """
        res = self.reasoner.infer_from_anchor_scores(
            anchor_scores,
            category_id=category_id,
            top_k_concepts=top_k,
            min_concept_score=min_score,
            l2_normalize_query=normalize,
            query_label=query_label,
        )

        return [it.to_dict(include_meta=False) for it in res.nearest_concepts]

    # ------------------------------------------------------------
    # Scenario selection (concept-based)
    # ------------------------------------------------------------
    def choose_for_scenario(
        self,
        mgr: ScenarioManager,
        *,
        round_id: int,
        text: Optional[str] = None,
        concepts: Optional[List[Dict[str, Any]]] = None,
        category_id: Optional[str] = None,
        top_k: int = 5,
        min_score: float = -1.0,
        ctx: Any = None,
    ):
        """Pick a choice for a specific round using ontology concept overlap.

        This wires ONLY: (text/vec -> concepts) -> (concept overlap) -> pick from round.
        It intentionally does not use vectors/embeddings from the scenario manager.

        Args:
            mgr: ScenarioManager
            round_id: target round
            text: if provided, concepts will be inferred from this text
            concepts: if provided, uses these {id,label,score} dicts directly
            category_id/top_k/min_score: forwarded to inference when `text` is used
            ctx: optional SelectionContext-like object; if it has `allowed_tags`, we will filter when choices expose tags

        Returns:
            The selected choice object from the manager (whatever type the manager stores).
        """
        if concepts is None:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Provide either `text` or `concepts`.")
            concepts = self.infer_concepts_from_text(
                text,
                category_id=category_id,
                top_k=top_k,
                min_score=min_score,
            )

        # Delegate scoring/picking to ScenarioManager helper.
        # `build_scenario_artifact_from_items_json()` may return a raw Scenario artifact; wrap if needed.
        if not hasattr(mgr, "select_choice_by_concepts"):
            try:
                mgr = ScenarioManager(mgr)  # type: ignore[arg-type]
            except Exception:
                # Try common alternative constructor names
                ctor = getattr(ScenarioManager, "from_scenario", None)
                if callable(ctor):
                    mgr = ctor(mgr)  # type: ignore[misc]
                else:
                    raise AttributeError(
                        "Expected ScenarioManager with select_choice_by_concepts(), "
                        "but got an object without that method. "
                        "Wrap the scenario with ScenarioManager(scenario)."
                    )

        return mgr.select_choice_by_concepts(round_id, concepts, ctx=ctx)


def pipeline_quick_test(anchor_scores: Dict[str, float] | Sequence[float]):
    """Quick manual test:

    ontology -> concepts -> pick a choice in a specific round
    """
    scenario = build_scenario_artifact_from_items_json()
    try:
        mgr = ScenarioManager(scenario)
    except Exception:
        ctor = getattr(ScenarioManager, "from_scenario", None)
        if callable(ctor):
            mgr = ctor(scenario)
        else:
            raise

    concepts = pipe.infer_concepts_from_anchor_scores(anchor_scores, top_k=8)

    print("[infer_concepts_from_anchor_scores]", anchor_scores)
    for it in concepts:
        print(f"  - {it['id']} / {it['label']}: {it['score']:.3f}")

    picked = pipe.choose_for_scenario(mgr, round_id=1, concepts=concepts)
    print("\n[picked]", getattr(picked, "choice_id", "<no_id>"), getattr(picked, "display_text", "<no_text>"))

    return picked


if __name__ == "__main__":
    anchor_scores = [
        0.10,
        0.80,
        0.05,
        0.10,
        0.00,
        0.20,
        0.05,
    ]
    pipe = OntologyConceptPipeline()
    picked = pipeline_quick_test(anchor_scores=anchor_scores)
    print(f"{picked}")