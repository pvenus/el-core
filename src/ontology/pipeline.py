from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.ontology.infer import (
    Reasoner,
    _cosine_to_01,
    _try_build_default_ontology,
)

from src.scenario.builder import build_simple_demo_scenario
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
        return mgr.select_choice_by_concepts(round_id, concepts, ctx=ctx)


def pipeline_quick_test():
    """Quick manual test:

    ontology -> concepts -> pick a choice in a specific round
    """
    pipe = OntologyConceptPipeline()

    mgr = build_simple_demo_scenario()

    demo = "rage"
    concepts = pipe.infer_concepts_from_text(demo, top_k=5)

    print("[infer_concepts_from_text]", demo)
    for it in concepts:
        print(f"  - {it['id']} / {it['label']}: {it['score']:.3f}")

    picked = pipe.choose_for_scenario(mgr, round_id=1, concepts=concepts)
    print("\n[picked]", getattr(picked, "choice_id", "<no_id>"), getattr(picked, "display_text", "<no_text>"))

    return picked


if __name__ == "__main__":
    picked = pipeline_quick_test()
    print(f"{picked}")