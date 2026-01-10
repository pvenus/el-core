from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

# Support both:
#   - package execution (e.g., `python -m src.scenario.manager`)
#   - direct script execution (e.g., `python src/scenario/manager.py`)
try:
    from .dto.scenario import Scenario
    from .dto.choice_artifact import ChoiceArtifact
    from .dto.selection_ctx import SelectionContext
except ImportError:  # pragma: no cover
    from src.scenario.dto.scenario import Scenario
    from src.scenario.dto.choice_artifact import ChoiceArtifact
    from src.scenario.dto.selection_ctx import SelectionContext

try:
    from src.simulation.dto.impact import Impact
except Exception:  # pragma: no cover
    Impact = None  # type: ignore

from src.helper.vector import l2_distance


def _get_attr(obj: Any, names: Sequence[str], default: Any) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


def _choice_tags(choice: Any) -> List[str]:
    """Best-effort tags extraction.

    New ChoiceArtifact no longer has explicit tags; we derive them from concepts when possible.
    Supported concept keys: id / concept_id / tag / name.
    """
    # Legacy explicit tags support
    v = _get_attr(choice, ["tags", "tag_ids", "tag", "labels"], None)
    if v is not None:
        try:
            return [str(x) for x in list(v) if str(x).strip()]
        except Exception:
            return []

    concepts = _get_attr(choice, ["concepts"], [])
    tags: List[str] = []
    if isinstance(concepts, list):
        for c in concepts:
            if not isinstance(c, dict):
                continue
            for k in ("id", "concept_id", "tag", "name"):
                if k in c and c[k] is not None:
                    s = str(c[k]).strip()
                    if s:
                        tags.append(s)
                    break
    # de-dupe but keep order
    seen = set()
    out: List[str] = []
    for t in tags:
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    return out


def _choice_constraints(choice: Any) -> Dict[str, Any]:
    # New ChoiceArtifact has no constraints; keep for backward compatibility.
    v = _get_attr(choice, ["constraints", "constraint"], None)
    return dict(v) if isinstance(v, dict) else {}


def _choice_action(choice: Any) -> Dict[str, Any]:
    """Action payload for front/tools.

    We standardize on action.embed_text derived from ChoiceArtifact.embed_text.
    """
    embed_text = _get_attr(choice, ["embed_text"], None)
    if isinstance(embed_text, str) and embed_text.strip():
        return {"embed_text": embed_text}

    # Legacy: if an explicit action/payload dict exists, pass it through.
    v = _get_attr(choice, ["action", "payload"], None)
    return dict(v) if isinstance(v, dict) else {}


class ScenarioManager:
    """Manages scenarios and selects choices per round."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

    # ---- IO (tool-friendly) ----

    def to_dict(self) -> Dict[str, Any]:
        s = self.scenario
        return {
            "scenario_id": s.scenario_id,
            "title": s.title,
            "rounds": [
                {
                    "round_id": int(r.round_id),
                    "choices": [
                        {
                            "choice_id": c.choice_id,
                            "display_text": c.display_text,
                            "concepts": list(getattr(c, "concepts", []) or []),
                            "ontology_best": (dict(getattr(c, "ontology_best", None)) if isinstance(getattr(c, "ontology_best", None), dict) else None),
                            "impact": (getattr(c, "impact", None).to_dict() if getattr(c, "impact", None) is not None and hasattr(getattr(c, "impact", None), "to_dict") else None),
                            "embed_text": getattr(c, "embed_text", ""),
                            "embed_vec": list(getattr(c, "embed_vec", []) or []),
                        }
                        for c in r.choices
                    ],
                }
                for r in s.rounds
            ],
        }

    def to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ScenarioManager":
        scenario = Scenario(
            scenario_id=str(d.get("scenario_id", "scenario")),
            title=str(d.get("title", "")),
        )

        for r in d.get("rounds", []):
            rs = RoundSpec(round_id=int(r["round_id"]))
            for c in r.get("choices", []):
                display_text = c.get("display_text", c.get("text", ""))
                embed_text = c.get("embed_text", display_text)
                concepts = c.get("concepts", []) or []
                embed_vec = c.get("embed_vec", []) or []
                ontology_best = c.get("ontology_best", None)
                impact_dict = c.get("impact", None)
                impact = None
                if isinstance(impact_dict, dict) and Impact is not None and hasattr(Impact, "from_dict"):
                    impact = Impact.from_dict(impact_dict)
                rs.add_choice(
                    ChoiceArtifact(
                        choice_id=str(c["choice_id"]),
                        round_id=int(r["round_id"]),
                        display_text=display_text,
                        embed_text=embed_text,
                        concepts=list(concepts) if isinstance(concepts, list) else [],
                        embed_vec=list(embed_vec) if isinstance(embed_vec, list) else [],
                        ontology_best=(dict(ontology_best) if isinstance(ontology_best, dict) else None),
                        impact=impact,
                    )
                )
            scenario.add_round(rs)

        return ScenarioManager(scenario)

    @staticmethod
    def from_json(s: str) -> "ScenarioManager":
        return ScenarioManager.from_dict(json.loads(s))

    # ---- core selection ----

    def normalize_tags(self, tags: Optional[Sequence[str]]) -> List[str]:
        if tags is None:
            return []
        return [t.strip().lower() for t in tags if t and t.strip()]

    def tags_match(self, choice: ChoiceArtifact, allowed_tags: Sequence[str]) -> bool:
        if not allowed_tags:
            return True
        choice_tags = set(t.lower() for t in _choice_tags(choice))
        allowed_set = set(allowed_tags)
        return bool(choice_tags.intersection(allowed_set))

    def _passes_constraints(self, choice: ChoiceArtifact, ctx: SelectionContext) -> bool:
        """Very small built-in constraint subset.

        Supported keys in choice.constraints:
          - min_turn, max_turn (inclusive)
          - min_dist_to_comfort, max_dist_to_comfort

        Other keys are ignored here (kept for external evaluators/reasoners).
        """
        cons = _choice_constraints(choice)
        t = int(ctx.turn)

        if "min_turn" in cons and t < int(cons["min_turn"]):
            return False
        if "max_turn" in cons and t > int(cons["max_turn"]):
            return False

        dist = l2_distance(ctx.current_vec, ctx.comfort_vec)
        if "min_dist_to_comfort" in cons and dist < float(cons["min_dist_to_comfort"]):
            return False
        if "max_dist_to_comfort" in cons and dist > float(cons["max_dist_to_comfort"]):
            return False

        return True

    def rank_choices(
        self,
        round_id: int,
        ctx: SelectionContext,
        top_k: int = 3,
        min_similarity: float = -1.0,
    ) -> List[Dict[str, Any]]:
        """Return ranked choices for a round.

        Scoring:
          - Tag-based filtering and scoring by number of matched tags.
          - If allowed_tags empty: all choices included with score=0.
          - Otherwise, only choices matching allowed_tags are included.

        Returns tool-friendly dict entries with score.
        """
        r = self.scenario.get_round(round_id)
        allowed_tags = self.normalize_tags(ctx.allowed_tags)

        candidates: List[ChoiceArtifact] = []
        for c in r.choices:
            if not self._passes_constraints(c, ctx):
                continue
            candidates.append(c)

        def _score(choice: ChoiceArtifact) -> int:
            if not allowed_tags:
                return 0
            c_tags_lower = set(t.lower() for t in _choice_tags(choice))
            return len(c_tags_lower.intersection(set(allowed_tags)))

        # First pass: if allowed_tags provided, try strict tag matching
        scored: List[Tuple[int, ChoiceArtifact]] = []
        if allowed_tags:
            for c in candidates:
                if not self.tags_match(c, allowed_tags):
                    continue
                scored.append((_score(c), c))

            # Graceful fallback: if nothing matched, return all candidates with score=0
            if not scored:
                scored = [(0, c) for c in candidates]
        else:
            scored = [(0, c) for c in candidates]

        scored.sort(key=lambda x: (-x[0], x[1].choice_id))
        out: List[Dict[str, Any]] = []
        for score, c in scored[: int(top_k)]:
            out.append(
                {
                    "choice_id": c.choice_id,
                    "display_text": c.display_text,
                    "score": float(score),
                    "concepts": list(getattr(c, "concepts", []) or []),
                    "ontology_best": (dict(getattr(c, "ontology_best", None)) if isinstance(getattr(c, "ontology_best", None), dict) else None),
                    "impact": (getattr(c, "impact", None).to_dict() if getattr(c, "impact", None) is not None and hasattr(getattr(c, "impact", None), "to_dict") else None),
                    "embed_text": getattr(c, "embed_text", ""),
                    "embed_vec": list(getattr(c, "embed_vec", []) or []),
                    # keep these for compatibility
                    "tags": _choice_tags(c),
                    "constraints": _choice_constraints(c),
                    "action": _choice_action(c),
                }
            )
        return out

    def select_choice(self, round_id: int, ctx: SelectionContext) -> ChoiceArtifact:
        ranked = self.rank_choices(round_id, ctx, top_k=1)
        if not ranked:
            # fallback: first choice, even if unscored
            r = self.scenario.get_round(round_id)
            if not r.choices:
                raise ValueError(f"Round {round_id} has no choices")
            return r.choices[0]

        top_id = ranked[0]["choice_id"]
        r = self.scenario.get_round(round_id)
        for c in r.choices:
            if c.choice_id == top_id:
                return c
        return r.choices[0]

    # ============================================================
    # Ontology concept–based selection (used by ontology.pipeline)
    # ============================================================

    def _get_round_obj(self, round_id: int):
        """Internal helper: return round object by id."""
        try:
            return self.scenario.get_round(round_id)
        except Exception:
            pass

        rounds = getattr(self.scenario, "rounds", [])
        for r in rounds:
            if getattr(r, "round_id", None) == round_id:
                return r

        raise KeyError(f"Round not found: {round_id}")

    @staticmethod
    def _concept_score_map(concepts):
        score_map = {}
        if not isinstance(concepts, list):
            return score_map

        for c in concepts:
            if not isinstance(c, dict):
                continue
            cid = c.get("id") or c.get("concept_id")
            if not cid:
                continue
            try:
                score_map[str(cid)] = float(c.get("score", 0.0))
            except Exception:
                score_map[str(cid)] = 0.0
        return score_map

    def rank_choices_by_concepts(
        self,
        round_id: int,
        concepts,
        *,
        ctx: Optional[SelectionContext] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Rank choices in a round by ontology concept overlap."""
        round_obj = self._get_round_obj(round_id)
        choices = getattr(round_obj, "choices", [])

        qmap = self._concept_score_map(concepts)

        allowed_tags = None
        if ctx is not None:
            allowed_tags = self.normalize_tags(getattr(ctx, "allowed_tags", None))

        ranked = []
        for c in choices:
            if ctx is not None and not self._passes_constraints(c, ctx):
                continue

            if allowed_tags:
                if not self.tags_match(c, allowed_tags):
                    continue

            score = 0.0
            c_concepts = getattr(c, "concepts", None)
            if isinstance(c_concepts, list):
                for cc in c_concepts:
                    if not isinstance(cc, dict):
                        continue
                    cid = cc.get("id") or cc.get("concept_id")
                    if cid and str(cid) in qmap:
                        score += qmap[str(cid)]

            ranked.append(
                {
                    "choice_id": c.choice_id,
                    "score": float(score),
                    "display_text": c.display_text,
                }
            )

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[: int(top_k)]

    def select_choice_by_concepts(
        self,
        round_id: int,
        concepts,
        *,
        ctx: Optional[SelectionContext] = None,
    ) -> ChoiceArtifact:
        """Select the best-matching choice by ontology concepts."""
        round_obj = self._get_round_obj(round_id)
        choices = getattr(round_obj, "choices", [])

        if not choices:
            raise ValueError(f"Round {round_id} has no choices")

        ranked = self.rank_choices_by_concepts(
            round_id, concepts, ctx=ctx, top_k=len(choices)
        )

        if not ranked:
            return choices[0]

        best_id = ranked[0]["choice_id"]
        for c in choices:
            if c.choice_id == best_id:
                return c

        return choices[0]
