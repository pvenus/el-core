from typing import Any, Dict, List, Optional, Sequence, Tuple
from .dto.scenario import Scenario
from .dto.choice import Choice
from .dto.selection_ctx import SelectionContext
from src.helper.vector import l2_distance
import json

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
            "dim": int(s.dim),
            "rounds": [
                {
                    "round_id": int(r.round_id),
                    "choices": [
                        {
                            "choice_id": c.choice_id,
                            "display_text": c.display_text,
                            "tags": list(c.tags),
                            "constraints": dict(c.constraints),
                            "effects": dict(c.effects),
                            "action": dict(c.action),
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
            dim=int(d.get("dim", 6)),
        )

        for r in d.get("rounds", []):
            rs = RoundSpec(round_id=int(r["round_id"]))
            for c in r.get("choices", []):
                display_text = c.get("display_text", c.get("text", ""))
                action = c.get("action", None)
                if not action:
                    # Backward compatibility: if legacy embed_text exists, store it in action.embed_text.
                    legacy_embed = c.get("embed_text", display_text)
                    action = {"embed_text": legacy_embed}
                elif isinstance(action, dict) and ("embed_text" not in action or not action.get("embed_text")):
                    action = dict(action)
                    action["embed_text"] = c.get("embed_text", display_text)
                rs.add_choice(
                    Choice.create(
                        choice_id=str(c["choice_id"]),
                        display_text=display_text,
                        tags=c.get("tags", None),
                        constraints=c.get("constraints", None),
                        effects=c.get("effects", None),
                        action=action,
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

    def tags_match(self, choice: Choice, allowed_tags: Sequence[str]) -> bool:
        if not allowed_tags:
            return True
        choice_tags = set(t.lower() for t in choice.tags)
        allowed_set = set(allowed_tags)
        return bool(choice_tags.intersection(allowed_set))

    def _passes_constraints(self, choice: Choice, ctx: SelectionContext) -> bool:
        """Very small built-in constraint subset.

        Supported keys in choice.constraints:
          - min_turn, max_turn (inclusive)
          - min_dist_to_comfort, max_dist_to_comfort

        Other keys are ignored here (kept for external evaluators/reasoners).
        """
        cons = choice.constraints or {}
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

        candidates: List[Choice] = []
        for c in r.choices:
            if not self._passes_constraints(c, ctx):
                continue
            candidates.append(c)

        def _score(choice: Choice) -> int:
            if not allowed_tags:
                return 0
            c_tags_lower = set(t.lower() for t in choice.tags)
            return len(c_tags_lower.intersection(set(allowed_tags)))

        # First pass: if allowed_tags provided, try strict tag matching
        scored: List[Tuple[int, Choice]] = []
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
                    "tags": list(c.tags),
                    "constraints": dict(c.constraints),
                    "action": dict(c.action),
                }
            )
        return out

    def select_choice(self, round_id: int, ctx: SelectionContext) -> Choice:
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
