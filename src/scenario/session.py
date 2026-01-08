from .dto.session_state import SessionState
from .dto.choice import Choice
from .dto.selection_ctx import SelectionContext
from .manager import ScenarioManager
from typing import Any, Dict, List, Optional

class ScenarioSession:
    """Runs a scenario sequentially (no branching), while choices can still affect state.

    This satisfies:
      - Any choice in a round continues to the next round.
      - Choices can optionally modify emotion vector / tags via `Choice.effects`.

    Supported effect keys (all optional):
      - delta_vec: list[float]  -> added to current_vec
      - add_tags: list[str]     -> appended to memory_tags (unique, lowercase)
      - remove_tags: list[str]  -> removed from memory_tags
    """

    def __init__(self, manager: ScenarioManager, init_round_id: int, ctx: SelectionContext):
        self.manager = manager
        self.state = SessionState(
            round_id=int(init_round_id),
            turn=int(ctx.turn),
            current_vec=[float(x) for x in ctx.current_vec],
            comfort_vec=[float(x) for x in ctx.comfort_vec],
            memory_tags=[t.strip().lower() for t in (ctx.allowed_tags or []) if t and t.strip()],
            history=[],
        )

    def _apply_effects(self, choice: Choice) -> None:
        eff = choice.effects or {}

        # Vector update
        delta = eff.get("delta_vec")
        if isinstance(delta, list) and delta:
            for i in range(min(len(self.state.current_vec), len(delta))):
                self.state.current_vec[i] = float(self.state.current_vec[i]) + float(delta[i])

        # Tag memory update
        add_tags = eff.get("add_tags")
        if isinstance(add_tags, list):
            for t in add_tags:
                if not t:
                    continue
                tt = str(t).strip().lower()
                if tt and tt not in self.state.memory_tags:
                    self.state.memory_tags.append(tt)

        remove_tags = eff.get("remove_tags")
        if isinstance(remove_tags, list):
            rm = {str(t).strip().lower() for t in remove_tags if t and str(t).strip()}
            if rm:
                self.state.memory_tags = [t for t in self.state.memory_tags if t not in rm]

    def get_round_choices(self, top_k: int = 6) -> List[Dict[str, Any]]:
        ctx = SelectionContext(
            turn=self.state.turn,
            current_vec=self.state.current_vec,
            comfort_vec=self.state.comfort_vec,
            allowed_tags=self.state.memory_tags,
        )
        return self.manager.rank_choices(self.state.round_id, ctx, top_k=top_k)

    def step(self, choice_id: Optional[str] = None) -> Dict[str, Any]:
        """Advance one round.

        If choice_id is None, the manager will select the top-ranked choice.
        Returns a tool-friendly dict containing chosen line + next state.
        """
        ctx = SelectionContext(
            turn=self.state.turn,
            current_vec=self.state.current_vec,
            comfort_vec=self.state.comfort_vec,
            allowed_tags=self.state.memory_tags,
        )

        r = self.manager.scenario.get_round(self.state.round_id)

        chosen: Optional[Choice] = None
        if choice_id is not None:
            for c in r.choices:
                if c.choice_id == choice_id:
                    chosen = c
                    break
        if chosen is None:
            chosen = self.manager.select_choice(self.state.round_id, ctx)

        # Apply effects so conversation continues with updated state
        self._apply_effects(chosen)

        record = {
            "round_id": int(self.state.round_id),
            "choice_id": chosen.choice_id,
            "display_text": chosen.display_text,
            "tags": list(chosen.tags),
            "effects": dict(chosen.effects),
            "action": dict(chosen.action),
            "turn": int(self.state.turn),
            "current_vec": list(self.state.current_vec),
            "memory_tags": list(self.state.memory_tags),
        }
        self.state.history.append(record)

        # Sequential progression: next round regardless of choice
        next_round_id = self.state.round_id + 1
        has_next = any(rr.round_id == next_round_id for rr in self.manager.scenario.rounds)
        self.state.round_id = next_round_id if has_next else self.state.round_id
        self.state.turn += 1

        record["next_round_id"] = int(self.state.round_id) if has_next else None
        record["has_next"] = bool(has_next)
        return record