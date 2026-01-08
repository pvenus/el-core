from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

@dataclass(frozen=True)
class Choice:
    """One candidate reply the agent can say.

    Fields
    ------
    - choice_id: stable id for tools/content pipelines
    - display_text: text shown to the user
    - tags: optional labels (e.g., "comforting", "avoidant", "provoking")
    - constraints: lightweight rule slots for future use. This module only supports:
        * min_turn, max_turn (inclusive)
        * min_dist_to_comfort, max_dist_to_comfort
      Everything else is preserved for external evaluators.
    - effects: optional dict of effects to apply if this choice is selected
    - action: optional dict payload for EventRecord bridging (embed_text, duration, magnitude, embed_vec); embed_text lives here
    """

    choice_id: str
    display_text: str
    tags: Tuple[str, ...] = ()
    constraints: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        choice_id: str,
        display_text: str,
        tags: Optional[Sequence[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        effects: Optional[Dict[str, Any]] = None,
        action: Optional[Dict[str, Any]] = None,
    ) -> "Choice":
        act = dict(action or {})
        # If caller didn't provide action.embed_text, default to display_text.
        if "embed_text" not in act or not act.get("embed_text"):
            act["embed_text"] = str(display_text)

        return Choice(
            choice_id=str(choice_id),
            display_text=str(display_text),
            tags=tuple(tags or ()),
            constraints=dict(constraints or {}),
            effects=dict(effects or {}),
            action=act,
        )