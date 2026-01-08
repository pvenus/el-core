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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple
import hashlib
import re

from src.test.helper.vector import normalize_vector


# ---------------------------
# Pipeline interfaces
# ---------------------------

class Embedder(Protocol):
    def embed(self, text: str, dim: int) -> List[float]:
        ...


class Tagger(Protocol):
    def extract_tags(self, text: str) -> List[str]:
        ...


class ActionClassifier(Protocol):
    def classify_action_id(self, text: str, tags: List[str]) -> str:
        ...


# ---------------------------
# Default implementations (MVP)
# ---------------------------

class DeterministicHashEmbedder:
    """Deterministic pseudo-embedding.

    Replace later with your real embedding model.
    This keeps the rest of the system testable right now.

    Output is roughly in [-1, 1] then L2-normalized.
    """

    def embed(self, text: str, dim: int) -> List[float]:
        # Create enough bytes using repeated sha256 chaining
        seed = text.encode("utf-8")
        buf = b""
        h = seed
        while len(buf) < dim * 4:
            h = hashlib.sha256(h).digest()
            buf += h

        # Map to floats in [-1,1]
        out: List[float] = []
        for i in range(dim):
            chunk = buf[i * 4 : (i + 1) * 4]
            u = int.from_bytes(chunk, "little", signed=False)
            # [0, 2^32-1] -> [-1, 1]
            x = (u / 4294967295.0) * 2.0 - 1.0
            out.append(float(x))

        return normalize_vector(out)


class SimpleKeywordTagger:
    """Very small tagger.

    Replace later with ontology-backed tag extraction.
    """

    KEYWORDS = [
        ("breath", ["grounding", "calming"]),
        ("breathe", ["grounding", "calming"]),
        ("safe", ["comforting", "reassuring"]),
        ("here with you", ["comforting", "support"]),
        ("tell me", ["listening", "open"]),
        ("what happened", ["listening", "open"]),
        ("no filter", ["vent", "expressive"]),
        ("say it all", ["vent", "expressive"]),
        ("small step", ["plan", "problem_solving"]),
        ("one step", ["plan", "problem_solving"]),
        ("pause", ["avoid", "rest"]),
        ("later", ["avoid", "rest"]),
    ]

    def extract_tags(self, text: str) -> List[str]:
        t = text.lower()
        tags: List[str] = []
        for kw, tg in self.KEYWORDS:
            if kw in t:
                for x in tg:
                    if x not in tags:
                        tags.append(x)
        # keep it stable and short for MVP
        return tags[:4]


class SimpleActionClassifier:
    """Rule-based action classifier for MVP.

    Replace later with:
    - ontology mapping (tag->action)
    - small LLM classifier
    - hybrid (rules first, model fallback)
    """

    def classify_action_id(self, text: str, tags: List[str]) -> str:
        t = text.lower()
        # Strong textual signals
        if "breath" in t or "breathe" in t:
            return "breath"
        if "no filter" in t or "say it all" in t:
            return "vent"
        if "small step" in t or "one step" in t or "plan" in t:
            return "plan"
        if "pause" in t or "come back" in t or "later" in t:
            return "avoid"
        if "tell me" in t or "what happened" in t:
            return "reach_out"

        # Tag-based fallback
        if "vent" in tags:
            return "vent"
        if "plan" in tags or "problem_solving" in tags:
            return "plan"
        if "grounding" in tags or "calming" in tags:
            return "breath"
        if "rest" in tags or "avoid" in tags:
            return "avoid"
        if "listening" in tags:
            return "reach_out"

        return "breath"  # safe default


# ---------------------------
# Data models
# ---------------------------

@dataclass
class ChoiceArtifact:
    """Output of the maker pipeline (transport-friendly)."""

    choice_id: str
    round_id: int
    display_text: str
    embed_text: str
    tags: List[str]

    action_id: str
    duration: int
    magnitude: float
    embed_vec: List[float]

    effects: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None

    def to_choice_payload(self) -> Dict[str, Any]:
        """Return a payload compatible with Choice.create(...)

        NOTE: This returns only data. Actual object construction lives elsewhere.
        """
        return {
            "choice_id": self.choice_id,
            "display_text": self.display_text,
            "tags": list(self.tags),
            "constraints": self.constraints,
            "effects": dict(self.effects),
            "action": {
                "action_id": self.action_id,
                "duration": int(self.duration),
                "magnitude": float(self.magnitude),
                "embed_text": self.embed_text,
                "embed_vec": list(self.embed_vec),
            },
        }

    def to_choice_object(self) -> Any:
        """Best-effort conversion to a real Choice object if available."""
        try:
            # Adjust import path if your project keeps Choice elsewhere.
            from src.test.sim_scenario import Choice  # type: ignore
        except Exception:
            try:
                from sim_scenario import Choice  # type: ignore
            except Exception:
                return None

        return Choice.create(**self.to_choice_payload())


# ---------------------------
# Text normalization
# ---------------------------

def extract_embed_text(display_text: str) -> str:
    return str(display_text).strip()


# ---------------------------
# Main pipeline
# ---------------------------

class ChoiceMakerPipeline:
    def __init__(
        self,
        dim: int = 6,
        embedder: Optional[Embedder] = None,
        tagger: Optional[Tagger] = None,
        action_classifier: Optional[ActionClassifier] = None,
    ):
        self.dim = int(dim)
        self.embedder = embedder or DeterministicHashEmbedder()
        self.tagger = tagger or SimpleKeywordTagger()
        self.action_classifier = action_classifier or SimpleActionClassifier()

        # MVP defaults by action_id
        self._defaults: Dict[str, Tuple[int, float]] = {
            "breath": (1, 0.25),
            "reach_out": (1, 0.20),
            "plan": (1, 0.35),
            "vent": (2, 0.80),
            "avoid": (2, 0.50),
        }

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
          {"action_id": "vent", "duration": 3, "magnitude": 0.6, "tags": ["vent"]}
        """
        overrides = overrides or {}

        embed_text = str(overrides.get("embed_text") or extract_embed_text(display_text))
        tags = list(overrides.get("tags") or self.tagger.extract_tags(embed_text))
        action_id = str(overrides.get("action_id") or self.action_classifier.classify_action_id(embed_text, tags))

        duration, magnitude = self._defaults.get(action_id, (1, 0.25))
        duration = int(overrides.get("duration") or duration)
        magnitude = float(overrides.get("magnitude") or magnitude)

        embed_vec = list(overrides.get("embed_vec") or self.embedder.embed(embed_text, self.dim))
        # Ensure correct length
        if len(embed_vec) != self.dim:
            raise ValueError(f"embed_vec dim mismatch: expected {self.dim}, got {len(embed_vec)}")

        # MVP effects: keep it simple; record tags as state updates.
        effects: Dict[str, Any] = dict(overrides.get("effects") or {"add_tags": list(tags)})
        constraints = overrides.get("constraints")

        # Stable IDs if omitted
        if choice_id is None:
            # deterministic id based on text
            h = hashlib.sha1(embed_text.encode("utf-8")).hexdigest()[:6]
            choice_id = f"r{int(round_id)}_{action_id}_{h}"

        art = ChoiceArtifact(
            choice_id=str(choice_id),
            round_id=int(round_id),
            display_text=str(display_text),
            embed_text=str(embed_text),
            tags=list(tags),
            action_id=str(action_id),
            duration=int(duration),
            magnitude=float(magnitude),
            embed_vec=embed_vec,
            effects=effects,
            constraints=constraints if isinstance(constraints, dict) else None,
        )

        if debug:
            print("[MAKER] display_text=", display_text)
            print("[MAKER] embed_text=", embed_text)
            print("[MAKER] tags=", tags)
            print("[MAKER] action_id=", action_id)
            print("[MAKER] duration/magnitude=", duration, magnitude)
            print("[MAKER] embed_vec=", [round(x, 4) for x in embed_vec])
            print("[MAKER] effects=", effects)

        return art


# ---------------------------
# Quick manual test
# ---------------------------

if __name__ == "__main__":
    pipe = ChoiceMakerPipeline(dim=6)

    demo = "I'm here with you. Let's take a breath together. (나는 네 곁에 있어. 같이 숨 쉬자.)"
    art = pipe.make_choice(
        demo,
        round_id=1,
        overrides={"embed_text": "I'm here with you. Let's take a breath together."},
        debug=True,
    )

    print("\nChoice payload:")
    print(art.to_choice_payload())
