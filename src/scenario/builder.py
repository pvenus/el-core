import json
from pathlib import Path
from typing import Union
from typing import Any, Dict, List, Optional

from src.scenario.dto.scenario import Scenario
from src.simulation.dto.impact import Impact
from src.scenario.dto.round_spec import RoundSpec
from src.scenario.dto.choice_artifact import ChoiceArtifact
from src.scenario.manager import ScenarioManager
from src.scenario.maker import ChoiceMakerPipeline

def build_simple_demo_scenario() -> ScenarioManager:
    """A tiny demo scenario built via ChoiceMakerPipeline.

    NOTE: This intentionally fail-fast loads the provided GGUF model path.
    """
    s = Scenario(scenario_id="simple_demo", title="Simple Demo")

    # Fail-fast: always try to load the requested llama.cpp model path.
    pipe = ChoiceMakerPipeline(model_path="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")

    r1 = RoundSpec(round_id=1)

    demo_1 = "I'm here with you. Let's take a breath together. (나는 네 곁에 있어. 같이 숨 쉬자.)"
    art_1 = pipe.make_choice(
        demo_1,
        round_id=1,
        overrides={"embed_text": "I'm here with you. Let's take a breath together."},
        debug=False,
    )

    demo_2 = "Tell me what happened—start anywhere. (무슨 일이 있었는지 말해줘. 어디서부터든 좋아.)"
    art_2 = pipe.make_choice(
        demo_2,
        round_id=1,
        overrides={"embed_text": "Tell me what happened. Start anywhere."},
        debug=False,
    )

    r1.add_choice(art_1)
    r1.add_choice(art_2)

    s.add_round(r1)
    return ScenarioManager(s)


def _project_root() -> Path:
    """Return repository root assuming this file is under src/scenario/."""
    # .../src/scenario/builder.py -> .../
    return Path(__file__).resolve().parents[2]


# Helper to safely convert impact payload to Impact object
def _impact_from_payload(payload: Any) -> Optional[Impact]:
    if payload is None:
        return None
    if isinstance(payload, Impact):
        return payload
    if isinstance(payload, dict):
        # Prefer Impact.from_dict if it exists; otherwise, keep as raw dict by returning None.
        from_dict = getattr(Impact, "from_dict", None)
        if callable(from_dict):
            try:
                return from_dict(payload)
            except Exception:
                return None
    return None


# Builder for transport-friendly dataclasses (Scenario -> RoundSpec -> ChoiceArtifact)
def build_scenario_artifact_from_items_json(
    json_path: Union[str, Path] = "data/scenario_items.json",
) -> Scenario:
    """Build a transport-friendly Scenario (Scenario -> RoundSpec -> ChoiceArtifact) from JSON.

    Supports these JSON formats:
    - Wrapper: {"items": [ ... ]}
    - Flat list: [ ... ]

    Each item may contain pipeline-export fields such as:
    - choice_id, round_id, display_text, embed_text, concepts, embed_vec
    - ontology_best, impact
    - enabled (false items are skipped)

    Scenario metadata (scenario_id/title/dim) is optional and read from the first item when present.
    """
    path = Path(json_path)
    if not path.is_absolute():
        path = _project_root() / path

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Unwrap wrapper format
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]

    if not isinstance(data, list):
        raise ValueError("Invalid JSON: expected list or {items: [...]} wrapper")

    scenario_id = "scenario_from_json"
    title = "Scenario From JSON"
    dim = 7

    if len(data) > 0 and isinstance(data[0], dict):
        first = data[0]
        scenario_id = str(first.get("scenario_id", scenario_id))
        title = str(first.get("title", title))
        if first.get("dim") is not None:
            dim = int(first["dim"])

    round_to_items: Dict[int, List[dict]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("enabled") is False:
            continue
        if item.get("round_id") is None:
            raise ValueError("Invalid JSON: each item must include 'round_id'")
        rid = int(item["round_id"])
        round_to_items.setdefault(rid, []).append(item)

    s = Scenario(scenario_id=scenario_id, title=title, dim=dim)

    for rid in sorted(round_to_items.keys()):
        rs = RoundSpec(round_id=rid)
        for it in round_to_items[rid]:
            if it.get("choice_id") is None or it.get("display_text") is None:
                raise ValueError(f"Invalid JSON: item in round {rid} must include 'choice_id' and 'display_text'")

            concepts = it.get("concepts")
            if concepts is None:
                concepts = []

            embed_text = it.get("embed_text")
            if embed_text is None:
                embed_text = str(it.get("display_text"))

            embed_vec = it.get("embed_vec")
            if embed_vec is None:
                embed_vec = []

            art = ChoiceArtifact(
                choice_id=str(it["choice_id"]),
                round_id=rid,
                display_text=str(it["display_text"]),
                embed_text=str(embed_text),
                concepts=list(concepts) if isinstance(concepts, list) else [],
                embed_vec=list(embed_vec) if isinstance(embed_vec, list) else [],
                ontology_best=(dict(it["ontology_best"]) if isinstance(it.get("ontology_best"), dict) else None),
                impact=_impact_from_payload(it.get("impact")),
            )
            rs.add_choice(art)

        s.add_round(rs)

    return s


def _merge_action_payload(choice_dict: dict) -> dict:
    """Merge action payload from JSON.

    This keeps backwards compatibility with older scenario JSON (which may already
    contain an `action` dict) while also preserving pipeline-exported fields like
    concepts / ontology / impact / embeddings.

    We store these extras inside `action` so downstream ranking/selection can use
    them without changing the `Choice` schema.
    """
    base = dict(choice_dict.get("action", {}) or {})

    # Common pipeline-export fields
    if choice_dict.get("embed_text") is not None:
        base.setdefault("embed_text", choice_dict.get("embed_text"))
    if choice_dict.get("embed_vec") is not None:
        base.setdefault("embed_vec", list(choice_dict.get("embed_vec") or []))

    if choice_dict.get("concepts") is not None:
        base.setdefault("concepts", list(choice_dict.get("concepts") or []))
    if choice_dict.get("ontology_best") is not None:
        base.setdefault("ontology_best", dict(choice_dict.get("ontology_best") or {}))
    if choice_dict.get("impact") is not None:
        base.setdefault("impact", dict(choice_dict.get("impact") or {}))

    # Optional flags
    if choice_dict.get("enabled") is not None:
        base.setdefault("enabled", bool(choice_dict.get("enabled")))

    return base


def build_scenario_from_items_json(
    json_path: Union[str, Path] = "data/scenario_items.json",
) -> ScenarioManager:
    """NOTE: This builder now produces ChoiceArtifact-based rounds (transport-friendly),
          not legacy Choice objects.

    Build a ScenarioManager from a JSON file.

    Expected JSON formats (either is OK):

    1) Object format:
       {
         "scenario_id": "demo",
         "title": "Demo Scenario",
         "dim": 6,
         "rounds": [
           {
             "round_id": 1,
             "choices": [
               {
                 "choice_id": "r1_a",
                 "display_text": "...",
                 "tags": ["..."],
                 "constraints": { ... },
                 "effects": { ... },
                 "action": { ... }
               }
             ]
           }
         ]
       }

    2) Flat list format (scenario metadata optional):
       [
         {
           "scenario_id": "demo",   # optional (only read from the first item)
           "title": "Demo Scenario", # optional
           "dim": 6,                 # optional
           "round_id": 1,
           "choice_id": "r1_a",
           "display_text": "...",
           "tags": ["..."],
           "constraints": { ... },
           "effects": { ... },
           "action": { ... }
         },
         ...
       ]

    3) Wrapper list format:
       {"items": [ ...same as flat list items... ]}

    Notes:
    - This builder intentionally constructs `Choice` objects (via Choice.create).
    - If you later want to generate `ChoiceArtifact` via the maker pipeline, do that as a separate step
      and store artifacts into the JSON.
    """
    path = Path(json_path)
    if not path.is_absolute():
        path = _project_root() / path

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Support wrapper format: {"items": [...]} (common pipeline export)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]

    # Normalize into: scenario_meta + list[(round_id, choice_dict)]
    scenario_id = "scenario_from_json"
    title = "Scenario From JSON"
    dim = 6

    round_to_choices: dict[int, list[dict]] = {}

    if isinstance(data, dict):
        scenario_id = str(data.get("scenario_id", scenario_id))
        title = str(data.get("title", title))
        if data.get("dim") is not None:
            dim = int(data["dim"])

        rounds = data.get("rounds", [])
        if not isinstance(rounds, list):
            raise ValueError("Invalid JSON: 'rounds' must be a list")

        for r in rounds:
            if not isinstance(r, dict):
                continue
            rid = int(r.get("round_id"))
            choices = r.get("choices", [])
            if not isinstance(choices, list):
                raise ValueError(f"Invalid JSON: round {rid} 'choices' must be a list")
            round_to_choices.setdefault(rid, []).extend([c for c in choices if isinstance(c, dict)])

    elif isinstance(data, list):
        if len(data) == 0:
            # Empty scenario
            s = Scenario(scenario_id=scenario_id, title=title, dim=dim)
            return ScenarioManager(s)

        # Allow scenario meta in the first item
        first = data[0] if isinstance(data[0], dict) else {}
        scenario_id = str(first.get("scenario_id", scenario_id))
        title = str(first.get("title", title))
        if first.get("dim") is not None:
            dim = int(first["dim"])

        for item in data:
            if not isinstance(item, dict):
                continue

            # Skip explicitly disabled items
            if item.get("enabled") is False:
                continue

            if item.get("round_id") is None:
                raise ValueError("Invalid JSON: list items must include 'round_id'")
            rid = int(item["round_id"])
            round_to_choices.setdefault(rid, []).append(item)

    else:
        raise ValueError("Invalid JSON: expected dict or list")

    s = Scenario(scenario_id=scenario_id, title=title, dim=dim)

    for rid in sorted(round_to_choices.keys()):
        rs = RoundSpec(round_id=rid)
        for c in round_to_choices[rid]:
            # Minimal required fields
            if "choice_id" not in c or "display_text" not in c:
                raise ValueError(f"Invalid JSON: choice in round {rid} must include 'choice_id' and 'display_text'")

            rs.add_choice(
                ChoiceArtifact(
                    choice_id=str(c["choice_id"]),
                    round_id=rid,
                    display_text=str(c["display_text"]),
                    embed_text=str(c.get("embed_text", c["display_text"])),
                    concepts=list(c.get("concepts", []) or []),
                    embed_vec=list(c.get("embed_vec", []) or []),
                    ontology_best=(dict(c["ontology_best"]) if isinstance(c.get("ontology_best"), dict) else None),
                    impact=_impact_from_payload(c.get("impact")),
                )
            )

        s.add_round(rs)

    return ScenarioManager(s)

if __name__ == "__main__":
    # Simple verification main: checks both artifact builder and legacy builder
    print("[builder] build_scenario_from_items_json quick check")

    json_path = "data/scenario_items.json"
    try:
        sim_s = build_scenario_artifact_from_items_json(json_path)
        total = sum(len(r.choices) for r in sim_s.rounds)
        print(f"[builder] artifact scenario: id={sim_s.scenario_id} title={sim_s.title} dim={sim_s.dim} rounds={len(sim_s.rounds)} choices={total}")

        mgr = build_scenario_from_items_json(json_path)
        print("[builder] scenario loaded successfully")
        print(mgr.to_json())

        # Optional: rank round 1 if available
        from src.scenario.dto.selection_ctx import SelectionContext

        ctx = SelectionContext(
            turn=1,
            current_vec=[0.0] * mgr.scenario.dim,
            comfort_vec=[0.0] * mgr.scenario.dim,
            allowed_tags=None,
        )

        if mgr.scenario.rounds:
            rid = mgr.scenario.rounds[0].round_id
            print(f"\n[builder] ranking round {rid}")
            for item in mgr.rank_choices(rid, ctx, top_k=5):
                print(f"- {item['choice_id']} score={item['score']:.3f} :: {item['display_text']}")
        else:
            print("[builder] no rounds found in scenario")

    except Exception as e:
        print("[builder] ERROR while building scenario:")
        raise