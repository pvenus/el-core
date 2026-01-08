from .dto.scenario import Scenario
from .dto.round_spec import RoundSpec
from .dto.choice import Choice
from .manager import ScenarioManager

def build_demo_scenario(dim: int = 6) -> ScenarioManager:
    s = Scenario(scenario_id="demo", title="Demo Scenario", dim=dim)

    r1 = RoundSpec(round_id=1)
    r1.add_choice(
        Choice.create(
            choice_id="r1_a",
            display_text="I'm here with you. Let's take a breath together. (나는 네 곁에 있어. 같이 숨 쉬자.)",
            tags=["comforting", "grounding"],
            effects={"delta_vec": [-0.03, 0.05, -0.01, 0.02, 0.01, -0.01][:dim], "add_tags": ["comforting"]},
            action={"action_id": "breath", "embed_text": "I'm here with you. Let's take a breath together.", "duration": 1, "magnitude": 0.25, "embed_vec": [0.12, -0.08, 0.05, 0.02, -0.04, 0.09][:dim]},
        )
    )
    r1.add_choice(
        Choice.create(
            choice_id="r1_b",
            display_text="Tell me what happened—start anywhere. (무슨 일이 있었는지 말해줘. 어디서부터든 좋아.)",
            tags=["listening", "support"],
            effects={"add_tags": ["listening"]},
            action={"action_id": "reach_out", "embed_text": "Tell me what happened. Start anywhere.", "duration": 1, "magnitude": 0.20, "embed_vec": [0.05, 0.10, -0.02, 0.07, 0.01, -0.03][:dim]},
        )
    )
    r1.add_choice(
        Choice.create(
            choice_id="r1_c",
            display_text="Let's step back for a moment and name the feeling. (잠깐 물러서서 감정을 이름 붙여보자.)",
            tags=["reframe", "reflect"],
            effects={"add_tags": ["reframe"]},
            action={"action_id": "plan", "embed_text": "Step back and name the feeling.", "duration": 1, "magnitude": 0.30, "embed_vec": [-0.01, 0.08, 0.11, -0.04, 0.06, 0.02][:dim]},
        )
    )
    r1.add_choice(
        Choice.create(
            choice_id="r1_d",
            display_text="Do you want advice, or just someone to listen? (조언이 필요해, 아니면 그냥 들어줄까?)",
            tags=["clarify", "respect"],
            effects={"add_tags": ["clarify"]},
            action={"action_id": "reach_out", "embed_text": "Do you want advice or listening?", "duration": 1, "magnitude": 0.15, "embed_vec": [0.03, 0.04, 0.01, 0.05, -0.02, 0.00][:dim]},
        )
    )
    r1.add_choice(
        Choice.create(
            choice_id="r1_e",
            display_text="Whatever. It's not a big deal. (뭐 어때. 별일 아니잖아.)",
            tags=["dismissive", "avoidant"],
            effects={"delta_vec": [0.04, -0.03, 0.05, 0.02, -0.06, 0.01][:dim], "add_tags": ["avoidant"]},
            action={"action_id": "avoid", "embed_text": "Dismiss the situation.", "duration": 2, "magnitude": 0.45, "embed_vec": [0.18, -0.12, 0.09, 0.06, -0.10, 0.04][:dim]},
        )
    )
    s.add_round(r1)

    r2 = RoundSpec(round_id=2)
    r2.add_choice(
        Choice.create(
            choice_id="r2_a",
            display_text="Let's turn this into one small step you can do now. (지금 당장 할 수 있는 작은 한 걸음으로 바꿔보자.)",
            tags=["coach", "plan"],
            effects={"add_tags": ["coach"]},
            action={"action_id": "plan", "embed_text": "Turn this into one small step now.", "duration": 1, "magnitude": 0.35, "embed_vec": [0.06, 0.02, 0.10, -0.03, 0.08, 0.01][:dim]},
        )
    )
    r2.add_choice(
        Choice.create(
            choice_id="r2_b",
            display_text="Say it all—no filter. I'll take it. (필터 없이 다 말해. 내가 받아줄게.)",
            tags=["vent", "impulsive"],
            constraints={"min_dist_to_comfort": 0.2},
            effects={"add_tags": ["vent"], "delta_vec": [0.05, 0.01, -0.03, 0.02, -0.01, 0.0][:dim]},
            action={"action_id": "vent", "embed_text": "Say it all, no filter.", "duration": 2, "magnitude": 0.85, "embed_vec": [0.22, -0.15, -0.05, 0.12, -0.08, 0.03][:dim]},
        )
    )
    r2.add_choice(
        Choice.create(
            choice_id="r2_c",
            display_text="We can pause and come back later. You're allowed to rest. (잠깐 멈췄다가 나중에 다시 해도 돼. 쉬어도 돼.)",
            tags=["comforting", "rest"],
            effects={"delta_vec": [-0.03, 0.04, -0.01, 0.03, 0.01, -0.01][:dim], "add_tags": ["comforting"]},
            action={"action_id": "avoid", "embed_text": "Pause and come back later.", "duration": 2, "magnitude": 0.40, "embed_vec": [-0.09, 0.14, -0.04, 0.10, 0.05, -0.02][:dim]},
        )
    )
    r2.add_choice(
        Choice.create(
            choice_id="r2_d",
            display_text="What do you need most right now: safety, fairness, or control? (지금 가장 필요한 건 안전, 공정함, 통제 중 뭐야?)",
            tags=["probe", "reflect"],
            effects={"add_tags": ["reflect"]},
            action={"action_id": "plan", "embed_text": "Need safety, fairness, or control?", "duration": 1, "magnitude": 0.25, "embed_vec": [0.02, 0.09, 0.04, -0.01, 0.07, 0.02][:dim]},
        )
    )
    r2.add_choice(
        Choice.create(
            choice_id="r2_e",
            display_text="Walk away. You don't have to deal with this right now. (떠나자. 지금 당장 감당할 필요 없어.)",
            tags=["avoidant"],
            effects={"add_tags": ["avoidant"]},
            action={"action_id": "avoid", "embed_text": "Walk away for now.", "duration": 2, "magnitude": 0.55, "embed_vec": [0.16, -0.10, 0.07, 0.05, -0.06, 0.01][:dim]},
        )
    )
    s.add_round(r2)

    r3 = RoundSpec(round_id=3)
    r3.add_choice(
        Choice.create(
            choice_id="r3_a",
            display_text="We can take it slowly. You're safe right now. (천천히 해도 돼. 지금은 안전해.)",
            tags=["comforting", "reassurance"],
            effects={"delta_vec": [-0.02, 0.03, -0.01, 0.02, 0.01, -0.01][:dim]},
            action={"action_id": "breath", "embed_text": "Take it slowly. You're safe.", "duration": 1, "magnitude": 0.30, "embed_vec": [-0.07, 0.12, -0.03, 0.08, 0.04, -0.01][:dim]},
        )
    )
    r3.add_choice(
        Choice.create(
            choice_id="r3_b",
            display_text="Let's name the fear and pick one tiny action. (두려움을 이름 붙이고 아주 작은 행동 하나만 고르자.)",
            tags=["coach", "reframe"],
            effects={"add_tags": ["coach", "reframe"]},
            action={"action_id": "plan", "embed_text": "Name the fear and pick one tiny action.", "duration": 1, "magnitude": 0.50, "embed_vec": [0.08, 0.05, 0.11, -0.04, 0.09, 0.03][:dim]},
        )
    )
    r3.add_choice(
        Choice.create(
            choice_id="r3_c",
            display_text="If it's too much, we can park it for tonight. (너무 힘들면 오늘은 잠깐 접어두자.)",
            tags=["rest", "boundary"],
            effects={"add_tags": ["rest"]},
            action={"action_id": "avoid", "embed_text": "Park it for tonight.", "duration": 2, "magnitude": 0.35, "embed_vec": [-0.10, 0.09, -0.02, 0.07, 0.06, -0.03][:dim]},
        )
    )
    r3.add_choice(
        Choice.create(
            choice_id="r3_d",
            display_text="I can be blunt if you want—what's the hardest truth here? (원하면 직설적으로 말할게. 여기서 가장 힘든 진실은 뭐야?)",
            tags=["direct", "probe"],
            effects={"add_tags": ["direct"]},
            action={"action_id": "vent", "embed_text": "What's the hardest truth here?", "duration": 1, "magnitude": 0.40, "embed_vec": [0.12, 0.00, 0.06, -0.02, 0.05, 0.01][:dim]},
        )
    )
    r3.add_choice(
        Choice.create(
            choice_id="r3_e",
            display_text="Let's write two lines: what you can control, and what you can't. (두 줄로 적자: 통제 가능한 것 / 불가능한 것.)",
            tags=["reframe", "plan"],
            effects={"add_tags": ["plan", "reframe"]},
            action={"action_id": "plan", "embed_text": "Write what you can control vs can't.", "duration": 1, "magnitude": 0.30, "embed_vec": [0.05, 0.07, 0.09, -0.03, 0.08, 0.02][:dim]},
        )
    )
    s.add_round(r3)

    return ScenarioManager(s)