from llama_cpp import Llama
from pathlib import Path
import json
from statistics import meanㄴㄴ


# ----------------------------------------
# 1. Llama 모델 로드
# ----------------------------------------
llm = Llama(
    model_path="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    embedding=True,
)


# ----------------------------------------
# 2. 임베딩할 감정/키워드 리스트
#    필요하면 계속 늘려도 됨
# ----------------------------------------
WORDS = [
    # Positive emotions
    "joy", "happiness", "delight", "pleasure", "love", "affection", "calm", "relief",
    "hope", "gratitude", "pride", "excitement", "enthusiasm", "amusement", "serenity",
    "satisfaction", "compassion", "trust", "admiration", "inspiration", "optimism",
    "confidence", "contentment", "friendship", "warmth", "kindness", "sympathy",
    "tenderness", "euphoria", "bliss", "cheerfulness", "radiance", "comfort",
    "encouragement", "supportiveness", "bravery", "courage", "self_love", "clarity",

    # Negative emotions
    "sadness", "grief", "sorrow", "despair", "depression", "loneliness", "remorse",
    "bitterness", "jealousy", "envy", "insecurity", "shame", "guilt", "humiliation",
    "boredom", "regret", "loss", "trauma", "shock", "abandonment", "isolation",
    "hopelessness", "melancholy", "emptiness", "misery", "resentment", "self_doubt",
    "worthlessness", "fatigue", "pain", "brokenness", "vulnerability", "heartbreak",

    # Anger family
    "anger", "rage", "irritation", "frustration", "resentment", "fury", "hostility",
    "annoyance", "indignation", "violence", "aggression", "wrath", "displeasure",
    "outrage", "offense", "provocation", "grudge", "hatred", "conflict",

    # Fear family
    "fear", "terror", "panic", "anxiety", "worry", "dread", "nervousness",
    "suspicion", "paranoia", "vulnerability", "helplessness", "phobia",
    "horror", "tension", "alertness", "unease", "startle", "timidity",
    "hesitation", "apprehension",

    # Disgust family
    "disgust", "aversion", "contempt", "revulsion", "rejection", "distaste",
    "loathing", "abhorrence", "disdain", "nausea",

    # Surprise family
    "surprise", "astonishment", "confusion", "disbelief", "amazement",
    "wonder", "curiosity", "intrigue", "shock_reaction", "uncertainty",

    # Anticipation / cognitive states
    "anticipation", "curiosity", "nostalgia", "uncertainty", "focus",
    "determination", "ambition", "motivation", "drive", "persistence",
    "critical_thinking", "reflection", "awareness", "perception",
    "intuition", "insight", "vigilance", "planning", "expectation",
    "prediction",

    # Survival / danger context
    "danger", "threat", "injury", "harm", "safety", "protection", "survival",
    "risk", "alert", "emergency", "failure", "collapse", "rescue", "escape",
    "fearlessness", "endurance", "resilience", "scarcity", "pressure", "burden",

    # Social concepts
    "success", "failure", "ambition", "determination", "exhaustion",
    "responsibility", "duty", "justice", "injustice", "leadership",
    "competition", "cooperation", "conflict", "betrayal", "loyalty",
    "support", "reconciliation", "dominance", "submission", "dependence",
    "freedom", "oppression", "authority", "rebellion", "obedience",
    "hierarchy", "unity", "division", "trustworthiness", "reliability",
    "influence", "prestige", "reputation", "identity", "connection",

    # Life events
    "birth", "death", "illness", "recovery", "separation", "reunion",
    "transformation", "growth", "learning", "awakening", "ending", "beginning",
    "transition", "renewal", "decline", "healing", "injury_event",
    "celebration", "tragedy", "sacrifice",

    # Personality / traits
    "introversion", "extroversion", "sensitivity", "stability", "chaos",
    "order", "discipline", "impulsiveness", "kindheartedness", "creativity",
    "logic", "rationality", "emotion", "stubbornness", "flexibility",
    "honesty", "deceit", "bravery_trait", "fearfulness", "dominance_trait",

    # Additional fine-grained emotion/psychological words
    "embarassment", "uneasiness", "admiration_deep", "longing", "yearning",
    "comforting", "soothing", "insecurity_deep", "self_blame", "pressure_high",
    "relaxation", "emptiness_deep", "compassion_deep", "affection_warm",
    "devotion", "attachment", "care", "pleading", "appeal", "submission_emotional"
]


# ----------------------------------------
# 3. 텍스트 임베딩 함수 (mean pooling)
# ----------------------------------------
def get_text_embedding(text: str):
    """
    llama_cpp.embed() 결과를 받아서
    토큰별 벡터가 여러 개면 mean pooling,
    하나면 그대로 단일 벡터로 변환한다.
    """
    emb = llm.embed(text)

    # 토큰별 벡터인지 단일 벡터인지 확인
    if emb and isinstance(emb[0], (list, tuple)):
        token_vecs = emb
    else:
        token_vecs = [emb]

    dim = len(token_vecs[0])

    # mean pooling: 각 차원별 평균값
    pooled = [mean(dim_vals) for dim_vals in zip(*token_vecs)]

    return {
        "raw": token_vecs,
        "pooled": pooled,
        "count": len(token_vecs),
        "dim": dim,
    }


# ----------------------------------------
# 4. 전체 단어 리스트 임베딩 처리
# ----------------------------------------
def main():
    results = {}

    for w in WORDS:
        print(f"Embedding: {w} ...")
        info = get_text_embedding(w)
        results[w] = {
            "count": info["count"],
            "dim": info["dim"],
            "pooled": info["pooled"],
        }

    # ----------------------------------------
    # 5. 결과 저장
    # ----------------------------------------
    out_dir = Path("data/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "emotion_words_llama3.2-1b.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                "words": list(results.keys()),
                "embeddings": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("====================================")
    print(f"Total words: {len(results)}")
    any_word = next(iter(results.values()))
    print(f"Vector dim: {any_word['dim']}")
    print(f"Saved to: {out_path}")
    print("====================================")


if __name__ == "__main__":
    main()