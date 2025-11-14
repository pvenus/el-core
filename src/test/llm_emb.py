from llama_cpp import Llama
from pathlib import Path
import json

# ----------------------------------------
# 1. 모델 로드
# ----------------------------------------
llm = Llama(
    model_path="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    embedding=True,
)

# ----------------------------------------
# 2. 입력 텍스트
# ----------------------------------------
input_text = "death"

# ----------------------------------------
# 3. 임베딩 생성
# ----------------------------------------
emb = llm.embed(input_text)

# ----------------------------------------
# 4. 벡터 구조 파악
# ----------------------------------------
if emb and isinstance(emb[0], (list, tuple)):
    count = len(emb)         # 토큰 개수
    dim = len(emb[0])        # 각 벡터 차원 수
else:
    count = 1
    dim = len(emb)

# ----------------------------------------
# 5. 데이터 저장 경로
# ----------------------------------------
out_dir = Path("data/embeddings")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / f"{input_text}_embedding.json"

# ----------------------------------------
# 6. JSON 파일로 저장
# ----------------------------------------
with out_path.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "input": input_text,
            "count": count,
            "dim": dim,
            "embedding": emb,
        },
        f,
        ensure_ascii=False,
        indent=2
    )

# ----------------------------------------
# 7. 콘솔 출력 (요약)
# ----------------------------------------
print("====================================")
print(f"Input: {input_text}")
print(f"Embedding saved to: {out_path}")
print(f"Shape: {count} x {dim}")
print("====================================")