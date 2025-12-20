from llama_cpp import Llama
from pathlib import Path
import json

MODEL_PATH = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# ----------------------------------------
# 1. 모델 로드
# ----------------------------------------
llm = Llama(
    model_path=MODEL_PATH,
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
# 간단한 통계 정보 (min/max) 계산
def _iter_values(e):
    if e and isinstance(e[0], (list, tuple)):
        for row in e:
            for v in row:
                yield v
    else:
        for v in e:
            yield v

try:
    vals = list(_iter_values(emb))
    v_min = min(vals) if vals else None
    v_max = max(vals) if vals else None
except Exception:
    v_min, v_max = None, None

with out_path.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "input": input_text,
            "count": count,
            "dim": dim,
            "model_path": MODEL_PATH,
            "stats": {
                "min": v_min,
                "max": v_max,
            },
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
print("[EMB] Input            :", input_text)
print("[EMB] Model            :", MODEL_PATH)
print(f"[EMB] Shape            : {count} x {dim}")
print("[EMB] Output JSON      :", out_path)
if v_min is not None and v_max is not None:
    print(f"[EMB] Value range      : min={v_min:.6f}, max={v_max:.6f}")
# 첫 번째 벡터의 앞 몇 개 값만 샘플로 출력
if emb:
    if isinstance(emb[0], (list, tuple)):
        sample_vec = emb[0]
    else:
        sample_vec = emb
    print("[EMB] Sample (first 5) :", sample_vec[:5])
print("====================================")