# src/llm_emb/emb_io.py

import csv
import json
from pathlib import Path
from typing import List, Dict, Any


# ---------------------------------------------------------
# TXT 파일 처리
# ---------------------------------------------------------

def _load_txt(path: Path) -> List[str]:
    """한 줄 = 하나의 단어/문장 형식의 TXT 파일 로드."""
    words = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                words.append(line)
    return words


# ---------------------------------------------------------
# CSV 파일 처리
# ---------------------------------------------------------

def _load_csv(path: Path) -> List[str]:
    """
    CSV 파일 로드.
    - text / word / content 컬럼 우선 사용
    - 없으면 첫 번째 컬럼 사용
    - 나머지 칼럼은 무시
    """
    preferred_cols = ["text", "word", "content"]

    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return rows

        # 우선 text/word/content 중 하나를 찾는다
        selected = None
        for col in preferred_cols:
            if col in reader.fieldnames:
                selected = col
                break

        # 못 찾으면 첫 번째 컬럼 사용
        if selected is None:
            selected = reader.fieldnames[0]

        for row in reader:
            val = row.get(selected, "").strip()
            if val:
                rows.append(val)

    return rows


# ---------------------------------------------------------
# JSON 파일 처리
# ---------------------------------------------------------

def _load_json(path: Path) -> List[str]:
    """
    JSON 파일 로드.

    지원 형식:
      1) ["happy", "sad", ...]               → 리스트(스칼라)
      2) {"words": ["happy", "sad", ...]}    → words 배열
      3) [{"text": "happy", "category": ...}, ...]
         [{"word": "happy", ...}], [{"content": "...", ...}]
         → CSV와 동일한 스키마(행 리스트)도 지원
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    preferred_cols = ["text", "word", "content"]

    # 1) 리스트 형태
    if isinstance(data, list):
        if not data:
            return []

        # 1-1) 리스트의 원소가 dict인 경우 → CSV와 비슷한 구조로 간주
        if isinstance(data[0], dict):
            rows: List[str] = []
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                for col in preferred_cols:
                    if col in obj and str(obj[col]).strip():
                        rows.append(str(obj[col]).strip())
                        break
            return rows

        # 1-2) 리스트의 원소가 스칼라(문자열/숫자 등)
        return [str(x).strip() for x in data if str(x).strip()]

    # 2) {"words": [...]} 형태
    if isinstance(data, dict):
        if "words" in data and isinstance(data["words"], list):
            return [str(x).strip() for x in data["words"] if str(x).strip()]

    raise ValueError(
        f"지원하지 않는 JSON 구조입니다. "
        f"리스트 / words 배열 / CSV 비슷한 행 리스트를 사용하세요: {path}"
    )


# ---------------------------------------------------------
# JSONL 파일 처리
# ---------------------------------------------------------

def _load_jsonl(path: Path) -> List[str]:
    """
    JSONL 파일 로드.
    - 문자열 라인
    - {"text": ...}, {"word": ...}, {"content": ...}
    """
    words = []
    preferred_cols = ["text", "word", "content"]

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 문자열 라인
            if not (line.startswith("{") and line.endswith("}")):
                words.append(line)
                continue

            obj = json.loads(line)

            if isinstance(obj, dict):
                for col in preferred_cols:
                    if col in obj and str(obj[col]).strip():
                        words.append(str(obj[col]).strip())
                        break

    return words


# ---------------------------------------------------------
# 메인 엔트리 포인트
# ---------------------------------------------------------

def load_words(path_str: str) -> List[str]:
    """
    텍스트/단어 리스트 로드.
    지원 형식:
      - .txt   → 줄 단위
      - .csv   → text/word/content 또는 첫 컬럼
      - .json  → 리스트 / {"words": [...]} / CSV와 비슷한 행 리스트
      - .jsonl → text/word/content 또는 문자열 라인
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")

    suf = path.suffix.lower()

    if suf == ".txt":
        return _load_txt(path)

    if suf == ".csv":
        return _load_csv(path)

    if suf == ".json":
        return _load_json(path)

    if suf == ".jsonl":
        return _load_jsonl(path)

    raise ValueError(
        f"지원하지 않는 입력 형식입니다: {path}. "
        "지원 형식: .txt / .csv / .json / .jsonl"
    )


# ---------------------------------------------------------
# 저장 유틸
# ---------------------------------------------------------

def save_embeddings(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_clusters(df, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def save_extracted_axes(data: List[Dict[str, Any]], path: str) -> None:
    """
    EA 축 정보를 CSV로 저장.
    word, axis_1_index, axis_1_value, ... 구조로 flatten.
    """
    import pandas as pd

    rows = []
    for item in data:
        word = item["word"]
        axes = item["axes"]
        base = {"word": word}

        for i, ax in enumerate(axes):
            base[f"axis_{i+1}_index"] = ax["index"]
            base[f"axis_{i+1}_value"] = ax["value"]

        rows.append(base)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")
