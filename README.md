# EL-Core Project

본 프로젝트는 감정 모델링, 강화학습, LLM 임베딩 기반 의미 분석 등의 실험 환경을 제공합니다.  
이 문서는 프로젝트 설치 방법과 LLM 임베딩 분석 패키지(`llm_emb`) 사용 방법을 포함합니다.

---

# 1. 프로젝트 환경 설정

## 1.1 가상환경 생성 및 패키지 설치

```bash
python -m venv .venv
````

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### 패키지 설치

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Torch CUDA 버전 설치 안내

GPU 환경에서는 로컬 CUDA 버전에 맞춰 torch를 설치해야 한다.

CUDA 12.x 예시:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.x 예시:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CPU 전용 환경에서는 requirements의 torch CPU 버전을 그대로 사용해도 된다.

---

## 1.2 PYTHONPATH 설정

프로젝트 구조상 `src` 디렉토리를 패키지 루트로 인식해야 한다.
Python 코드 실행 전 아래 명령을 먼저 적용해야 한다.

### Windows PowerShell

```powershell
$env:PYTHONPATH = "$(Get-Location)\src"
```

### macOS / Linux

```bash
export PYTHONPATH="$PWD/src"
```

---

# 2. llm_emb – LLM 기반 텍스트 임베딩 분석 도구

`llm_emb`는 GGUF 기반 LLM을 사용하여 텍스트 임베딩을 생성하고, PCA 분석 및 의미 시각화를 수행하는 도구이다.

지원 기능:

* 텍스트 임베딩 생성
* PCA 기반 2D/3D 맵핑
* K-Means 클러스터링
* EA(Extracted Axes) 분석 기능(1단계)

---

# 3. LLM 모델 준비

HuggingFace Hub CLI로 GGUF 모델을 다운로드한다.

```bash
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --local-dir models \
  --local-dir-use-symlinks False
```

모델 저장 구조 예:

```
models/
  Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

---

# 4. 입력 파일 형식

`llm_emb`는 다음 입력 형식을 지원한다.

## 4.1 TXT

줄 단위 텍스트를 그대로 사용한다.

```
happiness
sadness
anger
```

## 4.2 CSV

`text`, `word`, `content` 중 하나의 컬럼을 사용한다.

```csv
text,category
happiness,emotion
sadness,emotion
```

## 4.3 JSON

다음 형식을 모두 지원한다.

리스트:

```json
["happiness","sadness","anger"]
```

words 배열:

```json
{"words": ["happiness", "sadness", "anger"]}
```

CSV와 유사한 dict 리스트:

```json
[
  {"text": "happiness"},
  {"text": "sadness"}
]
```

## 4.4 JSONL

```jsonl
{"text": "happiness"}
{"text": "sadness"}
```

---

# 5. PCA 분석 기능

PCA 분석은 임베딩된 텍스트를 저차원(2D 또는 3D) 공간으로 투영하여
텍스트 의미 구조를 시각적으로 분석하는 기능이다.

## 5.1 실행

```bash
python -m src.llm_emb.emb_cli pca --input data/words.csv
```

## 5.2 출력

```
emb_output/words/
  embeddings.json
  pca_analysis.csv
  pca_plot.png
```

* `embeddings.json`: 텍스트 임베딩 원본
* `pca_analysis.csv`: PCA 좌표 및 클러스터 결과
* `pca_plot.png`: 2D/3D 분석 이미지

---

# 6. EA 분석 기능 (준비 중)

EA(Extracted Axes)는 임베딩 벡터에서 특정 단어를 규정하는 주요 축(top-k)을 추출하는 분석 방식이다.
현재는 1단계 기능(상위 축 추출 및 CSV 저장)까지만 제공된다.

## 6.1 실행

```bash
python -m src.llm_emb.emb_cli ea --input data/words.csv
```

## 6.2 출력

```
emb_output/words/
  embeddings.json
  axes.csv
```

---

# 7. EmbConfig 기본값 및 옵션 구조

`EmbConfig`는 PCA 및 EA 실행 시 기본값을 정의하는 설정 객체이다.
위치는 다음과 같다:

```
src/llm_emb/emb_config.py
```

예시:

```python
PCA_DEFAULT_CONFIG = EmbConfig(
    model="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    clusters=5,
    extract_axes=0,
    use_3d=True,
)
```

옵션 적용 우선순위:

1. CLI 옵션
2. EmbConfig 기본값
3. 지정하지 않은 항목은 기본값 유지

---

# 8. 옵션 상세 설명

## 8.1 공통 옵션

### `--input, -i`

분석에 사용할 텍스트 파일 경로.
TXT, CSV, JSON, JSONL 모두 지원한다.

### `--output, -o`

결과물을 저장할 디렉토리.
지정하지 않으면 `emb_output/{입력파일명}/`이 자동 생성된다.

### `--model, -m`

임베딩에 사용할 GGUF 모델 경로.
지정하지 않으면 `EmbConfig`의 모델 기본값이 사용된다.

---

## 8.2 PCA 옵션

### `--clusters, -k`

K-Means 클러스터 개수.
0 또는 미지정 시 클러스터링을 수행하지 않는다.
기본값은 `PCA_DEFAULT_CONFIG.clusters`.

### `--3d`

3D PCA 플롯 생성 여부.
지정하면 3D 플롯을 사용하고, 미지정 시 `EmbConfig.use_3d`가 적용된다.

---

## 8.3 EA 옵션

### `--extract-axes`

EA 분석 시 상위 축 개수(top-k).
기본값은 `EA_DEFAULT_CONFIG.extract_axes`.

---

# 9. 테스트 실행 흐름

아래 순서로 설치 및 기능을 빠르게 검증할 수 있다.

```bash
# 모델 다운로드
huggingface-cli download ...

# PCA 실행
python -m src.llm_emb.emb_cli pca --input data/words.csv

# EA 실행
python -m src.llm_emb.emb_cli ea --input data/words.csv
```

---