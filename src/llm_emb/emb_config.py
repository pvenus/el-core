# src/llm_emb/emb_config.py

from dataclasses import dataclass, asdict
from typing import Any


# 공통 기본값 ---------------------------------------------------------

# 기본 GGUF 모델 경로
DEFAULT_MODEL_PATH = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# 결과 폴더 베이스
DEFAULT_OUTPUT_BASE = "./emb_output"


@dataclass
class EmbConfig:
    """
    임베딩 / 분석 파이프라인에 사용되는 기본 설정값.

    - model:   사용할 GGUF 모델 경로
    - input:   입력 파일 경로 (txt/csv/json/jsonl)
    - output:  결과 저장 디렉토리 (지정 안 하면 output_base + 입력파일명)
    - output_base: 결과 디렉토리 자동 생성 기본 베이스 경로
    - clusters: K-Means 클러스터 개수 (0이면 클러스터링 생략)
    - extract_axes: EA(중요 축) 추출 개수 (0이면 EA 생략)
    - use_3d:  PCA 플롯에서 PC3까지 사용할지 여부
    """
    model: str = DEFAULT_MODEL_PATH
    input: str | None = None
    output: str | None = None
    output_base: str = DEFAULT_OUTPUT_BASE
    clusters: int = 0
    extract_axes: int = 0
    use_3d: bool = False


# 모드별 기본 설정 ----------------------------------------------------


#: PCA 전용 실행에 사용할 기본 설정
PCA_DEFAULT_CONFIG = EmbConfig(
    model=DEFAULT_MODEL_PATH,
    clusters=5,      # 기본 클러스터 개수
    extract_axes=0,  # PCA 모드에서는 EA 사용 안 함
    use_3d=True,     # 기본은 3D
)

#: EA 분석 전용 실행에 사용할 기본 설정 (임시: 축만 추출)
EA_DEFAULT_CONFIG = EmbConfig(
    model=DEFAULT_MODEL_PATH,
    clusters=0,      # EA 모드에서는 K-Means 미사용
    extract_axes=10, # 기본 EA top_k
    use_3d=False,
)


# CLI 인자와 합치는 헬퍼 ----------------------------------------------


def make_config(default_cfg: EmbConfig, args: Any) -> EmbConfig:
    """
    주어진 기본 설정(default_cfg)에 CLI 인자를 덮어써서 최종 EmbConfig를 만든다.

    - args.model / args.input / args.output / args.clusters / args.extract_axes / args.is_3d
      등을 지원한다.
    - args 쪽 값이 None일 때는 덮어쓰지 않고 default_cfg 값을 그대로 사용한다.
    """
    cfg_dict = asdict(default_cfg)
    cfg = EmbConfig(**cfg_dict)

    # 모델 경로
    if hasattr(args, "model") and args.model is not None:
        cfg.model = args.model

    # 입력 / 출력
    if hasattr(args, "input") and args.input is not None:
        cfg.input = args.input

    if hasattr(args, "output") and args.output is not None:
        cfg.output = args.output

    # 클러스터 개수
    if hasattr(args, "clusters") and args.clusters is not None:
        cfg.clusters = args.clusters

    # EA 축 개수
    if hasattr(args, "extract_axes") and args.extract_axes is not None:
        cfg.extract_axes = args.extract_axes

    # 3D 사용 여부
    if hasattr(args, "is_3d") and args.is_3d is not None:
        # is_3d가 None이 아닐 때에만 덮어쓴다 (True/False 둘 다 허용)
        cfg.use_3d = bool(args.is_3d)

    return cfg
