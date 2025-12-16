# src/llm_emb/emb_cli.py

import argparse
import json
from pathlib import Path

from .emb_model import LLMEmbeddingModel
from .emb_io import (
    load_words,
    save_embeddings,
    save_clusters,
    save_extracted_axes,
)
from .emb_analysis import (
    run_pca,
    run_kmeans,
    plot_pca,
    extract_axes,
)
from .emb_pipeline import run_pca_pipeline
from .emb_config import (
    PCA_DEFAULT_CONFIG,
    EA_DEFAULT_CONFIG,
    make_config,
)


# ---------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------


def make_output_dir_for_input(input_path: str, base_dir: str) -> Path:
    """
    입력 파일명을 따서 결과 디렉토리를 생성한다.
    예:
      input : data/emotion_words.csv
      base  : ./emb_output
      결과  : ./emb_output/emotion_words
    """
    in_path = Path(input_path)
    base = Path(base_dir)
    return base / in_path.stem


# ---------------------------------------------------------
# 1) embed 서브커맨드 (개발자용)
# ---------------------------------------------------------


def cmd_embed(args: argparse.Namespace) -> None:
    from .emb_config import DEFAULT_MODEL_PATH  # 필요시

    model_path = args.model or DEFAULT_MODEL_PATH

    print(f"[EMBED] 모델 로드: {model_path}")
    llm = LLMEmbeddingModel(model_path)

    print(f"[EMBED] 입력 로드: {args.input}")
    texts = load_words(args.input)

    embeddings = llm.embed_texts(texts)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "embeddings.json"
    print(f"[EMBED] 임베딩 저장: {out_path}")
    save_embeddings(embeddings, str(out_path))

    print("[EMBED] 완료.")


# ---------------------------------------------------------
# 2) analyze 서브커맨드 (개발자용)
# ---------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> None:
    print(f"[ANALYZE] 임베딩 로드: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # embeddings.json 또는 {"embeddings": {...}} 둘 다 대응
    embeddings = data.get("embeddings", data)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_components = 3 if args.is_3d else 2

    print(f"[ANALYZE] PCA 수행 (n_components={n_components})...")
    df = run_pca(embeddings, n_components=n_components)

    if args.clusters > 0:
        print(f"[ANALYZE] K-Means 수행 (k={args.clusters})...")
        df = run_kmeans(df, args.clusters)

    csv_path = out_dir / "pca_analysis.csv"
    print(f"[ANALYZE] 분석 결과 CSV 저장: {csv_path}")
    save_clusters(df, str(csv_path))

    plot_path = out_dir / "pca_plot.png"
    print(f"[ANALYZE] PCA 플롯 저장: {plot_path}")
    plot_pca(df, str(plot_path))

    print("[ANALYZE] 완료.")


# ---------------------------------------------------------
# 3) pca 서브커맨드 (기획자용: PCA 전용 원샷)
# ---------------------------------------------------------


def cmd_pca(args: argparse.Namespace) -> None:
    """
    기획자를 위한 '한 줄짜리 PCA 분석' 커맨드.

    예:
      python -m src.llm_emb.emb_cli pca --input data/emotion_words.csv

    - 모델 로드 (기본값: emb_config.PCA_DEFAULT_CONFIG.model)
    - 입력 파일 임베딩
    - PCA + (옵션) K-Means
    - 결과: ./emb_output/{입력파일명}/ 아래에 embeddings.json / pca_analysis.csv / pca_plot.png 생성
    """
    cfg = make_config(PCA_DEFAULT_CONFIG, args)

    if cfg.input is None:
        raise ValueError("PCA 실행을 위해서는 --input이 필요합니다.")

    # 출력 디렉토리 결정: 지정 없으면 emb_output/{입력파일명}
    if cfg.output:
        out_dir = Path(cfg.output)
    else:
        out_dir = make_output_dir_for_input(cfg.input, cfg.output_base)

    print(f"[PCA] 입력 파일       : {cfg.input}")
    print(f"[PCA] 모델 경로       : {cfg.model}")
    print(f"[PCA] 출력 디렉토리   : {out_dir}")
    print(f"[PCA] 클러스터 개수   : {cfg.clusters}")
    print(f"[PCA] 3D 사용 여부    : {cfg.use_3d}")

    run_pca_pipeline(
        input_path=cfg.input,
        output_dir=str(out_dir),
        model_path=cfg.model,
        clusters=cfg.clusters,
        use_3d=cfg.use_3d,
    )

    print("[PCA] 전체 PCA 파이프라인 완료.")


# ---------------------------------------------------------
# 4) ea 서브커맨드 (EA 전용: 현재는 축 추출까지만, 자리 유지용)
# ---------------------------------------------------------


def cmd_ea(args: argparse.Namespace) -> None:
    """
    EA(Extracted Axes) 분석용 커맨드 자리.

    현재 버전:
      - 입력 텍스트 임베딩
      - EA 축 추출(top_k = extract_axes)
      - axes.csv 저장

    추후:
      - EA 기반 클러스터링 / 플롯 추가 예정
    """
    cfg = make_config(EA_DEFAULT_CONFIG, args)

    if cfg.input is None:
        raise ValueError("EA 실행을 위해서는 --input이 필요합니다.")

    # 출력 디렉토리 결정
    if cfg.output:
        out_dir = Path(cfg.output)
    else:
        out_dir = make_output_dir_for_input(cfg.input, cfg.output_base)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EA] 입력 파일       : {cfg.input}")
    print(f"[EA] 모델 경로       : {cfg.model}")
    print(f"[EA] 출력 디렉토리   : {out_dir}")
    print(f"[EA] 추출 축 개수    : {cfg.extract_axes}")

    # 1) 모델 & 텍스트 로드
    llm = LLMEmbeddingModel(cfg.model)
    texts = load_words(cfg.input)

    # 2) 임베딩
    embeddings = llm.embed_texts(texts)
    emb_path = out_dir / "embeddings.json"
    print(f"[EA] 임베딩 저장: {emb_path}")
    save_embeddings(embeddings, str(emb_path))

    # 3) EA 추출
    if cfg.extract_axes > 0:
        print(f"[EA] EA 축 추출 (top_k={cfg.extract_axes})...")
        axes_data = extract_axes(embeddings, top_k=cfg.extract_axes)
        axes_path = out_dir / "axes.csv"
        print(f"[EA] EA 축 정보 저장: {axes_path}")
        save_extracted_axes(axes_data, str(axes_path))
    else:
        print("[EA] extract_axes 값이 0이어서 EA 축을 추출하지 않았습니다.")

    print("[EA] EA 파이프라인(임시 버전) 완료.")


# ---------------------------------------------------------
# 5) main( ) - CLI 엔트리포인트
# ---------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Embedding Tool (llm_emb)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---------- embed ----------
    p = sub.add_parser("embed", help="단어/문장 리스트로부터 임베딩 생성")
    p.add_argument("--input", "-i", required=True, help="입력 파일 경로 (txt/csv/json/jsonl)")
    p.add_argument("--output", "-o", required=True, help="embeddings.json을 저장할 디렉토리")
    p.add_argument("--model", "-m", required=False, default=None, help="GGUF 모델 경로")
    p.set_defaults(func=cmd_embed)

    # ---------- analyze ----------
    p = sub.add_parser("analyze", help="embeddings.json으로부터 PCA 분석 수행")
    p.add_argument("--input", "-i", required=True, help="embeddings.json 파일 경로")
    p.add_argument("--output", "-o", required=True, help="분석 결과를 저장할 디렉토리")
    p.add_argument("--clusters", "-k", type=int, default=0, help="K-Means 클러스터 개수 (0이면 미사용)")
    p.add_argument("--3d", dest="is_3d", action="store_true", help="3D PCA 사용")
    p.set_defaults(func=cmd_analyze)

    # ---------- pca (기획자용) ----------
    # ---------- pca (기획자용) ----------
    p = sub.add_parser("pca", help="입력 파일 한 번에 PCA 임베딩/분석 (기획자용)")
    p.add_argument("--input", "-i", required=True, help="입력 파일 경로 (txt/csv/json/jsonl)")
    p.add_argument("--output", "-o", required=False, default=None, help="결과 디렉토리 (생략 시 자동 결정)")
    p.add_argument("--model", "-m", required=False, default=None,
                   help="GGUF 모델 경로 (지정 안 하면 PCA_DEFAULT_CONFIG.model 사용)")
    p.add_argument("--clusters", "-k", type=int, default=None,
                   help="K-Means 클러스터 개수 (지정 안 하면 PCA_DEFAULT_CONFIG.clusters 사용)")
    p.add_argument("--3d", dest="is_3d", action="store_const", const=True, default=None,
                   help="3D PCA 사용 (지정 안 하면 PCA_DEFAULT_CONFIG.use_3d 사용)")
    p.set_defaults(func=cmd_pca)

    # ---------- ea (EA 전용 자리) ----------
    p = sub.add_parser("ea", help="입력 파일에 대해 EA(중요 축) 분석 수행 (임시 버전)")
    p.add_argument("--input", "-i", required=True, help="입력 파일 경로 (txt/csv/json/jsonl)")
    p.add_argument("--output", "-o", required=False, default=None, help="결과 디렉토리 (생략 시 자동 결정)")
    p.add_argument("--model", "-m", required=False, default=None, help="GGUF 모델 경로")
    p.add_argument("--extract-axes", type=int, default=None, help="추출할 중요 축 개수 (None이면 emb_config 기본값)")
    p.set_defaults(func=cmd_ea)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
