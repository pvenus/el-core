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
from .emb_pipeline import (
    run_pca_pipeline,
    run_ea_pipeline,
    compare_pca_results,
)
from .emb_config import (
    PCA_DEFAULT_CONFIG,
    EA_DEFAULT_CONFIG,
    make_config,
)


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
    from .emb_config import DEFAULT_MODEL_PATH

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
# 2) analyze (개발자용)
# ---------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> None:
    print(f"[ANALYZE] 임베딩 로드: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = data.get("embeddings", data)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # analyze는 config 안 거치므로 여기서는 명시적으로만 결정
    n_components = 3 if args.use_3d else 2

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
    plot_pca(df, str(plot_path), is_3d=args.use_3d, show_labels=True)

    print("[ANALYZE] 완료.")


# ---------------------------------------------------------
# 3) pca (파이프라인)
# ---------------------------------------------------------


def cmd_pca(args: argparse.Namespace) -> None:
    cfg = make_config(PCA_DEFAULT_CONFIG, args)

    if cfg.input is None:
        raise ValueError("PCA 실행을 위해서는 --input이 필요합니다.")

    # 출력 디렉토리 결정: 지정 없으면 emb_output/{입력파일명}
    if cfg.output:
        out_dir = Path(cfg.output)
    else:
        out_dir = make_output_dir_for_input(cfg.input, cfg.output_base)

    print(f"[PCA] 입력 파일     : {cfg.input}")
    print(f"[PCA] 모델 경로     : {cfg.model}")
    print(f"[PCA] 출력 디렉토리 : {out_dir}")
    print(f"[PCA] 클러스터 개수 : {cfg.clusters}")
    print(f"[PCA] 3D 사용 여부  : {cfg.use_3d}")

    run_pca_pipeline(
        input_path=cfg.input,
        output_dir=str(out_dir),
        model_path=cfg.model,
        clusters=cfg.clusters,
        use_3d=cfg.use_3d,
        show_labels=True,
        pca_top_dims=8,
    )

    print("[PCA] 완료.")


# ---------------------------------------------------------
# 4) ea (자리 유지)
# ---------------------------------------------------------


def cmd_ea(args: argparse.Namespace) -> None:
    cfg = make_config(EA_DEFAULT_CONFIG, args)

    if cfg.input is None:
        raise ValueError("EA 실행을 위해서는 --input이 필요합니다.")

    if cfg.output:
        out_dir = Path(cfg.output)
    else:
        out_dir = make_output_dir_for_input(cfg.input, cfg.output_base)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EA] 입력 파일     : {cfg.input}")
    print(f"[EA] 모델 경로     : {cfg.model}")
    print(f"[EA] 출력 디렉토리 : {out_dir}")
    print(f"[EA] 추출 축 개수  : {cfg.extract_axes}")

    run_ea_pipeline(
        input_path=cfg.input,
        output_dir=str(out_dir),
        model_path=cfg.model,
        extract_axes_count=cfg.extract_axes,
    )

    print("[EA] 완료.")


# ---------------------------------------------------------
# 5) pca-compare-results (PCA 결과 비교 전용)
# ---------------------------------------------------------


def cmd_pca_compare_results(args: argparse.Namespace) -> None:
    """
    이미 생성된 PCA 결과 폴더(--runs) 또는 report 파일(--reports)을 입력으로 받아
    비교 요약 파일(compare_summary.csv, compare_diffs.csv)을 생성한다.

    예:
      python -m src.llm_emb.emb_cli pca-compare-results --runs emb_output\\set_A emb_output\\set_B
      python -m src.llm_emb.emb_cli pca-compare-results --reports emb_output\\set_A\\pca_report.json emb_output\\set_B\\pca_report.json
    """
    runs = args.runs if args.runs else None
    reports = args.reports if args.reports else None

    # 최소 2개 이상 비교해야 의미 있음
    total = (len(runs) if runs else 0) + (len(reports) if reports else 0)
    if total < 2:
        raise ValueError("pca-compare-results는 --runs 또는 --reports로 2개 이상 입력이 필요합니다.")

    out_dir = Path(args.output) if args.output else Path("emb_output") / "compare_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PCA-COMPARE] runs    : {runs}")
    print(f"[PCA-COMPARE] reports : {reports}")
    print(f"[PCA-COMPARE] output  : {out_dir}")

    compare_pca_results(
        output_dir=str(out_dir),
        runs=runs,
        reports=reports,
    )

    print("[PCA-COMPARE] 완료.")


# ---------------------------------------------------------
# 6) main
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
    # analyze는 config를 타지 않는 개발자용이라 default False여도 OK
    p.add_argument("--3d", dest="use_3d", action="store_true", help="3D PCA 사용")
    p.set_defaults(func=cmd_analyze)

    # ---------- pca ----------
    p = sub.add_parser("pca", help="입력 파일에 대해 PCA 임베딩/분석 수행")
    p.add_argument("--input", "-i", required=True, help="입력 파일 경로 (txt/csv/json/jsonl)")
    p.add_argument("--output", "-o", required=False, default=None, help="결과 디렉토리 (생략 시 자동 결정)")
    p.add_argument("--model", "-m", required=False, default=None, help="GGUF 모델 경로 (emb_config 기본값 사용 가능)")
    p.add_argument("--clusters", "-k", type=int, default=None, help="K-Means 클러스터 개수 (None이면 기본값)")
    # ★핵심: 옵션 미지정 시 None → config 유지
    p.add_argument("--3d", dest="use_3d", action="store_const", const=True, default=None, help="3D PCA 강제 사용(지정 시 True)")
    p.add_argument("--2d", dest="use_3d", action="store_const", const=False, default=None, help="2D PCA 강제 사용(지정 시 False)")
    p.set_defaults(func=cmd_pca)

    # ---------- ea ----------
    p = sub.add_parser("ea", help="입력 파일에 대해 EA(중요 축) 분석 수행 (임시 버전)")
    p.add_argument("--input", "-i", required=True, help="입력 파일 경로 (txt/csv/json/jsonl)")
    p.add_argument("--output", "-o", required=False, default=None, help="결과 디렉토리 (생략 시 자동 결정)")
    p.add_argument("--model", "-m", required=False, default=None, help="GGUF 모델 경로")
    p.add_argument("--extract-axes", type=int, default=None, help="추출할 중요 축 개수 (None이면 기본값)")
    p.set_defaults(func=cmd_ea)

    # ---------- pca-compare-results ----------
    p = sub.add_parser("pca-compare-results", help="이미 생성된 PCA 결과(run/report)를 비교하여 요약 CSV 생성")
    p.add_argument("--runs", nargs="*", default=None, help="비교할 PCA 결과 폴더들 (각 폴더에 pca_report.json 필요)")
    p.add_argument("--reports", nargs="*", default=None, help="비교할 pca_report.json 파일들")
    p.add_argument("--output", "-o", required=False, default=None, help="비교 결과 저장 디렉토리 (생략 시 emb_output/compare_results)")
    p.set_defaults(func=cmd_pca_compare_results)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
