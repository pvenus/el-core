# src/llm_emb/emb_pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import json

import pandas as pd

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
    build_pca_report,
    report_to_components_df,
)


def run_pca_pipeline(
    input_path: str,
    output_dir: str,
    model_path: str,
    clusters: int = 0,
    use_3d: bool = False,
    show_labels: bool = True,
    pca_top_dims: int = 8,
) -> None:
    """
    입력 파일 -> 임베딩 -> PCA -> (옵션)KMeans -> plot/CSV 저장
    + PC 축별 dominant_dim(지배 축) 리포트 저장

    생성 파일:
      - embeddings.json
      - pca_analysis.csv
      - pca_plot.png
      - pca_report.json       (PC별 dominant_dim, explained variance)
      - pca_components.csv    (PC별 top loading dims 표)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) 모델 로드
    print(f"[PIPELINE-PCA] 모델 로드: {model_path}")
    llm = LLMEmbeddingModel(model_path)

    # 2) 입력 로드
    print(f"[PIPELINE-PCA] 입력 로드: {input_path}")
    texts = load_words(input_path)
    if not texts:
        raise ValueError(f"[PIPELINE-PCA] 입력 텍스트가 비었습니다: {input_path}")

    # 3) 임베딩
    print(f"[PIPELINE-PCA] 임베딩 생성: {len(texts)} items")
    embeddings = llm.embed_texts(texts)

    emb_path = out / "embeddings.json"
    print(f"[PIPELINE-PCA] 임베딩 저장: {emb_path}")
    save_embeddings(embeddings, str(emb_path))

    # 4) PCA
    n_components = 3 if use_3d else 2
    print(f"[PIPELINE-PCA] PCA 수행: n_components={n_components}")
    df, pca_model = run_pca(embeddings, n_components=n_components, return_model=True)

    # 5) (옵션) KMeans
    if clusters and clusters > 0:
        print(f"[PIPELINE-PCA] KMeans 수행: k={clusters}")
        df = run_kmeans(df, clusters)

    # 6) CSV 저장
    csv_path = out / "pca_analysis.csv"
    print(f"[PIPELINE-PCA] PCA CSV 저장: {csv_path}")
    save_clusters(df, str(csv_path))

    # 7) Plot 저장
    plot_path = out / "pca_plot.png"
    print(f"[PIPELINE-PCA] PCA Plot 저장: {plot_path}")
    plot_pca(df, str(plot_path), is_3d=use_3d, show_labels=show_labels)

    # 8) PCA 리포트 저장 (dominant_dim 포함)
    # 임베딩 차원은 pooled 첫 벡터에서 추정
    first = next(iter(embeddings.values()))
    emb_dim = len(first["pooled"])

    report = build_pca_report(
        pca=pca_model,
        embedding_dim=emb_dim,
        n_samples=len(df),
        top_k_dims=pca_top_dims,
    )

    report_json = out / "pca_report.json"
    print(f"[PIPELINE-PCA] PCA 리포트 저장: {report_json}")
    report_json.write_text(
        json.dumps(report, default=lambda o: o.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    comp_df = report_to_components_df(report)
    comp_csv = out / "pca_components.csv"
    print(f"[PIPELINE-PCA] PCA 컴포넌트(로딩) 표 저장: {comp_csv}")
    comp_df.to_csv(comp_csv, index=False, encoding="utf-8-sig")

    # 콘솔 요약 로그
    for ax in report.axes:
        print(
            f"[PIPELINE-PCA] {ax.pc.upper()} dominant_dim={ax.dominant_dim} "
            f"(loading={ax.dominant_loading:+.4f})"
        )

    print("[PIPELINE-PCA] 완료.")


def run_ea_pipeline(
    input_path: str,
    output_dir: str,
    model_path: str,
    extract_axes_count: int = 10,
) -> None:
    """
    EA(준비중) 파이프라인: 임베딩 + top-k 축 추출 CSV 저장

    생성 파일:
      - embeddings.json
      - axes.csv (extract_axes_count > 0인 경우)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[PIPELINE-EA] 모델 로드: {model_path}")
    llm = LLMEmbeddingModel(model_path)

    print(f"[PIPELINE-EA] 입력 로드: {input_path}")
    texts = load_words(input_path)
    if not texts:
        raise ValueError(f"[PIPELINE-EA] 입력 텍스트가 비었습니다: {input_path}")

    print(f"[PIPELINE-EA] 임베딩 생성: {len(texts)} items")
    embeddings = llm.embed_texts(texts)

    emb_path = out / "embeddings.json"
    print(f"[PIPELINE-EA] 임베딩 저장: {emb_path}")
    save_embeddings(embeddings, str(emb_path))

    if extract_axes_count and extract_axes_count > 0:
        print(f"[PIPELINE-EA] EA 중요 축 추출(top_k={extract_axes_count}) 수행 중...")
        axes_data = extract_axes(embeddings, top_k=extract_axes_count)

        axes_path = out / "axes.csv"
        print(f"[PIPELINE-EA] EA 축 정보 저장: {axes_path}")
        save_extracted_axes(axes_data, str(axes_path))
    else:
        print("[PIPELINE-EA] extract_axes_count = 0 → EA 축 추출 생략")

    print("[PIPELINE-EA] EA 파이프라인 완료.")


# ---------------------------------------------------------
# PCA 결과 비교 (중요: PCA 실행과 "분리된" 단계)
# ---------------------------------------------------------


def _normalize_run_inputs_to_reports(
    runs: Optional[List[str]],
    reports: Optional[List[str]],
) -> List[str]:
    """
    --runs 또는 --reports 입력을 받아 최종 report_paths 리스트로 통일한다.
    - runs: emb_output/set_A 처럼 '결과 폴더'들을 받음 (각 폴더에 pca_report.json 필요)
    - reports: emb_output/set_A/pca_report.json 처럼 report 파일들을 받음
    """
    report_paths: List[str] = []

    if runs:
        for r in runs:
            rp = Path(r) / "pca_report.json"
            if not rp.exists():
                raise FileNotFoundError(f"pca_report.json not found in run dir: {r}")
            report_paths.append(str(rp))

    if reports:
        for rp in reports:
            p = Path(rp)
            if not p.exists():
                raise FileNotFoundError(f"report file not found: {rp}")
            report_paths.append(str(p))

    # 중복 제거(순서 유지)
    seen = set()
    uniq: List[str] = []
    for x in report_paths:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)

    return uniq


def compare_pca_results(
    output_dir: str,
    runs: Optional[List[str]] = None,
    reports: Optional[List[str]] = None,
) -> None:
    """
    이미 생성된 pca_report.json 들을 비교해서
    dominant_dim 변화와 EVR(explained variance ratio) 변화를 요약 CSV로 저장한다.

    입력:
      - runs: 결과 폴더 리스트(각 폴더에 pca_report.json 존재)
      - reports: pca_report.json 파일 리스트
      둘 중 하나(또는 둘 다) 제공 가능. 최종적으로 report_paths로 통합 후 비교.

    생성 파일:
      - compare_summary.csv   (각 run별 요약)
      - compare_diffs.csv     (연속 run 간 dominant_dim 변화)
    """
    report_paths = _normalize_run_inputs_to_reports(runs, reports)
    if len(report_paths) < 2:
        raise ValueError("비교를 위해 report가 2개 이상 필요합니다. --runs 또는 --reports를 2개 이상 주세요.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for rp in report_paths:
        p = Path(rp)
        run_name = p.parent.name if p.name == "pca_report.json" else p.stem

        rep_obj = json.loads(p.read_text(encoding="utf-8"))
        axes = {a["pc"]: a for a in rep_obj.get("axes", [])}
        evr = rep_obj.get("explained_variance_ratio", [])

        row = {
            "run": run_name,
            "report_path": str(p),
            "n_samples": rep_obj.get("n_samples"),
            "embedding_dim": rep_obj.get("embedding_dim"),
            "n_components": rep_obj.get("n_components"),
            "evr_pc1": evr[0] if len(evr) > 0 else None,
            "evr_pc2": evr[1] if len(evr) > 1 else None,
            "evr_pc3": evr[2] if len(evr) > 2 else None,
            "pc1_dominant_dim": axes.get("pc1", {}).get("dominant_dim"),
            "pc2_dominant_dim": axes.get("pc2", {}).get("dominant_dim"),
            "pc3_dominant_dim": axes.get("pc3", {}).get("dominant_dim"),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_csv = out / "compare_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[COMPARE] 요약 저장: {summary_csv}")

    diffs = []
    for i in range(1, len(rows)):
        a = rows[i - 1]
        b = rows[i]
        diffs.append(
            {
                "from": a["run"],
                "to": b["run"],
                "pc1": f'{a["pc1_dominant_dim"]} -> {b["pc1_dominant_dim"]}',
                "pc2": f'{a["pc2_dominant_dim"]} -> {b["pc2_dominant_dim"]}',
                "pc3": f'{a["pc3_dominant_dim"]} -> {b["pc3_dominant_dim"]}',
            }
        )

    diffs_df = pd.DataFrame(diffs)
    diffs_csv = out / "compare_diffs.csv"
    diffs_df.to_csv(diffs_csv, index=False, encoding="utf-8-sig")
    print(f"[COMPARE] 변화(diff) 저장: {diffs_csv}")

    print("[COMPARE] 완료.")
