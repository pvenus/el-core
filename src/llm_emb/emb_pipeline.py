# src/llm_emb/emb_pipeline.py

from pathlib import Path
from typing import Dict, Any

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


# ---------------------------------------------------------
# 1) PCA 파이프라인
# ---------------------------------------------------------


def run_pca_pipeline(
    input_path: str,
    output_dir: str,
    model_path: str,
    clusters: int = 0,
    use_3d: bool = False,
) -> None:
    """
    텍스트 → 임베딩 → PCA → (옵션) K-Means → CSV + 플롯까지 한 번에 수행.

    결과물:
      - embeddings.json   : 텍스트별 임베딩(pooled)
      - pca_analysis.csv  : word / pc1, pc2, (pc3) / cluster(옵션)
      - pca_plot.png      : PCA 플롯 (2D 또는 3D, 클러스터 색상 + 단어 라벨)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) 모델 로드
    print(f"[PIPELINE-PCA] 모델 로드 중: {model_path}...")
    model = LLMEmbeddingModel(model_path)

    # 2) 입력 텍스트 로드
    print(f"[PIPELINE-PCA] 입력 텍스트 로드: {input_path}...")
    texts = load_words(input_path)
    print(f"[PIPELINE-PCA] 총 {len(texts)}개 텍스트 임베딩 예정.")

    # 3) 임베딩 생성
    print("[PIPELINE-PCA] 임베딩 생성 중...")
    embeddings: Dict[str, Dict[str, Any]] = model.embed_texts(texts)

    emb_path = out / "embeddings.json"
    print(f"[PIPELINE-PCA] 임베딩 결과 저장: {emb_path}")
    save_embeddings(embeddings, str(emb_path))

    # 4) PCA
    n_components = 3 if use_3d else 2
    print(f"[PIPELINE-PCA] PCA 수행 (n_components={n_components})...")
    df_pca = run_pca(embeddings, n_components=n_components)  # word / pc*

    # 5) K-Means (옵션)
    if clusters > 0:
        print(f"[PIPELINE-PCA] K-Means 수행 (k={clusters})...")
        df_pca = run_kmeans(df_pca, clusters)  # 'cluster' 컬럼 추가

    # 6) 결과 CSV 저장
    csv_path = out / "pca_analysis.csv"
    print(f"[PIPELINE-PCA] PCA 분석 결과 CSV 저장: {csv_path}")
    save_clusters(df_pca, str(csv_path))

    # 7) 플롯 저장
    plot_path = out / "pca_plot.png"
    print(f"[PIPELINE-PCA] PCA 플롯 저장: {plot_path}")
    plot_pca(df_pca, str(plot_path))

    print("[PIPELINE-PCA] PCA 파이프라인 완료.")


# 옛 코드 호환용 alias (예전 run_pipeline 호출 대비)
run_pipeline = run_pca_pipeline


# ---------------------------------------------------------
# 2) EA 파이프라인 (임시: EA 축 추출까지만)
# ---------------------------------------------------------


def run_ea_pipeline(
    input_path: str,
    output_dir: str,
    model_path: str,
    extract_axes_count: int = 10,
) -> None:
    """
    텍스트 → 임베딩 → EA 중요 축 추출(top_k)만 수행하는 파이프라인.

    결과물:
      - embeddings.json  : 단어별 pooled 임베딩 벡터
      - axes.csv         : word / axis_1_index/value, axis_2_index/value ... 구조의 EA top_k 축 정보
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) 모델 로드
    print(f"[PIPELINE-EA] 모델 로드 중: {model_path}...")
    model = LLMEmbeddingModel(model_path)

    # 2) 입력 텍스트 로드
    print(f"[PIPELINE-EA] 입력 텍스트 로드: {input_path}...")
    texts = load_words(input_path)
    print(f"[PIPELINE-EA] 총 {len(texts)}개 텍스트 임베딩 예정.")

    # 3) 임베딩 생성
    print("[PIPELINE-EA] 임베딩 생성 중...")
    embeddings: Dict[str, Dict[str, Any]] = model.embed_texts(texts)

    emb_path = out / "embeddings.json"
    print(f"[PIPELINE-EA] 임베딩 결과 저장: {emb_path}")
    save_embeddings(embeddings, str(emb_path))

    # 4) EA 축 추출
    if extract_axes_count and extract_axes_count > 0:
        print(f"[PIPELINE-EA] EA 중요 축 추출(top_k={extract_axes_count}) 수행 중...")
        axes_data = extract_axes(embeddings, top_k=extract_axes_count)

        axes_path = out / "axes.csv"
        print(f"[PIPELINE-EA] EA 축 정보 저장: {axes_path}")
        save_extracted_axes(axes_data, str(axes_path))
    else:
        print("[PIPELINE-EA] extract_axes_count = 0 → EA 축 추출 생략")

    print("[PIPELINE-EA] EA 파이프라인 완료.")
