from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# -----------------------------
# 1. 공통 변환(Transform) 계층
# -----------------------------


def transform_pca(
    embeddings: Dict[str, Dict[str, Any]],
    n_components: int = 2,
) -> Tuple[List[str], np.ndarray, PCA]:
    """
    임베딩의 pooled 벡터를 PCA 공간으로 투영합니다.

    반환:
        words: 각 행(row)에 대응하는 텍스트 리스트
        X_pca: shape = (num_words, n_components) 인 PCA 벡터 배열
        pca_model: 학습된 sklearn PCA 객체
    """
    words: List[str] = []
    vectors: List[List[float]] = []

    for word, info in embeddings.items():
        pooled = info.get("pooled")
        if pooled is None:
            continue
        words.append(word)
        vectors.append(pooled)

    if not vectors:
        raise ValueError("No 'pooled' vectors found in embeddings for PCA.")

    X = np.asarray(vectors, dtype=float)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    return words, X_pca, pca


def transform_ea(
    embeddings: Dict[str, Dict[str, Any]],
    top_k: int,
) -> Tuple[List[str], np.ndarray]:
    """
    EA(Extracted Axes) 기반 sparse 벡터를 생성합니다.

    각 단어의 pooled 벡터에서 절댓값이 큰 상위 top_k 차원만 남기고
    나머지 차원은 0으로 채운 고차원 벡터를 반환합니다.

    반환:
        words: 각 행(row)에 대응하는 텍스트 리스트
        X_ea: shape = (num_words, dim) 인 EA 기반 벡터 배열
    """
    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    words: List[str] = []
    rows: List[np.ndarray] = []

    for word, info in embeddings.items():
        pooled = info.get("pooled")
        if pooled is None:
            continue

        vec = np.asarray(pooled, dtype=float)
        dim = vec.shape[0]

        if top_k == 0 or top_k >= dim:
            # top_k가 0이거나 dim 이상이면 전체 차원을 그대로 사용
            row = vec
        else:
            idxs = np.argsort(np.abs(vec))[::-1][:top_k]
            row = np.zeros(dim, dtype=float)
            row[idxs] = vec[idxs]

        words.append(word)
        rows.append(row)

    if not rows:
        raise ValueError("No 'pooled' vectors found in embeddings for EA transform.")

    X_ea = np.vstack(rows)
    return words, X_ea


# -----------------------------
# 2. 기존 인터페이스 (PCA)
# -----------------------------


def run_pca(
    embeddings: Dict[str, Dict[str, Any]],
    n_components: int = 2,
) -> pd.DataFrame:
    """
    기존 코드와 호환되는 PCA 헬퍼.

    내부적으로 transform_pca()를 사용하여
    word / pc1 / pc2 / ... 형태의 DataFrame을 반환합니다.
    """
    words, X_pca, _ = transform_pca(embeddings, n_components=n_components)

    data = {"word": words}
    for i in range(n_components):
        data[f"pc{i + 1}"] = X_pca[:, i]

    df = pd.DataFrame(data)
    return df


def run_kmeans(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """
    PCA 결과 DataFrame 위에서 K-Means를 수행합니다.

    기본적으로 'pc'로 시작하는 컬럼들을 모두 특징으로 사용하여
    'cluster' 컬럼을 추가한 뒤 DataFrame을 반환합니다.
    """
    if n_clusters <= 0:
        raise ValueError("n_clusters must be > 0 for K-Means.")

    feature_cols = [c for c in df.columns if c.startswith("pc")]
    if not feature_cols:
        raise ValueError("No PCA columns (pc1, pc2, ...) found for K-Means.")

    X = df[feature_cols].to_numpy(dtype=float)

    model = KMeans(n_clusters=n_clusters, n_init="auto")
    labels = model.fit_predict(X)

    df = df.copy()
    df["cluster"] = labels
    return df


# -----------------------------
# 3. 시각화
# -----------------------------


def plot_pca(
    df: pd.DataFrame,
    output_path: str,
    cluster_col: str = "cluster",
) -> None:
    """
    PCA 결과를 2D 또는 3D 산점도로 저장합니다.

    - 'pc1', 'pc2'가 반드시 존재해야 하며,
    - 'pc3'가 존재하면 3D 플롯을 생성합니다.
    - cluster_col 컬럼이 있을 경우 군집별로 색상을 다르게 표시합니다.
    - 'word' 컬럼이 있으면 각 점 옆에 텍스트 라벨을 표시합니다.
    """
    has_pc1 = "pc1" in df.columns
    has_pc2 = "pc2" in df.columns

    if not (has_pc1 and has_pc2):
        raise ValueError("plot_pca requires at least 'pc1' and 'pc2' columns.")

    has_pc3 = "pc3" in df.columns
    has_cluster = cluster_col in df.columns
    has_word = "word" in df.columns

    if has_cluster:
        groups = df.groupby(cluster_col)
    else:
        groups = [(None, df)]

    if has_pc3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        for cluster_id, g in groups:
            xs = g["pc1"]
            ys = g["pc2"]
            zs = g["pc3"]
            label = f"{cluster_col} {cluster_id}" if cluster_id is not None else None
            ax.scatter(xs, ys, zs, label=label)

            if has_word:
                for _, row in g.iterrows():
                    ax.text(
                        row["pc1"],
                        row["pc2"],
                        row["pc3"],
                        str(row["word"]),
                        fontsize=8,
                        alpha=0.7,
                    )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        if has_cluster:
            ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(10, 8))

        for cluster_id, g in groups:
            xs = g["pc1"]
            ys = g["pc2"]
            label = f"{cluster_col} {cluster_id}" if cluster_id is not None else None
            ax.scatter(xs, ys, label=label)

            if has_word:
                for _, row in g.iterrows():
                    ax.text(
                        row["pc1"],
                        row["pc2"],
                        str(row["word"]),
                        fontsize=8,
                        alpha=0.7,
                    )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        if has_cluster:
            ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# -----------------------------
# 4. EA(Extracted Axes)
# -----------------------------


def extract_axes(
    embeddings: Dict[str, Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    각 단어의 pooled 벡터에서 절댓값 기준 상위 top_k 축을 추출합니다.

    반환 형식 예:
        [
            {
                "word": "happy",
                "axes": [
                    {"rank": 1, "index": 182, "value": 0.92},
                    ...
                ],
            },
            ...
        ]
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0 for extract_axes.")

    result: List[Dict[str, Any]] = []

    for word, info in embeddings.items():
        pooled = info.get("pooled")
        if pooled is None:
            continue

        vec = np.asarray(pooled, dtype=float)
        idxs = np.argsort(np.abs(vec))[::-1][:top_k]

        axes = [
            {
                "rank": r + 1,
                "index": int(idx),
                "value": float(vec[idx]),
            }
            for r, idx in enumerate(idxs)
        ]

        result.append({"word": word, "axes": axes})

    return result
