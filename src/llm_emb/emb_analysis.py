# src/llm_emb/emb_analysis.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# -----------------------------
# PCA 결과 메타/리포트 구조
# -----------------------------


@dataclass
class PcaAxisInfo:
    pc: str                 # "pc1", "pc2", "pc3" ...
    dominant_dim: int       # 원본 임베딩 차원 중 |loading|이 가장 큰 index
    dominant_loading: float # 그 loading 값
    top_dims: List[int]     # 상위 차원 index 목록 (내림차순)
    top_loadings: List[float]  # 상위 loading 값 목록 (top_dims와 길이 동일)


@dataclass
class PcaReport:
    n_samples: int
    embedding_dim: int
    n_components: int
    explained_variance_ratio: List[float]  # 길이 = n_components
    axes: List[PcaAxisInfo]                # pc1..pcN 정보


# -----------------------------
# 1) Embeddings -> matrix
# -----------------------------


def _embeddings_to_matrix(
    embeddings: Dict[str, Dict[str, Any]],
    use_key: str = "pooled",
) -> Tuple[List[str], np.ndarray]:
    """
    embeddings dict:
      { "word": {"pooled":[...], ...}, ... }

    returns:
      words(list), X(np.ndarray) shape = (n, dim)
    """
    words: List[str] = []
    rows: List[np.ndarray] = []

    for word, info in embeddings.items():
        vec = info.get(use_key)
        if vec is None:
            continue
        v = np.asarray(vec, dtype=float)
        if v.ndim != 1:
            continue
        words.append(word)
        rows.append(v)

    if not rows:
        raise ValueError(f"embeddings_to_matrix: no vectors found with key='{use_key}'")

    X = np.vstack(rows)
    return words, X


# -----------------------------
# 2) PCA / KMeans
# -----------------------------


def run_pca(
    embeddings: Dict[str, Dict[str, Any]],
    n_components: int = 2,
    use_key: str = "pooled",
    return_model: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, PCA]:
    """
    pooled 임베딩 벡터들로 PCA 수행.
    - 반환 DF 컬럼: word, pc1, pc2(, pc3)
    - return_model=True면 (df, pca_model) 반환
    """
    words, X = _embeddings_to_matrix(embeddings, use_key=use_key)

    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)

    data = {"word": words}
    for i in range(n_components):
        data[f"pc{i+1}"] = Z[:, i]

    df = pd.DataFrame(data)

    if return_model:
        return df, pca
    return df


def run_kmeans(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    df의 pc1/pc2/(pc3) 컬럼을 사용해서 KMeans 클러스터링.
    """
    pc_cols = [c for c in df.columns if c.startswith("pc")]
    if not pc_cols:
        raise ValueError("run_kmeans: PCA columns not found. Run PCA first.")

    X = df[pc_cols].to_numpy(dtype=float)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    out = df.copy()
    out["cluster"] = labels
    return out


# -----------------------------
# 3) Plot
# -----------------------------


def plot_pca(df: pd.DataFrame, out_path: str, is_3d: bool = False, show_labels: bool = True) -> None:
    """
    PCA 결과 플롯 저장.
    - is_3d=True면 pc1/pc2/pc3 사용
    - show_labels=True면 각 점에 단어 텍스트 표시
    """
    pc_cols = [c for c in df.columns if c.startswith("pc")]
    if is_3d and len(pc_cols) < 3:
        raise ValueError("plot_pca: 3D requires pc1, pc2, pc3.")
    if not is_3d and len(pc_cols) < 2:
        raise ValueError("plot_pca: 2D requires pc1, pc2.")

    has_cluster = "cluster" in df.columns

    if is_3d:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        if has_cluster:
            ax.scatter(df["pc1"], df["pc2"], df["pc3"], c=df["cluster"].astype(int), s=30)
        else:
            ax.scatter(df["pc1"], df["pc2"], df["pc3"], s=30)

        if show_labels:
            for _, r in df.iterrows():
                ax.text(r["pc1"], r["pc2"], r["pc3"], str(r["word"]), fontsize=8)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    # 2D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    if has_cluster:
        ax.scatter(df["pc1"], df["pc2"], c=df["cluster"].astype(int), s=30)
    else:
        ax.scatter(df["pc1"], df["pc2"], s=30)

    if show_labels:
        for _, r in df.iterrows():
            ax.text(r["pc1"], r["pc2"], str(r["word"]), fontsize=9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# 4) PCA "지배 축" 추출 (PC -> original dim index)
# -----------------------------


def build_pca_report(pca: PCA, embedding_dim: int, n_samples: int, top_k_dims: int = 8) -> PcaReport:
    """
    PCA 축은 원본 축의 선형결합이므로 "PC1=12번 축" 같은 1:1은 불가.
    대신 각 PC에서 |loading|이 가장 큰 원본 차원 index를 dominant_dim으로 뽑아 리포트한다.
    """
    comps = pca.components_  # shape: (n_components, embedding_dim)
    n_components = comps.shape[0]

    axes: List[PcaAxisInfo] = []
    for i in range(n_components):
        load = comps[i]
        abs_load = np.abs(load)
        order = np.argsort(abs_load)[::-1]  # 큰 순

        dominant_dim = int(order[0])
        dominant_loading = float(load[dominant_dim])

        top_dims = [int(x) for x in order[:top_k_dims]]
        top_loadings = [float(load[d]) for d in top_dims]

        axes.append(
            PcaAxisInfo(
                pc=f"pc{i+1}",
                dominant_dim=dominant_dim,
                dominant_loading=dominant_loading,
                top_dims=top_dims,
                top_loadings=top_loadings,
            )
        )

    return PcaReport(
        n_samples=n_samples,
        embedding_dim=int(embedding_dim),
        n_components=int(n_components),
        explained_variance_ratio=[float(x) for x in pca.explained_variance_ratio_],
        axes=axes,
    )


def report_to_components_df(report: PcaReport) -> pd.DataFrame:
    """
    pc별 top_dims/top_loadings를 표로 정리해 CSV로 저장하기 쉽게 만든 DF.
    """
    rows = []
    for a in report.axes:
        for rank, (d, w) in enumerate(zip(a.top_dims, a.top_loadings), start=1):
            rows.append(
                {
                    "pc": a.pc,
                    "rank": rank,
                    "dim_index": d,
                    "loading": w,
                    "dominant_dim": a.dominant_dim,
                    "dominant_loading": a.dominant_loading,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------
# 5) EA(Extracted Axes) - 자리 유지용 (현재는 pooled 기반 top-k 축)
# -----------------------------


def extract_axes(
    embeddings: Dict[str, Dict[str, Any]],
    top_k: int = 10,
    use_key: str = "pooled",
) -> List[Dict[str, Any]]:
    """
    EA(Extracted Axes) 1단계: 각 단어 pooled 벡터에서 |value| 큰 차원 top_k를 뽑는다.
    (현재는 해석/시각화/클러스터링은 확장 예정)
    """
    result: List[Dict[str, Any]] = []
    for word, info in embeddings.items():
        vec = info.get(use_key)
        if vec is None:
            continue
        v = np.asarray(vec, dtype=float)
        if v.ndim != 1:
            continue

        dim = v.shape[0]
        if top_k <= 0 or top_k >= dim:
            idxs = np.argsort(np.abs(v))[::-1]
        else:
            idxs = np.argsort(np.abs(v))[::-1][:top_k]

        axes = [{"rank": r + 1, "index": int(idx), "value": float(v[idx])} for r, idx in enumerate(idxs)]
        result.append({"word": word, "axes": axes})

    return result
