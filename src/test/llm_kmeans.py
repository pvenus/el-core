import argparse
import os

import pandas as pd
from sklearn.cluster import KMeans
from typing import Optional


def run_kmeans(
    input_csv: str = "data/embeddings/pca/emotion_pca_overall_3d.csv",
    output_csv: Optional[str] = None,
    k: int = 6,
    random_state: int = 42,
    n_init: int = 10,
) -> None:
    """
    Run K-Means clustering on PCA(3D) emotion data.

    - input_csv: CSV 파일 경로 (pc1, pc2, pc3 컬럼 필수)
    - output_csv: 결과를 저장할 CSV 경로 (None 이면 자동 생성)
    - k: 클러스터 개수
    - random_state: KMeans 랜덤 시드
    - n_init: KMeans n_init 파라미터
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = {"pc1", "pc2", "pc3"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Input CSV must contain columns {required_cols}, "
            f"but got: {df.columns.tolist()}"
        )

    X = df[["pc1", "pc2", "pc3"]].to_numpy()

    print("====================================")
    print(f"[KMEANS] Input: {input_csv}")
    print(f"[KMEANS] Num words: {len(df)}")
    print(f"[KMEANS] Dim used: 3 (pc1, pc2, pc3)")
    print(f"[KMEANS] k = {k}, random_state = {random_state}, n_init = {n_init}")

    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    df["cluster"] = labels

    # output 경로 자동 생성
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_k{k}_kmeans.csv"

    df.to_csv(output_csv, index=False)

    print(f"[KMEANS] Saved clustered CSV -> {output_csv}")

    # 클러스터 요약 출력 (word 컬럼이 있을 때만)
    word_col = None
    for cand in ["word", "token", "emotion", "label"]:
        if cand in df.columns:
            word_col = cand
            break

    print("[KMEANS] Cluster summary:")
    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]
        count = len(sub)
        if word_col is not None:
            sample_words = sub[word_col].head(8).tolist()
            print(f"  - Cluster {cid}: n={count}, samples={sample_words}")
        else:
            print(f"  - Cluster {cid}: n={count}")

    print("====================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-Means clustering on emotion PCA (3D)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/embeddings/pca/emotion_pca_overall_3d.csv",
        help="입력 CSV 파일 경로 (pc1, pc2, pc3 포함)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 CSV 파일 경로 (지정하지 않으면 자동 생성)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="클러스터 개수 (K)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="KMeans random_state 값",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=10,
        help="KMeans n_init 값",
    )

    args = parser.parse_args()

    run_kmeans(
        input_csv=args.input,
        output_csv=args.output,
        k=args.k,
        random_state=args.random_state,
        n_init=args.n_init,
    )


if __name__ == "__main__":
    main()