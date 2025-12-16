from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_nearest(
    target_vector: List[float],
    embedding_dict: Dict[str, Any],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    코사인 유사도로 가장 가까운 단어 찾기.
    """
    words = list(embedding_dict.keys())
    vectors = np.array([embedding_dict[w]["pooled"] for w in words])
    target = np.array(target_vector).reshape(1, -1)

    sims = cosine_similarity(target, vectors)[0]
    idxs = sims.argsort()[::-1][:top_k]

    return [(words[i], float(sims[i])) for i in idxs]


def decode(
    target_vector: List[float],
    embedding_dict: Dict[str, Any],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    의미적 디코딩: 벡터가 의미적으로 어떤 단어에 가까운지 반환.
    """
    return find_nearest(target_vector, embedding_dict, top_k)
