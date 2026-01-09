# helper/vector.py
from __future__ import annotations

from typing import List, Sequence

# -----------------------------
# internal primitives
# -----------------------------

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))

def _norm(a: Sequence[float]) -> float:
    return _dot(a, a) ** 0.5

# -----------------------------
# constructors / helpers
# -----------------------------

def zero_vector(dim: int) -> List[float]:
    """
    Create a zero vector of given dimension.
    """
    if dim < 0:
        raise ValueError("dim must be >= 0")
    return [0.0 for _ in range(dim)]

# -----------------------------
# core vector ops (math_vector 병합)
# -----------------------------

def add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Element-wise add."""
    return [float(x) + float(y) for x, y in zip(a, b)]

def sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Element-wise subtract."""
    return [float(x) - float(y) for x, y in zip(a, b)]

def scale(v: Sequence[float], s: float) -> List[float]:
    """Scale vector by scalar."""
    ss = float(s)
    return [float(x) * ss for x in v]

def clamp_norm(v: Sequence[float], max_norm: float, eps: float = 1e-12) -> List[float]:
    """
    Clamp vector magnitude to max_norm (if current norm > max_norm).
    """
    mn = float(max_norm)
    if mn <= 0:
        return [0.0 for _ in v]

    n = _norm(v)
    if n < eps:
        return [0.0 for _ in v]
    if n <= mn:
        return [float(x) for x in v]

    ratio = mn / n
    return [float(x) * ratio for x in v]

def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Public dot product."""
    return _dot(a, b)

def cosine_similarity(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> float:
    """
    Cosine similarity between vectors.
    """
    na = _norm(a)
    nb = _norm(b)
    denom = na * nb
    if denom < eps:
        return 0.0
    return _dot(a, b) / denom

# -----------------------------
# existing API (keep)
# -----------------------------

def normalize(v: Sequence[float], eps: float = 1e-12) -> List[float]:
    n = _norm(v)
    if n < eps:
        return [0.0 for _ in v]
    return [float(x) / n for x in v]

def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5

# ------------------------------------------------------------
# Generic aliases for semantic / embedding use-cases
# (These wrap pure math utilities above to avoid name confusion)
# ------------------------------------------------------------

def vector_norm(v: Sequence[float]) -> float:
    """Alias of _norm for general vector magnitude."""
    return _norm(v)

def normalize_vector(v: Sequence[float], eps: float = 1e-12) -> List[float]:
    """Generic vector normalization (semantic-safe alias)."""
    return normalize(v, eps=eps)

def vector_l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Explicit L2 distance alias for clarity in higher-level code."""
    return l2_distance(a, b)


# ------------------------------------------------------------
# Similarity utilities (cosine-based) for embeddings / semantics
# ------------------------------------------------------------

def cosine_similarity(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> float:
    """Cosine similarity between two vectors.

    Notes
    -----
    - If vectors are already L2-normalized, this equals their dot product.
    - If either vector has ~0 norm, returns 0.0.
    """
    na = vector_norm(a)
    nb = vector_norm(b)
    if na < eps or nb < eps:
        return 0.0
    return _dot(a, b) / (na * nb)


def cosine_distance(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> float:
    """Cosine distance = 1 - cosine_similarity."""
    return 1.0 - cosine_similarity(a, b, eps=eps)


def pairwise_cosine_similarity(
    vectors: Sequence[Sequence[float]],
    *,
    eps: float = 1e-12,
    normalize_inputs: bool = False,
) -> List[List[float]]:
    """Compute an NxN cosine similarity matrix.

    Parameters
    ----------
    vectors:
        List of vectors.
    normalize_inputs:
        If True, L2-normalize each input vector before computing.
        Useful when inputs may not already be normalized.
    """
    vecs: List[List[float]] = [list(v) for v in vectors]
    if normalize_inputs:
        vecs = [normalize_vector(v, eps=eps) for v in vecs]

    n = len(vecs)
    out: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        out[i][i] = 1.0
        for j in range(i + 1, n):
            s = cosine_similarity(vecs[i], vecs[j], eps=eps)
            out[i][j] = s
            out[j][i] = s
    return out


def nearest_neighbors_by_cosine(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    k: int = 1,
    eps: float = 1e-12,
    normalize_inputs: bool = False,
    include_self: bool = False,
) -> List[List[tuple[str, float]]]:
    """For each vector, return top-k nearest neighbors (cosine similarity).

    Returns
    -------
    List[List[(label, cosine_sim)]]
        Outer list is per input i, inner list is sorted high->low similarity.
    """
    if len(vectors) != len(labels):
        raise ValueError(f"vectors/labels length mismatch: {len(vectors)} vs {len(labels)}")
    if k <= 0:
        raise ValueError("k must be >= 1")

    sim = pairwise_cosine_similarity(vectors, eps=eps, normalize_inputs=normalize_inputs)
    n = len(labels)
    result: List[List[tuple[str, float]]] = []

    for i in range(n):
        pairs: List[tuple[str, float]] = []
        for j in range(n):
            if (not include_self) and i == j:
                continue
            pairs.append((str(labels[j]), float(sim[i][j])))
        pairs.sort(key=lambda x: x[1], reverse=True)
        result.append(pairs[: min(k, len(pairs))])
    return result


def cosine_spread_stats(
    vectors: Sequence[Sequence[float]],
    *,
    eps: float = 1e-12,
    normalize_inputs: bool = False,
) -> dict:
    """Quick summary stats for how "spread" vectors are under cosine similarity.

    Useful for checking whether embedding points are clustered.

    Returns
    -------
    dict with:
      - n: number of vectors
      - min_offdiag_cos
      - mean_offdiag_cos
      - max_offdiag_cos
    """
    sim = pairwise_cosine_similarity(vectors, eps=eps, normalize_inputs=normalize_inputs)
    n = len(sim)
    if n <= 1:
        return {
            "n": n,
            "min_offdiag_cos": 0.0,
            "mean_offdiag_cos": 0.0,
            "max_offdiag_cos": 0.0,
        }

    vals: List[float] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vals.append(float(sim[i][j]))

    return {
        "n": n,
        "min_offdiag_cos": min(vals) if vals else 0.0,
        "mean_offdiag_cos": (sum(vals) / len(vals)) if vals else 0.0,
        "max_offdiag_cos": max(vals) if vals else 0.0,
    }


def analyze_vector_similarity(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    top_k: int = 1,
    eps: float = 1e-12,
    normalize_inputs: bool = False,
) -> dict:
    """Comprehensive similarity analysis in one call.

    This is meant for quick diagnostics and UI display.

    Returns
    -------
    dict with:
      - labels
      - cosine_similarity_matrix
      - spread_stats
      - nearest_neighbors (top-k per label)
      - closest_pairs (global top pairs)
    """
    if len(vectors) != len(labels):
        raise ValueError(f"vectors/labels length mismatch: {len(vectors)} vs {len(labels)}")

    sim = pairwise_cosine_similarity(vectors, eps=eps, normalize_inputs=normalize_inputs)
    stats = cosine_spread_stats(vectors, eps=eps, normalize_inputs=normalize_inputs)
    nn = nearest_neighbors_by_cosine(
        vectors,
        labels,
        k=top_k,
        eps=eps,
        normalize_inputs=normalize_inputs,
        include_self=False,
    )

    # Global closest pairs (highest cosine, off-diagonal)
    pairs: List[tuple[str, str, float]] = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((str(labels[i]), str(labels[j]), float(sim[i][j])))
    pairs.sort(key=lambda x: x[2], reverse=True)

    return {
        "labels": [str(x) for x in labels],
        "cosine_similarity_matrix": sim,
        "spread_stats": stats,
        "nearest_neighbors": nn,
        "closest_pairs": pairs[: max(0, min(50, len(pairs)))],
    }
