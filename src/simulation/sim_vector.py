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
