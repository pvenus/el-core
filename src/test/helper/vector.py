from typing import List, Sequence

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))

def _norm(a: Sequence[float]) -> float:
    return _dot(a, a) ** 0.5

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
    n = vector_norm(v)
    if n < eps:
        return [0.0 for _ in v]
    return [float(x) / n for x in v]


def vector_l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Explicit L2 distance alias for clarity in higher-level code."""
    return l2_distance(a, b)
