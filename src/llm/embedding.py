import hashlib
from typing import Optional, Sequence, Iterable

import numpy as np

# ============================================================
# Public API (ONLY these should be used outside this file)
# ============================================================

def embed_texts_normalized(
    texts: Sequence[str],
    dim: int,
    *,
    backend: str = "hash",
    llm: Optional[object] = None,
) -> np.ndarray:
    """
    Shared embedding utility used by ontology / infer / maker.

    Contract:
      - All returned vectors are L2-normalized
      - All vectors live in ONE embedding space per backend
      - Mixing backends for cosine similarity is INVALID

    Parameters
    ----------
    texts : Sequence[str]
        Input texts
    dim : int
        Expected embedding dimension (required for validation)
    backend : str
        "hash"       : deterministic pseudo-embedding (TEST ONLY)
        "llama_cpp"  : llama_cpp.Llama.embed() based embedding (PRODUCTION)
    llm : Optional[object]
        Required when backend == "llama_cpp"

    Returns
    -------
    np.ndarray
        shape: (N, D), float32, L2-normalized
    """
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    vecs: list[np.ndarray] = []
    for text in texts:
        raw = _embed_raw(text, dim=dim, backend=backend, llm=llm)
        vec = _l2_normalize(raw)
        vecs.append(vec)

    out = np.stack(vecs, axis=0).astype(np.float32)

    # Final dimension sanity check
    if out.shape[1] != dim:
        raise ValueError(
            f"Embedding dim mismatch: expected {dim}, got {out.shape[1]} (backend={backend})"
        )

    return out


def embed_text_normalized(
    text: str,
    dim: int,
    *,
    backend: str = "hash",
    llm: Optional[object] = None,
) -> np.ndarray:
    """Single-text version of `embed_texts_normalized`."""
    return embed_texts_normalized([text], dim, backend=backend, llm=llm)[0]


# ============================================================
# Internal helpers (DO NOT import from outside this file)
# ============================================================

def _embed_raw(text: str, dim: int, *, backend: str, llm: Optional[object]) -> np.ndarray:
    text = str(text).strip()
    if not text:
        return np.zeros((dim,), dtype=np.float32)

    if backend == "hash":
        return _hash_embed_raw(text, dim)

    if backend == "llama_cpp":
        if llm is None:
            raise ValueError('backend="llama_cpp" requires llm argument (llama_cpp.Llama instance).')

        emb = llm.embed(text)
        vec = _to_sentence_vector(emb)

        if vec.ndim != 1:
            raise ValueError(f"Invalid llama_cpp embedding shape: {vec.shape}")

        if vec.shape[0] != dim:
            raise ValueError(
                f"llama_cpp embedding dim mismatch: expected {dim}, got {vec.shape[0]}"
            )

        return vec.astype(np.float32)

    raise ValueError(f"Unknown embedding backend: {backend!r}")


def _to_sentence_vector(emb: Iterable) -> np.ndarray:
    """
    Normalize llama_cpp embed() outputs into a single sentence vector.
    - token-level embeddings -> mean pooling
    - single vector -> passthrough
    """
    if isinstance(emb, list) and emb and isinstance(emb[0], (list, tuple, np.ndarray)):
        return np.mean(np.asarray(emb, dtype=np.float32), axis=0)
    return np.asarray(emb, dtype=np.float32)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        return vec / norm
    return vec


def _hash_embed_raw(text: str, dim: int) -> np.ndarray:
    """
    Deterministic pseudo-embedding (TEST ONLY).
    - NOT semantically meaningful
    - Must NEVER be mixed with model-based embeddings
    """
    seed = text.encode("utf-8")
    buf = b""
    h = seed

    while len(buf) < dim * 4:
        h = hashlib.sha256(h).digest()
        buf += h

    out = np.empty((dim,), dtype=np.float32)
    denom = 4294967295.0  # 2^32 - 1
    for i in range(dim):
        chunk = buf[i * 4 : (i + 1) * 4]
        u = int.from_bytes(chunk, "little", signed=False)
        out[i] = (u / denom) * 2.0 - 1.0

    return out
