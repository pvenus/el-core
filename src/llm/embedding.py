import hashlib
from typing import Optional, Sequence

import numpy as np


def embed_texts_normalized(
    texts: Sequence[str],
    dim: int,
    *,
    backend: str = "hash",
    llm: Optional[object] = None,
) -> np.ndarray:
    """
    텍스트 리스트 -> 임베딩 벡터 -> (필요 시) mean pooling -> L2 normalize 까지를
    이 파일에서 모두 책임지는 공용 유틸.

    현재 지원:
      - backend="hash": 결정적(deterministic) 해시 기반 임베딩 (모델 없이 테스트/시나리오 생성용)

    확장 예정(이 파일 내에서 확장):
      - backend="llama_cpp": llama_cpp LLM 객체의 .embed(text) 사용 (llm 인자 필요)

    Parameters
    ----------
    texts : Sequence[str]
        임베딩할 텍스트 리스트
    dim : int
        backend="hash"일 때 생성할 임베딩 차원
        (다른 backend에서는 무시되거나 검증에만 사용될 수 있음)
    backend : str
        임베딩 백엔드 선택 ("hash", ...)
    llm : Optional[object]
        backend가 모델 기반일 때 사용할 LLM 객체 (예: llama_cpp.Llama)

    Returns
    -------
    np.ndarray
        shape: (N, D), L2-normalized embedding vectors
    """
    vectors = []
    for text in texts:
        vec = _embed_raw(text, dim=dim, backend=backend, llm=llm)
        vec = _l2_normalize(vec)
        vectors.append(vec)
    return np.stack(vectors, axis=0)


def embed_text_normalized(
    text: str,
    dim: int,
    *,
    backend: str = "hash",
    llm: Optional[object] = None,
) -> np.ndarray:
    """단일 텍스트 임베딩 + L2 정규화."""
    return embed_texts_normalized([text], dim, backend=backend, llm=llm)[0]


# -------------------------
# Internal helpers (keep in this file)
# -------------------------


def _embed_raw(text: str, dim: int, *, backend: str, llm: Optional[object]) -> np.ndarray:
    if backend == "hash":
        return _hash_embed_raw(text, dim)

    if backend == "llama_cpp":
        if llm is None:
            raise ValueError('backend="llama_cpp" requires llm argument (llama_cpp.Llama instance).')
        emb = llm.embed(text)
        vec = _to_sentence_vector(emb)
        # Optional dim check (do not hard-fail unless dim is obviously mismatched)
        if dim is not None and isinstance(dim, int) and dim > 0 and vec.shape[0] != dim:
            # Allow mismatch silently; caller may pass dim for hash only.
            pass
        return vec

    raise ValueError(f"Unknown embedding backend: {backend!r}")


def _to_sentence_vector(emb) -> np.ndarray:
    """
    llama_cpp embed() 등의 반환 형태(토큰별 리스트 vs 단일 벡터)를
    문장/텍스트 단일 벡터로 통일한다.
    """
    # 토큰 단위 임베딩인 경우 → mean pooling
    if isinstance(emb, list) and emb and isinstance(emb[0], (list, tuple)):
        return np.mean(np.asarray(emb, dtype=np.float32), axis=0)
    return np.asarray(emb, dtype=np.float32)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        return vec / norm
    return vec


def _hash_embed_raw(text: str, dim: int) -> np.ndarray:
    """
    모델 없이도 재현 가능한(결정적) pseudo-embedding 생성.
    - sha256 chaining으로 충분한 바이트 확보
    - [0, 2^32-1] -> [-1, 1] 매핑
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
