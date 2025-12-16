from statistics import mean
from typing import Any, Dict, List

from llama_cpp import Llama


class LLMEmbeddingModel:
    """
    텍스트를 임베딩 벡터로 변환하는 모델 래퍼.
    llama.cpp 기반 GGUF 모델을 사용합니다.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            embedding=True,
            verbose=False,
        )

    def embed_text(self, text: str) -> Dict[str, Any]:
        """
        텍스트 한 개를 임베딩합니다.
        """
        emb = self.llm.embed(text)

        # 토큰 단위 또는 단일 벡터 보호 처리
        if emb and isinstance(emb[0], (list, tuple)):
            token_vecs = emb
        else:
            token_vecs = [emb]

        dim = len(token_vecs[0])
        pooled = [mean(d) for d in zip(*token_vecs)]

        return {
            "pooled": pooled,
            "count": len(token_vecs),
            "dim": dim,
        }

    def embed_texts(self, texts: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        여러 텍스트를 한 번에 임베딩합니다.
        """
        return {text: self.embed_text(text) for text in texts}
