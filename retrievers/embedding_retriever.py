from typing import List
from .base import Retriever

try:
    import torch  # noqa: F401
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np
    from numpy.linalg import norm
except Exception:
    SentenceTransformer = None
    np = None


def cosine_similarity(a, b):
    denom = (norm(a) * norm(b))
    if denom == 0:
        return 0.0
    return float(a.dot(b) / denom)


class EmbeddingRetriever(Retriever):
    def __init__(self, keywords: List[str], model_name: str = 'sentence-transformers/paraphrase-MiniLM-L6-v2'):
        if SentenceTransformer is None or np is None:
            raise RuntimeError("sentence-transformers and numpy are required. Install with: pip install sentence-transformers numpy torch")
        super().__init__(keywords)
        self.model = SentenceTransformer(model_name)
        self.keyword_embeddings = self.model.encode(self.keywords, convert_to_numpy=True, normalize_embeddings=False)

    def score(self, query: str, keyword: str) -> float:
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=False)[0]
        try:
            idx = self.keywords.index(keyword)
        except ValueError:
            return 0.0
        k_emb = self.keyword_embeddings[idx]
        return cosine_similarity(q_emb, k_emb)

    def search(self, query: str, top_k: int = 5):
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=False)[0]
        scores = [cosine_similarity(q_emb, k_emb) for k_emb in self.keyword_embeddings]
        pairs = list(zip(self.keywords, [float(s) for s in scores]))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[: max(1, top_k)] 