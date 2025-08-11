from typing import List
from .base import Retriever

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception as exc:  # pragma: no cover
    BM25Okapi = None


def simple_tokenize(text: str) -> List[str]:
    text = (text or '').strip()
    if not text:
        return []
    if ' ' in text:
        return [t for t in text.split() if t]
    return list(text)


class BM25Retriever(Retriever):
    def __init__(self, keywords: List[str]):
        if BM25Okapi is None:
            raise RuntimeError("rank-bm25 is required for BM25Retriever. Install with: pip install rank-bm25")
        super().__init__(keywords)
        self.corpus_tokens: List[List[str]] = [simple_tokenize(k) for k in keywords]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def score(self, query: str, keyword: str) -> float:
        # BM25 is naturally a per-corpus retrieval; we will compute per-item when sorting.
        # For efficiency, prefer overriding search; but keep score for compatibility.
        q_tokens = simple_tokenize(query)
        # Approximate: use idf-weighted term overlap as a quick proxy
        scores = self.bm25.get_scores(q_tokens)
        # Map keyword to its index
        try:
            idx = self.keywords.index(keyword)
        except ValueError:
            return 0.0
        return float(scores[idx])

    def search(self, query: str, top_k: int = 5):
        q_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        pairs = list(zip(self.keywords, [float(s) for s in scores]))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[: max(1, top_k)] 