from typing import List
from .base import Retriever
from similarity.Smith_Waterman import SimpleCharacterAligner


class SmithWatermanRetriever(Retriever):
    def __init__(self, keywords: List[str]):
        super().__init__(keywords)
        self.aligner = SimpleCharacterAligner()

    def score(self, query: str, keyword: str) -> float:
        result = self.aligner.align(query or '', keyword or '', normalize=True)
        return float(result.get('normalized_score') or 0.0) 