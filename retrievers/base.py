from abc import ABC, abstractmethod
from typing import List, Tuple


class Retriever(ABC):
    def __init__(self, keywords: List[str]):
        self.keywords: List[str] = keywords

    @abstractmethod
    def score(self, query: str, keyword: str) -> float:
        """
        Return a similarity score where higher is better.
        """
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        for term in self.keywords:
            try:
                s = self.score(query, term)
            except Exception:
                s = 0.0
            scored.append((term, float(s)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, top_k)] 