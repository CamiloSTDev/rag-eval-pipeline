"""Caso de uso: los chunks más parecidos a una consulta, por rama o fusionados."""

from typing import Literal

from rag.application.retrieval import Retriever
from rag.domain.models import RetrievedChunk

SearchMode = Literal["vector", "bm25", "hybrid"]


class SearchChunks:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def __call__(self, query: str, k: int, mode: SearchMode = "vector") -> list[RetrievedChunk]:
        if mode != "vector" and not self._retriever.hybrid:
            raise ValueError(f"El modo {mode} necesita un índice BM25 en el Retriever")
        retrieval = self._retriever(query, k)
        if mode == "vector":
            return retrieval.dense
        if mode == "bm25":
            return retrieval.sparse or []
        return retrieval.ranking
