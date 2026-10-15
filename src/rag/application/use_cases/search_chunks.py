"""Caso de uso: los chunks más parecidos a una consulta, solo con búsqueda vectorial."""

from rag.application.ports import Embedder, VectorStore
from rag.domain.models import RetrievedChunk


class SearchChunks:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def __call__(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._store.search(self._embedder.embed_query(query), k)
