"""Recuperación de candidatos: búsqueda vectorial y, si hay índice por palabras, fusión RRF."""

from dataclasses import dataclass

from rag.application.ports import Embedder, KeywordIndex, VectorStore
from rag.domain.fusion import reciprocal_rank_fusion
from rag.domain.models import RetrievedChunk


@dataclass(frozen=True)
class Retrieval:
    ranking: list[RetrievedChunk]  # lo que se usa: el vectorial, o el fusionado si hay BM25
    dense: list[RetrievedChunk]  # rama vectorial
    sparse: list[RetrievedChunk] | None  # rama por palabras; None sin índice BM25

    @property
    def top_similarity(self) -> float | None:
        """Mejor similitud vectorial. Es la que se compara con el umbral de rechazo: los
        puntajes de RRF tienen otra escala (máximo 2/61) y lo romperían."""
        return self.dense[0].score if self.dense else None


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        keywords: KeywordIndex | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._keywords = keywords
        self._rrf_k = rrf_k

    @property
    def hybrid(self) -> bool:
        return self._keywords is not None

    def __call__(self, query: str, k: int) -> Retrieval:
        dense = self._store.search(self._embedder.embed_query(query), k)
        if self._keywords is None:
            return Retrieval(ranking=dense, dense=dense, sparse=None)
        sparse = self._keywords.search(query, k)
        fused = reciprocal_rank_fusion([dense, sparse], k=self._rrf_k)[:k]
        return Retrieval(ranking=fused, dense=dense, sparse=sparse)
