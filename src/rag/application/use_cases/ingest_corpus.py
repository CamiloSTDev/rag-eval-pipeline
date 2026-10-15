"""Caso de uso: partir los documentos en chunks, calcular sus vectores y guardarlos."""

from dataclasses import dataclass

from rag.application.config import ChunkingConfig
from rag.application.ports import Embedder, VectorStore
from rag.domain.chunking import TokenCounter, chunk_document
from rag.domain.models import Chunk, Document


@dataclass(frozen=True)
class IngestReport:
    documents: int
    chunks: int


class IngestCorpus:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        count_tokens: TokenCounter,
        config: ChunkingConfig,
        batch_size: int = 100,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._count_tokens = count_tokens
        self._config = config
        self._batch_size = batch_size

    def __call__(self, documents: list[Document]) -> IngestReport:
        pairs = [
            (chunk, _embedding_text(chunk, document))
            for document in documents
            for chunk in chunk_document(
                document,
                max_tokens=self._config.max_tokens,
                overlap_tokens=self._config.overlap_tokens,
                count_tokens=self._count_tokens,
            )
        ]
        # Por lotes: si algo falla a mitad, lo guardado sirve y reintentar no duplica nada
        for start in range(0, len(pairs), self._batch_size):
            batch = pairs[start : start + self._batch_size]
            vectors = self._embedder.embed_documents([text for _, text in batch])
            self._store.add([chunk for chunk, _ in batch], vectors)
        return IngestReport(documents=len(documents), chunks=len(pairs))


def _embedding_text(chunk: Chunk, document: Document) -> str:
    """Texto que se convierte en vector: el chunk precedido por su sección, o por el título."""
    heading = chunk.section or document.title
    return f"{heading}\n\n{chunk.text}" if heading else chunk.text
