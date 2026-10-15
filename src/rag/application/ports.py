"""Contratos que la aplicación necesita del exterior. La infraestructura los implementa."""

from typing import Protocol

from rag.domain.models import Chunk, LLMResult, RetrievedChunk


class Embedder(Protocol):
    """Convierte texto en vectores. Documentos y consultas deben pasar por el mismo modelo."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Guarda cada chunk con su vector. Si el chunk_id ya existe, lo reemplaza."""
        ...

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        """Hasta k chunks, del más parecido al menos parecido."""
        ...


class KeywordIndex(Protocol):
    def build(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Hasta k chunks que comparten palabras con la consulta, del más al menos relevante."""
        ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Vuelve a puntuar los candidatos y devuelve los top_n mejores, ordenados."""
        ...


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> LLMResult: ...
