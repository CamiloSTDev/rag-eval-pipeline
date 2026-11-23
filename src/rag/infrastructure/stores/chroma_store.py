"""Índice vectorial persistente en disco con Chroma."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import PyEmbeddings

from rag.domain.models import Chunk, RetrievedChunk


class ChromaStore:
    def __init__(self, path: Path, collection: str) -> None:
        client = chromadb.PersistentClient(path=str(path))
        # Sin función de embeddings de Chroma: los vectores llegan calculados por el Embedder
        self._collection = client.get_or_create_collection(
            collection,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        embeddings: PyEmbeddings = list(vectors)
        # upsert y no add: un chunk_id que ya existe se reemplaza, así reingerir no duplica
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_metadata(chunk) for chunk in chunks],
        )

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        query: PyEmbeddings = [vector]
        result = self._collection.query(
            query_embeddings=query,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        documents, metadatas, distances = (
            result["documents"],
            result["metadatas"],
            result["distances"],
        )
        if documents is None or metadatas is None or distances is None:
            raise RuntimeError("Chroma no devolvió los documentos, metadatos o distancias pedidos")
        rows = zip(result["ids"][0], documents[0], metadatas[0], distances[0], strict=True)
        # Chroma devuelve distancia coseno (0 = idénticos); el puerto pide "mayor = mejor"
        return [
            RetrievedChunk(_chunk(chunk_id, text, metadata), score=1.0 - distance)
            for chunk_id, text, metadata, distance in rows
        ]

    def all_chunks(self) -> list[Chunk]:
        """Todos los chunks guardados, en orden estable: con ellos se arma el índice BM25."""
        result = self._collection.get(include=["documents", "metadatas"])
        documents, metadatas = result["documents"], result["metadatas"]
        if documents is None or metadatas is None:
            raise RuntimeError("Chroma no devolvió los documentos o metadatos pedidos")
        chunks = [
            _chunk(chunk_id, text, metadata)
            for chunk_id, text, metadata in zip(result["ids"], documents, metadatas, strict=True)
        ]
        return sorted(chunks, key=lambda chunk: (chunk.source, chunk.page or 0, chunk.position))


def _metadata(chunk: Chunk) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source": chunk.source, "position": chunk.position}
    if chunk.page is not None:
        metadata["page"] = chunk.page
    if chunk.section is not None:
        metadata["section"] = chunk.section
    return metadata


def _chunk(chunk_id: str, text: str, metadata: Mapping[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source=metadata["source"],
        page=metadata.get("page"),
        section=metadata.get("section"),
        position=metadata["position"],
    )
