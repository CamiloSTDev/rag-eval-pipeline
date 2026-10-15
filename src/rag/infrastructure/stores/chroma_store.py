"""Índice vectorial persistente en disco con Chroma."""

from pathlib import Path
from typing import Any

import chromadb

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
        # upsert y no add: un chunk_id que ya existe se reemplaza, así reingerir no duplica
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_metadata(chunk) for chunk in chunks],
        )

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        rows = zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        )
        # Chroma devuelve distancia coseno (0 = idénticos); el puerto pide "mayor = mejor"
        return [
            RetrievedChunk(_chunk(chunk_id, text, metadata), score=1.0 - distance)
            for chunk_id, text, metadata, distance in rows
        ]


def _metadata(chunk: Chunk) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source": chunk.source, "position": chunk.position}
    if chunk.page is not None:
        metadata["page"] = chunk.page
    if chunk.section is not None:
        metadata["section"] = chunk.section
    return metadata


def _chunk(chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source=metadata["source"],
        page=metadata.get("page"),
        section=metadata.get("section"),
        position=metadata["position"],
    )
