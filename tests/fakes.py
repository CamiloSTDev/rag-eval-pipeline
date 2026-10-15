"""Adaptadores falsos: cumplen los puertos sin red, sin modelos y sin costo."""

import math
import re
import zlib

from rag.domain.models import Chunk, LLMResult, RetrievedChunk


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


class FakeEmbedder:
    """Bolsa de palabras con hashing: textos que comparten palabras quedan cerca."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in words(text):
            # crc32 y no hash(): hash() cambia en cada ejecución y los vectores no serían estables
            vector[zlib.crc32(word.encode("utf-8")) % self.dimensions] += 1.0
        return vector


class FakeVectorStore:
    """Diccionario en memoria con búsqueda por similitud coseno."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[Chunk, list[float]]] = {}

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.rows[chunk.chunk_id] = (chunk, vector)

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        results = [
            RetrievedChunk(chunk, cosine(vector, stored)) for chunk, stored in self.rows.values()
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:k]


class FakeKeywordIndex:
    """Puntúa cada chunk con la cantidad de palabras de la consulta que contiene."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        terms = set(words(query))
        results = [
            RetrievedChunk(chunk, float(len(terms & set(words(chunk.text)))))
            for chunk in self.chunks
        ]
        matches = [result for result in results if result.score > 0]
        return sorted(matches, key=lambda result: result.score, reverse=True)[:k]


class FakeReranker:
    """Puntúa cada candidato con la fracción de palabras de la consulta que contiene."""

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        terms = set(words(query))
        rescored = [
            RetrievedChunk(
                candidate.chunk,
                len(terms & set(words(candidate.chunk.text))) / len(terms) if terms else 0.0,
            )
            for candidate in candidates
        ]
        return sorted(rescored, key=lambda result: result.score, reverse=True)[:top_n]


class FakeLLM:
    """Responde siempre lo mismo y anota cada llamada para que los tests la inspeccionen."""

    def __init__(self, reply: str = "Respuesta de prueba [1].") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        return LLMResult(
            text=self.reply,
            tokens_in=len(words(f"{system} {user}")),
            tokens_out=len(words(self.reply)),
        )
