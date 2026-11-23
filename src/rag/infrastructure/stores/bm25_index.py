"""Índice por palabras en memoria con BM25 (rank_bm25)."""

import heapq
import re
import unicodedata

from rank_bm25 import BM25Okapi

from rag.domain.models import Chunk, RetrievedChunk

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Palabras en minúsculas y sin tildes: "Parámetro" y "parametro" son la misma.

    `\\w+` conserva los guiones bajos, así un identificador como include_in_schema es una
    sola palabra: justo lo que la búsqueda vectorial tiende a difuminar.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    return _WORD.findall("".join(char for char in decomposed if not unicodedata.combining(char)))


class Bm25Index:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        # La sección se indexa con el texto: los títulos concentran los términos clave
        corpus = [tokenize(f"{chunk.section or ''}\n{chunk.text}") for chunk in self._chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None  # BM25Okapi falla sin documentos

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        best = heapq.nlargest(k, range(len(self._chunks)), key=lambda i: scores[i])
        # Puntaje 0 = ninguna palabra en común: no es un resultado, aunque quepa en los k
        return [RetrievedChunk(self._chunks[i], float(scores[i])) for i in best if scores[i] > 0]
