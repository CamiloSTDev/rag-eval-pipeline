"""Reciprocal Rank Fusion: combina rankings por posición, sin comparar puntajes."""

from collections.abc import Sequence

from rag.domain.models import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Suma 1 / (k + posición) de cada chunk en cada ranking y ordena por esa suma.

    Solo importa la posición, no el puntaje: la similitud coseno (0 a 1) y BM25 (0 a ~20)
    no se pueden sumar, pero "tercero en una lista" significa lo mismo en las dos. Un chunk
    que falta en un ranking simplemente no suma nada de él. k amortigua la ventaja de los
    primeros puestos; 60 es el valor del artículo original. Los empates conservan el orden
    de primera aparición.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for position, result in enumerate(ranking, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
            chunks.setdefault(chunk_id, result.chunk)
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [RetrievedChunk(chunks[chunk_id], scores[chunk_id]) for chunk_id in ordered]
