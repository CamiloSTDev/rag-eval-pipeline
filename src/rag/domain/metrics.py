"""Métricas de evaluación: funciones puras sobre resultados ya calculados."""

import math


def hit_at_k(relevant: list[bool], k: int) -> int:
    """1 si alguno de los primeros k recuperados es relevante; 0 si ninguno.

    `relevant[i]` dice si el chunk en la posición i+1 del ranking es relevante.
    """
    return int(any(relevant[:k]))


def reciprocal_rank(relevant: list[bool]) -> float:
    """1 / posición del primer relevante (1, 0,5, 0,33...), o 0 si no hay ninguno.

    Su promedio sobre todas las preguntas es el MRR (Mean Reciprocal Rank).
    """
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: list[float], q: float) -> float:
    """El valor bajo el cual queda el q % de las mediciones (método del rango más cercano)."""
    if not values:
        raise ValueError("No hay valores para calcular un percentil")
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]
