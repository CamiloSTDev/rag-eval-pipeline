"""Parámetros de los casos de uso. bootstrap.py los llena desde las variables de entorno."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 512
    overlap_tokens: int = 64


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_final: int = 5  # chunks que recibe el modelo como contexto
    # Si el mejor puntaje queda por debajo, se rechaza sin llamar al modelo.
    # None = sin umbral, hasta calibrarlo con preguntas reales.
    refusal_threshold: float | None = None
