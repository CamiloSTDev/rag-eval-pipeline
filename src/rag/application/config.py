"""Parámetros de los casos de uso. bootstrap.py los llena desde las variables de entorno."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 512
    overlap_tokens: int = 64
