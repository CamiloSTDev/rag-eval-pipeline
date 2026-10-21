"""Formas de los datos que viajan por el pipeline."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    text: str
    source: str  # ruta del archivo dentro del corpus, o URL
    title: str | None = None
    page: int | None = None  # solo en formatos con páginas (PDF)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # hash de (source, page, position, text): hace la ingesta idempotente
    text: str
    source: str
    page: int | None
    section: str | None  # ruta de encabezados, p. ej. "Guía > Instalación"
    position: int  # orden dentro del documento


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # mayor = más relevante; la escala depende de quién lo recuperó


@dataclass(frozen=True)
class Citation:
    n: int  # el número [n] que aparece en el texto de la respuesta
    chunk_id: str
    source: str
    page: int | None
    section: str | None


@dataclass(frozen=True)
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    invalid_citations: tuple[int, ...]  # números [n] sin fuente detrás: señal de invención
    refused: bool
    retrieved: tuple[RetrievedChunk, ...]  # el contexto que vio el modelo, para evaluar
    latency_ms: Mapping[str, float]  # por etapa: "retrieve", "generate"
    tokens_in: int
    tokens_out: int
