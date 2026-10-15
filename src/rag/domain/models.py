"""Formas de los datos que viajan por el pipeline."""

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
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
