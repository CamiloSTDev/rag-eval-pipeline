"""Formas de los datos que viajan por el pipeline."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


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
    # Ranking completo de la búsqueda; el modelo solo ve los primeros top_k_final
    retrieved: tuple[RetrievedChunk, ...]
    latency_ms: Mapping[str, float]  # por etapa: "retrieve", "generate"
    tokens_in: int
    tokens_out: int


@dataclass(frozen=True)
class SourceRef:
    """Dónde está la respuesta: un archivo y, si se indica, una sección o página exactas."""

    source: str
    section: str | None = None
    page: int | None = None

    def matches(self, chunk: Chunk) -> bool:
        return (
            chunk.source == self.source
            and (self.section is None or chunk.section == self.section)
            and (self.page is None or chunk.page == self.page)
        )


QuestionKind = Literal["single", "multi", "unanswerable"]


@dataclass(frozen=True)
class GoldenQuestion:
    """Una pregunta de referencia, con su respuesta verificada a mano."""

    id: str
    question: str
    reference_answer: str
    relevant_sources: tuple[SourceRef, ...]  # vacío en las preguntas sin respuesta
    kind: QuestionKind
    holdout: bool = False  # las de hold-out solo se miran al final, para no sobreajustar

    @property
    def answerable(self) -> bool:
        return self.kind != "unanswerable"
