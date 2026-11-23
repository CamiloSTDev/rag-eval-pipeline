"""Lector del golden set: un JSON por línea, validado con Pydantic antes de evaluar nada."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from rag.domain.models import GoldenQuestion, SourceRef


class GoldenSetError(ValueError):
    """El archivo del golden set tiene un error; el mensaje indica la línea."""


class _SourceRow(BaseModel):
    # extra="forbid": un campo mal escrito ("seccion") falla en vez de ignorarse en silencio
    model_config = ConfigDict(extra="forbid")

    source: str
    section: str | None = None
    page: int | None = None


class _QuestionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    reference_answer: str
    relevant_sources: list[_SourceRow] = []
    type: Literal["single", "multi", "unanswerable"]
    holdout: bool = False

    @model_validator(mode="after")
    def sources_match_the_type(self) -> Self:
        if self.type == "unanswerable" and self.relevant_sources:
            raise ValueError("una pregunta sin respuesta no lleva relevant_sources")
        if self.type != "unanswerable" and not self.relevant_sources:
            raise ValueError("falta relevant_sources: dónde está la respuesta")
        return self


def load_golden(path: Path) -> list[GoldenQuestion]:
    if not path.is_file():
        raise GoldenSetError(f"No existe el golden set {path}")
    questions: list[GoldenQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = _QuestionRow.model_validate_json(line)
        except ValidationError as error:
            raise GoldenSetError(f"{path}, línea {number}: {error}") from error
        if row.id in seen:
            raise GoldenSetError(f"{path}, línea {number}: el id {row.id!r} está repetido")
        seen.add(row.id)
        questions.append(
            GoldenQuestion(
                id=row.id,
                question=row.question,
                reference_answer=row.reference_answer,
                relevant_sources=tuple(
                    SourceRef(ref.source, ref.section, ref.page) for ref in row.relevant_sources
                ),
                kind=row.type,
                holdout=row.holdout,
            )
        )
    return questions
