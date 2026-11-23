"""Caso de uso: correr el golden set contra AnswerQuestion y medir cada etapa por separado."""

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag.application.ports import ProviderError
from rag.application.prompts import InvalidLLMOutput
from rag.domain.metrics import hit_at_k, percentile, reciprocal_rank
from rag.domain.models import Answer, GoldenQuestion, QuestionKind


@dataclass(frozen=True)
class QuestionResult:
    id: str
    kind: QuestionKind
    holdout: bool
    # Métricas de recuperación: None en preguntas sin respuesta (no hay fuente que encontrar)
    hit_at_5: int | None
    hit_at_10: int | None
    reciprocal_rank: float | None
    refused: bool
    refusal_correct: bool  # rechazó una sin respuesta, o respondió una con respuesta
    invalid_citations: int
    tokens_in: int
    tokens_out: int
    latency_ms: float
    answer: str
    error: str | None = None  # la respuesta del modelo no se pudo leer


@dataclass(frozen=True)
class Summary:
    questions: int
    errors: int
    hit_at_5: float | None
    hit_at_10: float | None
    mrr: float | None
    refusal_accuracy: float | None
    false_refusals: int  # tenían respuesta y el sistema se negó
    missed_refusals: int  # no tenían respuesta y el sistema respondió igual
    answers_with_invalid_citations: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    tokens_in_mean: float | None
    tokens_out_mean: float | None


class Evaluate:
    def __init__(self, answer_question: Callable[[str], Answer]) -> None:
        self._answer_question = answer_question

    def __call__(
        self,
        questions: Sequence[GoldenQuestion],
        on_result: Callable[[QuestionResult], None] | None = None,
    ) -> list[QuestionResult]:
        """`on_result` recibe cada resultado apenas está listo: una corrida tarda minutos."""
        results: list[QuestionResult] = []
        for question in questions:
            result = self._evaluate(question)
            if on_result is not None:
                on_result(result)
            results.append(result)
        return results

    def _evaluate(self, question: GoldenQuestion) -> QuestionResult:
        try:
            answer = self._answer_question(question.question)
        except (InvalidLLMOutput, ProviderError) as error:
            # Una respuesta ilegible o un corte del proveedor no deben tirar abajo la corrida
            return _failed(question, str(error))

        relevant = [
            any(ref.matches(result.chunk) for ref in question.relevant_sources)
            for result in answer.retrieved
        ]
        measurable = question.answerable
        return QuestionResult(
            id=question.id,
            kind=question.kind,
            holdout=question.holdout,
            hit_at_5=hit_at_k(relevant, 5) if measurable else None,
            hit_at_10=hit_at_k(relevant, 10) if measurable else None,
            reciprocal_rank=reciprocal_rank(relevant) if measurable else None,
            refused=answer.refused,
            refusal_correct=answer.refused != question.answerable,
            invalid_citations=len(answer.invalid_citations),
            tokens_in=answer.tokens_in,
            tokens_out=answer.tokens_out,
            latency_ms=sum(answer.latency_ms.values()),
            answer=answer.text,
        )


def summarize(results: Sequence[QuestionResult]) -> Summary:
    """Promedios de la corrida. Las preguntas con error se cuentan, pero no entran en ninguna
    métrica."""
    ok = [result for result in results if result.error is None]
    retrieval = [result for result in ok if result.reciprocal_rank is not None]
    latencies = [result.latency_ms for result in ok]
    return Summary(
        questions=len(results),
        errors=len(results) - len(ok),
        hit_at_5=_mean([r.hit_at_5 for r in retrieval if r.hit_at_5 is not None]),
        hit_at_10=_mean([r.hit_at_10 for r in retrieval if r.hit_at_10 is not None]),
        mrr=_mean([r.reciprocal_rank for r in retrieval if r.reciprocal_rank is not None]),
        refusal_accuracy=_mean([float(r.refusal_correct) for r in ok]),
        false_refusals=sum(r.refused and r.kind != "unanswerable" for r in ok),
        missed_refusals=sum(not r.refused and r.kind == "unanswerable" for r in ok),
        answers_with_invalid_citations=sum(r.invalid_citations > 0 for r in ok),
        latency_p50_ms=percentile(latencies, 50) if latencies else None,
        latency_p95_ms=percentile(latencies, 95) if latencies else None,
        tokens_in_mean=_mean([float(r.tokens_in) for r in ok]),
        tokens_out_mean=_mean([float(r.tokens_out) for r in ok]),
    )


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _failed(question: GoldenQuestion, error: str) -> QuestionResult:
    return QuestionResult(
        id=question.id,
        kind=question.kind,
        holdout=question.holdout,
        hit_at_5=None,
        hit_at_10=None,
        reciprocal_rank=None,
        refused=False,
        refusal_correct=False,
        invalid_citations=0,
        tokens_in=0,
        tokens_out=0,
        latency_ms=0.0,
        answer="",
        error=error,
    )
