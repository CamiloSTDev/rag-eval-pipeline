"""Caso de uso: responder una pregunta con los chunks recuperados y citas verificadas."""

import time

from rag.application.config import RetrievalConfig
from rag.application.ports import LLMClient, Reranker
from rag.application.prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt, parse_llm_answer
from rag.application.retrieval import Retrieval, Retriever
from rag.domain.citations import resolve_citations
from rag.domain.models import Answer, Chunk, RetrievedChunk


class AnswerQuestion:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        config: RetrievalConfig,
        reranker: Reranker | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._config = config
        self._reranker = reranker

    def __call__(self, question: str) -> Answer:
        start = time.perf_counter()
        retrieval = self._retriever(question, self._config.top_k_retrieve)
        latency = {"retrieve": _ms_since(start)}

        if self._not_enough_context(retrieval):
            # Sin contexto suficiente no se llama al modelo: no se gasta y no hay nada que inventar
            return _refusal(retrieval.ranking, retrieval, [], latency, tokens_in=0, tokens_out=0)

        ranking = retrieval.ranking
        if self._reranker is not None:
            # Se reordenan solo los primeros candidatos: leer los 30 costaría más y suma poco
            start = time.perf_counter()
            head = ranking[: self._config.rerank_candidates]
            ranking = self._reranker.rerank(question, head, top_n=len(head)) + ranking[len(head) :]
            latency["rerank"] = _ms_since(start)

        sources = [result.chunk for result in ranking[: self._config.top_k_final]]
        start = time.perf_counter()
        result = self._llm.generate(SYSTEM_PROMPT, build_user_prompt(question, sources))
        latency["generate"] = _ms_since(start)

        parsed = parse_llm_answer(result.text)
        if parsed.refused or parsed.answer.strip() == REFUSAL:
            return _refusal(
                ranking, retrieval, sources, latency, result.tokens_in, result.tokens_out
            )

        citations, invalid = resolve_citations(parsed.answer, sources)
        return Answer(
            text=parsed.answer,
            citations=tuple(citations),
            invalid_citations=tuple(invalid),
            refused=False,
            retrieved=tuple(ranking),
            context=tuple(sources),
            top_similarity=retrieval.top_similarity,
            latency_ms=latency,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
        )

    def _not_enough_context(self, retrieval: Retrieval) -> bool:
        threshold = self._config.refusal_threshold
        best = retrieval.top_similarity
        return not retrieval.ranking or (
            threshold is not None and (best is None or best < threshold)
        )


def _refusal(
    ranking: list[RetrievedChunk],
    retrieval: Retrieval,
    context: list[Chunk],
    latency: dict[str, float],
    tokens_in: int,
    tokens_out: int,
) -> Answer:
    return Answer(
        text=REFUSAL,
        citations=(),
        invalid_citations=(),
        refused=True,
        retrieved=tuple(ranking),
        context=tuple(context),
        top_similarity=retrieval.top_similarity,
        latency_ms=latency,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000
