"""Caso de uso: responder una pregunta con los chunks recuperados y citas verificadas."""

import time

from rag.application.config import RetrievalConfig
from rag.application.ports import LLMClient, Reranker
from rag.application.prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt, parse_llm_answer
from rag.application.retrieval import Retrieval, Retriever
from rag.domain.citations import resolve_citations
from rag.domain.models import Answer, Chunk, LLMResult, Reranking, RetrievedChunk


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
            return _refusal(retrieval.ranking, retrieval, [], latency, None, None)

        ranking = retrieval.ranking
        reranking = None
        if self._reranker is not None:
            # Se reordenan solo los primeros candidatos: leer los 30 costaría más y suma poco
            start = time.perf_counter()
            head = ranking[: self._config.rerank_candidates]
            reranking = self._reranker.rerank(question, head, top_n=len(head))
            ranking = list(reranking.ranking) + ranking[len(head) :]
            latency["rerank"] = _ms_since(start)

        sources = [result.chunk for result in ranking[: self._config.top_k_final]]
        start = time.perf_counter()
        generation = self._llm.generate(SYSTEM_PROMPT, build_user_prompt(question, sources))
        latency["generate"] = _ms_since(start)

        parsed = parse_llm_answer(generation.text)
        if parsed.refused or parsed.answer.strip() == REFUSAL:
            return _refusal(ranking, retrieval, sources, latency, generation, reranking)

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
            **_usage(generation, reranking),
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
    generation: LLMResult | None,
    reranking: Reranking | None,
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
        **_usage(generation, reranking),
    )


def _usage(generation: LLMResult | None, reranking: Reranking | None) -> dict[str, int]:
    """Tokens por etapa; una etapa que no corrió (rechazo previo, sin reranker) cuenta 0."""
    return {
        "tokens_in": generation.tokens_in if generation else 0,
        "tokens_out": generation.tokens_out if generation else 0,
        "rerank_tokens_in": reranking.tokens_in if reranking else 0,
        "rerank_tokens_out": reranking.tokens_out if reranking else 0,
    }


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000
