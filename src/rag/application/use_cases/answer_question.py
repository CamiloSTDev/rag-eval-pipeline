"""Caso de uso: responder una pregunta con los chunks recuperados y citas verificadas."""

import time

from rag.application.config import RetrievalConfig
from rag.application.ports import Embedder, LLMClient, VectorStore
from rag.application.prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt, parse_llm_answer
from rag.domain.citations import resolve_citations
from rag.domain.models import Answer, RetrievedChunk


class AnswerQuestion:
    def __init__(
        self, embedder: Embedder, store: VectorStore, llm: LLMClient, config: RetrievalConfig
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._config = config

    def __call__(self, question: str) -> Answer:
        start = time.perf_counter()
        retrieved = self._store.search(
            self._embedder.embed_query(question), self._config.top_k_retrieve
        )
        latency = {"retrieve": _ms_since(start)}

        if self._not_enough_context(retrieved):
            # Sin contexto suficiente no se llama al modelo: no se gasta y no hay nada que inventar
            return _refusal(retrieved, latency, tokens_in=0, tokens_out=0)

        sources = [result.chunk for result in retrieved[: self._config.top_k_final]]
        start = time.perf_counter()
        result = self._llm.generate(SYSTEM_PROMPT, build_user_prompt(question, sources))
        latency["generate"] = _ms_since(start)

        parsed = parse_llm_answer(result.text)
        if parsed.refused or parsed.answer.strip() == REFUSAL:
            return _refusal(retrieved, latency, result.tokens_in, result.tokens_out)

        citations, invalid = resolve_citations(parsed.answer, sources)
        return Answer(
            text=parsed.answer,
            citations=tuple(citations),
            invalid_citations=tuple(invalid),
            refused=False,
            retrieved=tuple(retrieved),
            latency_ms=latency,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
        )

    def _not_enough_context(self, retrieved: list[RetrievedChunk]) -> bool:
        threshold = self._config.refusal_threshold
        return not retrieved or (threshold is not None and retrieved[0].score < threshold)


def _refusal(
    retrieved: list[RetrievedChunk], latency: dict[str, float], tokens_in: int, tokens_out: int
) -> Answer:
    return Answer(
        text=REFUSAL,
        citations=(),
        invalid_citations=(),
        refused=True,
        retrieved=tuple(retrieved),
        latency_ms=latency,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000
