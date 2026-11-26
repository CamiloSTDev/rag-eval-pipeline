import json

import pytest

from rag.application.prompts import InvalidLLMOutput
from rag.application.reranking import RERANK_PROMPT, LLMReranker
from rag.domain.models import Chunk, LLMResult, RetrievedChunk
from tests.fakes import FakeLLM


def candidate(chunk_id: str, text: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id, text=text, source="doc.md", page=None, section=None, position=0
    )
    return RetrievedChunk(chunk, score=0.016)


CANDIDATES = [
    candidate("a", "Sub aplicaciones montadas"),
    candidate("b", "Archivos estáticos"),
    candidate("c", "Montar una aplicación Flask con WSGIMiddleware"),
]


def ranking(*numbers: int) -> str:
    return json.dumps({"ranking": list(numbers)})


def test_the_chosen_candidates_go_first_and_the_rest_keep_their_order():
    reranked = LLMReranker(FakeLLM(ranking(3))).rerank("¿Flask?", CANDIDATES, top_n=3).ranking

    assert [r.chunk.chunk_id for r in reranked] == ["c", "a", "b"]
    assert [r.score for r in reranked] == sorted((r.score for r in reranked), reverse=True)


def test_invented_and_repeated_numbers_are_ignored():
    reranked = LLMReranker(FakeLLM(ranking(9, 2, 2, 0))).rerank("?", CANDIDATES, top_n=3).ranking

    assert [r.chunk.chunk_id for r in reranked] == ["b", "a", "c"]


def test_top_n_cuts_the_new_order():
    reranked = LLMReranker(FakeLLM(ranking(3, 1))).rerank("?", CANDIDATES, top_n=1).ranking

    assert [r.chunk.chunk_id for r in reranked] == ["c"]


def test_the_model_sees_the_numbered_candidates_and_the_question():
    llm = FakeLLM(ranking(1))

    LLMReranker(llm).rerank("¿Puedo montar Flask?", CANDIDATES, top_n=3)

    [(system, user)] = llm.calls
    assert system == RERANK_PROMPT and "JSON" in system
    assert '<fragmento n="3">' in user and "WSGIMiddleware" in user
    assert user.endswith("Pregunta: ¿Puedo montar Flask?")


def test_no_candidates_means_no_call():
    llm = FakeLLM()

    reranking = LLMReranker(llm).rerank("?", [], top_n=5)

    assert reranking.ranking == ()
    assert (reranking.tokens_in, reranking.tokens_out) == (0, 0)
    assert llm.calls == []


def test_the_reranking_reports_the_tokens_of_the_model_call():
    class PricedLLM:
        def generate(self, system: str, user: str) -> LLMResult:
            return LLMResult(ranking(3, 1), tokens_in=4000, tokens_out=12)

    reranking = LLMReranker(PricedLLM()).rerank("¿Flask?", CANDIDATES, top_n=3)

    assert (reranking.tokens_in, reranking.tokens_out) == (4000, 12)


def test_an_unreadable_ranking_raises():
    with pytest.raises(InvalidLLMOutput):
        LLMReranker(FakeLLM('{"orden": "c primero"}')).rerank("?", CANDIDATES, top_n=3)
