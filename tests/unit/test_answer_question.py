import json

import pytest

from rag.application.config import RetrievalConfig
from rag.application.prompts import REFUSAL, SYSTEM_PROMPT, InvalidLLMOutput
from rag.application.retrieval import Retriever
from rag.application.use_cases.answer_question import AnswerQuestion
from rag.domain.models import Chunk, Citation
from tests.fakes import FakeEmbedder, FakeKeywordIndex, FakeLLM, FakeVectorStore

QUERY_CHUNK = Chunk(
    chunk_id="query",
    text="Los parámetros de query son opcionales si tienen un valor por defecto.",
    source="tutorial/query-params.md",
    page=None,
    section="Parámetros de Query",
    position=0,
)
PATH_CHUNK = Chunk(
    chunk_id="path",
    text="Los parámetros de path son obligatorios.",
    source="tutorial/path-params.md",
    page=None,
    section="Parámetros de Path",
    position=0,
)
QUESTION = "¿Los parámetros de query son opcionales?"


def reply(answer: str, refused: bool = False) -> str:
    return json.dumps({"answer": answer, "refused": refused})


def make_retriever(chunks: list[Chunk], hybrid: bool = False) -> Retriever:
    embedder, store = FakeEmbedder(), FakeVectorStore()
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    keywords = None
    if hybrid:
        keywords = FakeKeywordIndex()
        keywords.build(chunks)
    return Retriever(embedder, store, keywords)


def make_answer_question(
    llm: FakeLLM, threshold: float | None = None, hybrid: bool = False
) -> AnswerQuestion:
    config = RetrievalConfig(top_k_final=2, refusal_threshold=threshold)
    return AnswerQuestion(make_retriever([QUERY_CHUNK, PATH_CHUNK], hybrid), llm, config)


def test_answer_cites_the_retrieved_sources_and_reports_its_cost():
    llm = FakeLLM(reply("Son opcionales si tienen valor por defecto [1]."))

    answer = make_answer_question(llm)(QUESTION)

    assert answer.text == "Son opcionales si tienen valor por defecto [1]."
    assert answer.citations == (
        Citation(
            n=1,
            chunk_id="query",
            source="tutorial/query-params.md",
            page=None,
            section="Parámetros de Query",
        ),
    )
    assert answer.invalid_citations == ()
    assert not answer.refused
    assert [result.chunk for result in answer.retrieved] == [QUERY_CHUNK, PATH_CHUNK]
    assert answer.top_similarity == answer.retrieved[0].score
    assert set(answer.latency_ms) == {"retrieve", "generate"}
    assert answer.tokens_in > 0 and answer.tokens_out > 0
    [(system, user)] = llm.calls
    assert system == SYSTEM_PROMPT
    assert QUERY_CHUNK.text in user and user.endswith(QUESTION)


def test_the_model_sees_only_the_first_top_k_final_of_the_ranking():
    llm = FakeLLM()
    config = RetrievalConfig(top_k_retrieve=2, top_k_final=1)

    answer = AnswerQuestion(make_retriever([QUERY_CHUNK, PATH_CHUNK]), llm, config)(QUESTION)

    assert len(answer.retrieved) == 2  # el ranking completo, para medir Hit@k
    [(_, user)] = llm.calls
    assert QUERY_CHUNK.text in user and PATH_CHUNK.text not in user


def test_the_reranker_decides_which_sources_the_model_sees():
    class PreferPath:
        def rerank(self, query, candidates, top_n):
            return sorted(candidates, key=lambda r: r.chunk.chunk_id != "path")[:top_n]

    llm = FakeLLM()
    config = RetrievalConfig(top_k_final=1)
    answer_question = AnswerQuestion(
        make_retriever([QUERY_CHUNK, PATH_CHUNK]), llm, config, reranker=PreferPath()
    )

    answer = answer_question(QUESTION)

    assert [r.chunk for r in answer.retrieved] == [PATH_CHUNK, QUERY_CHUNK]
    assert answer.context == (PATH_CHUNK,)
    assert "rerank" in answer.latency_ms


def test_citations_without_a_source_are_reported():
    llm = FakeLLM(reply("Son opcionales [1], según un dato inventado [9]."))

    answer = make_answer_question(llm)(QUESTION)

    assert [citation.n for citation in answer.citations] == [1]
    assert answer.invalid_citations == (9,)


def test_model_refusal_returns_the_refusal_sentence_without_citations():
    llm = FakeLLM(reply("No sé, pero quizás [1].", refused=True))

    answer = make_answer_question(llm)(QUESTION)

    assert answer.refused
    assert answer.text == REFUSAL
    assert answer.citations == ()


def test_low_retrieval_score_refuses_without_calling_the_model():
    llm = FakeLLM()

    answer = make_answer_question(llm, threshold=0.99)("¿Cómo cambio la contraseña?")

    assert answer.refused
    assert llm.calls == []
    assert answer.tokens_in == 0
    assert set(answer.latency_ms) == {"retrieve"}


def test_with_hybrid_search_the_threshold_still_uses_the_vector_similarity():
    # Los puntajes de RRF no pasan de 2/61 ≈ 0,033: compararlos con 0,3 rechazaría todo
    llm = FakeLLM()

    answer = make_answer_question(llm, threshold=0.3, hybrid=True)(QUESTION)

    assert answer.retrieved[0].score < 0.05  # el ranking es el fusionado
    assert answer.top_similarity is not None and answer.top_similarity > 0.3
    assert not answer.refused
    assert len(llm.calls) == 1


def test_empty_index_refuses_without_calling_the_model():
    llm = FakeLLM()
    config = RetrievalConfig(top_k_final=2)

    answer = AnswerQuestion(make_retriever([]), llm, config)(QUESTION)

    assert answer.refused
    assert llm.calls == []


def test_output_that_is_not_the_expected_json_raises():
    with pytest.raises(InvalidLLMOutput):
        make_answer_question(FakeLLM("Son opcionales [1]."))(QUESTION)
