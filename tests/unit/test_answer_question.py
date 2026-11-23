import json

import pytest

from rag.application.config import RetrievalConfig
from rag.application.prompts import REFUSAL, SYSTEM_PROMPT, InvalidLLMOutput
from rag.application.use_cases.answer_question import AnswerQuestion
from rag.domain.models import Chunk, Citation
from tests.fakes import FakeEmbedder, FakeLLM, FakeVectorStore

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


def make_answer_question(llm: FakeLLM, threshold: float | None = None) -> AnswerQuestion:
    embedder, store = FakeEmbedder(), FakeVectorStore()
    chunks = [QUERY_CHUNK, PATH_CHUNK]
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    config = RetrievalConfig(top_k_final=2, refusal_threshold=threshold)
    return AnswerQuestion(embedder, store, llm, config)


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
    assert set(answer.latency_ms) == {"retrieve", "generate"}
    assert answer.tokens_in > 0 and answer.tokens_out > 0
    [(system, user)] = llm.calls
    assert system == SYSTEM_PROMPT
    assert QUERY_CHUNK.text in user and user.endswith(QUESTION)


def test_the_model_sees_only_the_first_top_k_final_of_the_ranking():
    embedder, store, llm = FakeEmbedder(), FakeVectorStore(), FakeLLM()
    chunks = [QUERY_CHUNK, PATH_CHUNK]
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    config = RetrievalConfig(top_k_retrieve=2, top_k_final=1)

    answer = AnswerQuestion(embedder, store, llm, config)(QUESTION)

    assert len(answer.retrieved) == 2  # el ranking completo, para medir Hit@k
    [(_, user)] = llm.calls
    assert QUERY_CHUNK.text in user and PATH_CHUNK.text not in user


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


def test_empty_index_refuses_without_calling_the_model():
    llm = FakeLLM()
    config = RetrievalConfig(top_k_final=2)

    answer = AnswerQuestion(FakeEmbedder(), FakeVectorStore(), llm, config)(QUESTION)

    assert answer.refused
    assert llm.calls == []


def test_output_that_is_not_the_expected_json_raises():
    with pytest.raises(InvalidLLMOutput):
        make_answer_question(FakeLLM("Son opcionales [1]."))(QUESTION)
