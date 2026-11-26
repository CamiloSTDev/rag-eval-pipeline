import json
from typing import Any

import httpx2
from fastapi.testclient import TestClient

from rag.application.config import RetrievalConfig
from rag.application.ports import ProviderError
from rag.application.retrieval import Retriever
from rag.application.use_cases.answer_question import AnswerQuestion
from rag.application.use_cases.ingest_corpus import IngestReport
from rag.domain.models import Chunk, LLMResult
from rag.interfaces.api import ApiServices, create_app
from tests.fakes import FakeEmbedder, FakeLLM, FakeVectorStore

CHUNK = Chunk(
    chunk_id="query",
    text="Los parámetros de query son opcionales si tienen un valor por defecto.",
    source="tutorial/query-params.md",
    page=None,
    section="Parámetros de Query",
    position=0,
)
GOOD_REPLY = json.dumps(
    {"answer": "Son opcionales si tienen un valor por defecto [1].", "refused": False}
)


class BrokenLLM:
    def generate(self, system: str, user: str) -> LLMResult:
        raise ProviderError("El proveedor no responde")


def make_services(
    llm: FakeLLM | BrokenLLM | None = None, indexed: bool = True, admin_key: str | None = None
) -> ApiServices:
    embedder, store = FakeEmbedder(), FakeVectorStore()
    if indexed:
        store.add([CHUNK], embedder.embed_documents([CHUNK.text]))
    answer_question = AnswerQuestion(
        Retriever(embedder, store), llm or FakeLLM(GOOD_REPLY), RetrievalConfig()
    )
    return ApiServices(
        answer_question=answer_question,
        ingest=lambda: IngestReport(documents=123, chunks=1117),
        indexed_chunks=lambda: len(store.rows),
        config={"embedding_model": "falso", "chunk_tokens": 512},
        corpus_version="0123456789ab",
        admin_key=admin_key,
    )


def call(services: ApiServices, method: str, path: str, **kwargs: Any) -> httpx2.Response:
    # El with arranca el lifespan de la app, como haría el servidor
    with TestClient(create_app(services)) as client:
        return client.request(method, path, **kwargs)


def test_ask_returns_the_answer_with_its_citations():
    response = call(make_services(), "POST", "/ask", json={"question": "¿Son opcionales?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Son opcionales si tienen un valor por defecto [1]."
    assert body["citations"] == [
        {
            "n": 1,
            "source": "tutorial/query-params.md",
            "page": None,
            "section": "Parámetros de Query",
            "chunk_id": "query",
        }
    ]
    assert body["refused"] is False
    assert set(body["latency_ms"]) == {"retrieve", "generate"}


def test_empty_or_too_long_questions_are_rejected():
    services = make_services()

    for question in ["", "   ", "x" * 1001]:
        response = call(services, "POST", "/ask", json={"question": question})
        assert response.status_code == 422, question


def test_ask_with_an_empty_index_is_503():
    response = call(make_services(indexed=False), "POST", "/ask", json={"question": "¿Hola?"})

    assert response.status_code == 503


def test_failures_of_the_model_provider_are_502():
    unreadable = make_services(llm=FakeLLM("esto no es JSON"))
    down = make_services(llm=BrokenLLM())

    for services in [unreadable, down]:
        response = call(services, "POST", "/ask", json={"question": "¿Son opcionales?"})
        assert response.status_code == 502


def test_health_reports_whether_the_index_can_answer():
    ready = call(make_services(), "GET", "/health")
    empty = call(make_services(indexed=False), "GET", "/health")

    assert (ready.status_code, ready.json()) == (200, {"status": "ok", "indexed_chunks": 1})
    assert (empty.status_code, empty.json()["status"]) == (503, "index_empty")


def test_ingest_needs_the_admin_key():
    disabled = make_services(admin_key=None)
    protected = make_services(admin_key="secreta")

    assert call(disabled, "POST", "/ingest", headers={"X-Admin-Key": "x"}).status_code == 403
    assert call(protected, "POST", "/ingest").status_code == 403
    assert call(protected, "POST", "/ingest", headers={"X-Admin-Key": "otra"}).status_code == 403
    accepted = call(protected, "POST", "/ingest", headers={"X-Admin-Key": "secreta"})
    assert (accepted.status_code, accepted.json()) == (200, {"documents": 123, "chunks": 1117})


def test_version_reports_the_corpus_the_config_and_a_stable_hash():
    first = call(make_services(), "GET", "/version").json()
    second = call(make_services(), "GET", "/version").json()

    assert first["config"] == {"embedding_model": "falso", "chunk_tokens": 512}
    assert len(first["config_hash"]) == 12
    assert first["config_hash"] == second["config_hash"]
    assert first["version"] == "0.1.0"
    assert first["corpus_version"] == "0123456789ab"
