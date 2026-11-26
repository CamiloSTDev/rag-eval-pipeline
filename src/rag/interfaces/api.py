"""API HTTP: una capa delgada sobre los casos de uso, que recibe ya construidos.

Para correrla: uv run uvicorn rag.interfaces.api:app
"""

import hashlib
import json
import secrets
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version as package_version
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from rag.application.ports import ProviderError
from rag.application.prompts import InvalidLLMOutput
from rag.application.use_cases.answer_question import AnswerQuestion, RerankerUnavailable
from rag.application.use_cases.ingest_corpus import IngestReport
from rag.domain.corpus import corpus_version


@dataclass
class ApiServices:
    """Lo que la API necesita, ya armado. En producción lo crea bootstrap; en tests, falsos."""

    answer_question: AnswerQuestion
    ingest: Callable[[], IngestReport]
    indexed_chunks: Callable[[], int]
    config: Mapping[str, object]
    corpus_version: str
    admin_key: str | None


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    # Límite de largo: una pregunta enorme solo sirve para gastar tokens o inyectar instrucciones
    question: str = Field(min_length=1, max_length=1000)
    # Opcionales: cambian solo esta pregunta. Más de 10 fuentes sube el costo y el ruido (E2)
    top_k: int | None = Field(default=None, ge=1, le=10)
    use_reranker: bool | None = None


class CitationOut(BaseModel):
    n: int
    source: str
    page: int | None
    section: str | None
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    invalid_citations: list[int]
    refused: bool
    top_similarity: float | None
    latency_ms: dict[str, float]
    tokens_in: int
    tokens_out: int
    rerank_tokens_in: int
    rerank_tokens_out: int


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int


class VersionResponse(BaseModel):
    version: str
    corpus_version: str
    config: dict[str, object]
    config_hash: str


def get_services(request: Request) -> ApiServices:
    services: ApiServices = request.app.state.services
    return services


Services = Annotated[ApiServices, Depends(get_services)]
router = APIRouter()


# Endpoints con def y no async def: el trabajo es bloqueante (red, Chroma) y FastAPI los corre
# en un pool de hilos, así una consulta lenta no frena a las demás.
@router.post("/ask")
def ask(body: AskRequest, services: Services) -> AskResponse:
    _require_index(services)
    try:
        answer = services.answer_question(
            body.question, top_k=body.top_k, use_reranker=body.use_reranker
        )
    except RerankerUnavailable as error:
        raise HTTPException(422, str(error)) from error
    except (InvalidLLMOutput, ProviderError) as error:
        # 502 Bad Gateway: el error no es de la API ni del cliente, sino del servicio de atrás
        raise HTTPException(502, str(error)) from error
    return AskResponse(
        answer=answer.text,
        citations=[CitationOut(**vars(citation)) for citation in answer.citations],
        invalid_citations=list(answer.invalid_citations),
        refused=answer.refused,
        top_similarity=answer.top_similarity,
        latency_ms=dict(answer.latency_ms),
        tokens_in=answer.tokens_in,
        tokens_out=answer.tokens_out,
        rerank_tokens_in=answer.rerank_tokens_in,
        rerank_tokens_out=answer.rerank_tokens_out,
    )


@router.post("/ingest")
def ingest(
    services: Services, x_admin_key: Annotated[str | None, Header()] = None
) -> IngestResponse:
    if services.admin_key is None:
        raise HTTPException(403, "La ingesta por API está desactivada: configura ADMIN_API_KEY.")
    # compare_digest tarda lo mismo acierte o no: no deja adivinar la clave midiendo tiempos
    if x_admin_key is None or not secrets.compare_digest(
        x_admin_key.encode(), services.admin_key.encode()
    ):
        raise HTTPException(403, "Clave de administrador inválida.")
    try:
        report = services.ingest()
    except ProviderError as error:
        raise HTTPException(502, str(error)) from error
    return IngestResponse(documents=report.documents, chunks=report.chunks)


@router.get("/health")
def health(services: Services, response: Response) -> HealthResponse:
    chunks = services.indexed_chunks()
    if chunks == 0:
        response.status_code = 503  # la API responde, pero todavía no puede contestar preguntas
        return HealthResponse(status="index_empty", indexed_chunks=0)
    return HealthResponse(status="ok", indexed_chunks=chunks)


@router.get("/version")
def version(services: Services) -> VersionResponse:
    config = dict(services.config)
    canonical = json.dumps(config, sort_keys=True, default=str)
    return VersionResponse(
        version=package_version("rag"),
        corpus_version=services.corpus_version,
        config=config,
        config_hash=hashlib.sha256(canonical.encode()).hexdigest()[:12],
    )


def _require_index(services: ApiServices) -> None:
    if services.indexed_chunks() == 0:
        raise HTTPException(503, "El índice está vacío: corre primero la ingesta.")


def create_app(services: ApiServices | None = None) -> FastAPI:
    """Con `services`, la app usa esos (tests). Sin ellos, los arma al arrancar (producción)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Embedder, Chroma y BM25 se cargan una sola vez al arrancar, no en cada petición
        app.state.services = services or _production_services()
        yield

    app = FastAPI(title="RAG con evaluación", lifespan=lifespan)
    app.include_router(router)
    return app


def _production_services() -> ApiServices:
    from rag import bootstrap  # aquí y no arriba: los tests importan la API sin configuración
    from rag.settings import Settings

    settings = Settings()
    ingest_corpus = bootstrap.build_ingest_corpus(settings)

    def ingest_and_reload() -> IngestReport:
        documents = bootstrap.load_documents(settings)
        report = ingest_corpus(documents)
        # BM25 vive en memoria: se rearma para que también vea los chunks recién ingeridos
        services.answer_question = bootstrap.build_answer_question(settings)
        services.corpus_version = corpus_version(documents)
        return report

    services = ApiServices(
        answer_question=bootstrap.build_answer_question(settings),
        ingest=ingest_and_reload,
        indexed_chunks=bootstrap.build_index_size(settings),
        config=bootstrap.run_config(settings),
        corpus_version=bootstrap.current_corpus_version(settings),
        admin_key=settings.admin_api_key.get_secret_value() if settings.admin_api_key else None,
    )
    return services


app = create_app()
