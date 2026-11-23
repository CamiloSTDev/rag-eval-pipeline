"""Único lugar que conoce las clases concretas: crea los adaptadores y arma los casos de uso."""

from pathlib import Path

from openai import OpenAI

from rag.application.config import ChunkingConfig, RetrievalConfig
from rag.application.retrieval import Retriever
from rag.application.use_cases.answer_question import AnswerQuestion
from rag.application.use_cases.evaluate import Evaluate
from rag.application.use_cases.ingest_corpus import IngestCorpus
from rag.application.use_cases.search_chunks import SearchChunks, SearchMode
from rag.domain.chunking import chunk_document
from rag.domain.models import Document, GoldenQuestion
from rag.infrastructure.embeddings.openai_embed import OpenAIEmbedder, token_counter
from rag.infrastructure.golden import GoldenSetError as GoldenSetError
from rag.infrastructure.golden import load_golden
from rag.infrastructure.llm.openai_compatible import OpenAICompatibleLLM
from rag.infrastructure.loaders import load_corpus
from rag.infrastructure.stores.bm25_index import Bm25Index
from rag.infrastructure.stores.chroma_store import ChromaStore
from rag.settings import Settings

# base_url por proveedor; None = el servidor por defecto del SDK (OpenAI)
_LLM_BASE_URLS = {"openai": None, "deepseek": "https://api.deepseek.com"}


class ConfigurationError(Exception):
    """Falta o sobra algo en la configuración; el mensaje dice cómo arreglarlo."""


def load_documents(settings: Settings) -> list[Document]:
    return load_corpus(settings.corpus_path, exclude=settings.corpus_exclude)


def build_ingest_corpus(settings: Settings) -> IngestCorpus:
    config = ChunkingConfig(max_tokens=settings.chunk_tokens, overlap_tokens=settings.chunk_overlap)
    return IngestCorpus(
        _embedder(settings), _store(settings), token_counter(settings.embedding_model), config
    )


def build_search_chunks(settings: Settings, mode: SearchMode = "vector") -> SearchChunks:
    # El índice BM25 solo se arma si el modo lo usa: cuesta leer todos los chunks de Chroma
    return SearchChunks(_retriever(settings, hybrid=mode != "vector"))


def build_answer_question(settings: Settings) -> AnswerQuestion:
    config = RetrievalConfig(
        top_k_retrieve=settings.top_k_retrieve,
        top_k_final=settings.top_k_final,
        refusal_threshold=settings.refusal_threshold,
    )
    return AnswerQuestion(_retriever(settings, hybrid=settings.use_bm25), _llm(settings), config)


def build_evaluate(settings: Settings) -> Evaluate:
    return Evaluate(build_answer_question(settings))


def load_golden_set(settings: Settings, path: Path | None = None) -> list[GoldenQuestion]:
    """Lee el golden set y lo contrasta con el corpus limpio, tal como lo ve el sistema.

    Cada fuente debe ser un documento del corpus, cada sección debe existir tal cual y cada
    cita (`evidence`) debe aparecer literalmente en esa sección. Un error de tipeo en
    cualquiera de las tres daría Hit@k = 0 sin avisar.
    """
    questions = load_golden(path or settings.golden_path)
    documents = {document.source: document for document in load_documents(settings)}
    sections: dict[str, dict[str | None, str]] = {}  # source -> sección -> texto
    problems: list[str] = []
    for question in questions:
        for ref in question.relevant_sources:
            document = documents.get(ref.source)
            if document is None:
                problems.append(f"{question.id}: {ref.source} no es un documento del corpus")
                continue
            if ref.source not in sections:
                sections[ref.source] = _section_texts(document)
            if ref.section is not None and ref.section not in sections[ref.source]:
                problems.append(f"{question.id}: no existe la sección {ref.section!r}")
                continue
            if ref.evidence is not None:
                scope = (
                    sections[ref.source][ref.section] if ref.section is not None else document.text
                )
                if _squeeze(ref.evidence) not in _squeeze(scope):
                    problems.append(
                        f"{question.id}: la cita no está en {ref.section or ref.source}"
                    )
    if problems:
        raise GoldenSetError("El golden set no coincide con el corpus:\n- " + "\n- ".join(problems))
    return questions


def run_config(settings: Settings) -> dict[str, object]:
    """La configuración que define una corrida, para guardarla junto a sus resultados.

    Se eligen los campos uno por uno: las claves de API nunca deben terminar en results/.
    """
    return {
        "collection": collection_name(settings),
        "embedding_model": settings.embedding_model,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap": settings.chunk_overlap,
        "top_k_retrieve": settings.top_k_retrieve,
        "top_k_final": settings.top_k_final,
        "use_bm25": settings.use_bm25,
        "rrf_k": settings.rrf_k,
        "refusal_threshold": settings.refusal_threshold,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }


def collection_name(settings: Settings) -> str:
    # Una colección por modelo y chunking: cambiar cualquiera no mezcla chunks viejos y nuevos
    return f"{settings.embedding_model}-{settings.chunk_tokens}-{settings.chunk_overlap}"


def _embedder(settings: Settings) -> OpenAIEmbedder:
    if settings.openai_api_key is None:
        raise ConfigurationError(
            "Falta OPENAI_API_KEY. Copia .env.example como .env y pega tu clave de OpenAI."
        )
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    return OpenAIEmbedder(client, settings.embedding_model)


def _section_texts(document: Document) -> dict[str | None, str]:
    """Texto de cada sección del documento. Las secciones no dependen del tamaño de chunk,
    así que un límite enorme y un contador de palabras bastan (y son rápidos)."""
    texts: dict[str | None, list[str]] = {}
    for chunk in chunk_document(
        document, max_tokens=1_000_000, overlap_tokens=0, count_tokens=lambda t: len(t.split())
    ):
        texts.setdefault(chunk.section, []).append(chunk.text)
    return {section: "\n\n".join(parts) for section, parts in texts.items()}


def _squeeze(text: str) -> str:
    return " ".join(text.split())


def _store(settings: Settings) -> ChromaStore:
    return ChromaStore(settings.index_path, collection_name(settings))


def _retriever(settings: Settings, hybrid: bool) -> Retriever:
    embedder, store = _embedder(settings), _store(settings)
    keywords = None
    if hybrid:
        # BM25 vive en memoria: se rearma al arrancar con los mismos chunks que tiene Chroma
        keywords = Bm25Index()
        keywords.build(store.all_chunks())
    return Retriever(embedder, store, keywords, rrf_k=settings.rrf_k)


def _llm(settings: Settings) -> OpenAICompatibleLLM:
    if settings.llm_model is None:
        raise ConfigurationError(
            "Falta LLM_MODEL en .env: el nombre exacto del modelo, según la documentación "
            "de tu proveedor."
        )
    if settings.llm_provider == "openai":
        key, variable = settings.openai_api_key, "OPENAI_API_KEY"
    else:
        key, variable = settings.deepseek_api_key, "DEEPSEEK_API_KEY"
    if key is None:
        raise ConfigurationError(f"Falta {variable} en .env para usar {settings.llm_provider}.")
    base_url = settings.llm_base_url or _LLM_BASE_URLS[settings.llm_provider]
    client = OpenAI(api_key=key.get_secret_value(), base_url=base_url)
    return OpenAICompatibleLLM(client, settings.llm_model)
