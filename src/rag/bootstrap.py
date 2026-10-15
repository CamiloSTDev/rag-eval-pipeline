"""Único lugar que conoce las clases concretas: crea los adaptadores y arma los casos de uso."""

from openai import OpenAI

from rag.application.config import ChunkingConfig
from rag.application.use_cases.ingest_corpus import IngestCorpus
from rag.application.use_cases.search_chunks import SearchChunks
from rag.domain.models import Document
from rag.infrastructure.embeddings.openai_embed import OpenAIEmbedder, token_counter
from rag.infrastructure.loaders import load_corpus
from rag.infrastructure.stores.chroma_store import ChromaStore
from rag.settings import Settings


class ConfigurationError(Exception):
    """Falta o sobra algo en la configuración; el mensaje dice cómo arreglarlo."""


def load_documents(settings: Settings) -> list[Document]:
    return load_corpus(settings.corpus_path, exclude=settings.corpus_exclude)


def build_ingest_corpus(settings: Settings) -> IngestCorpus:
    config = ChunkingConfig(max_tokens=settings.chunk_tokens, overlap_tokens=settings.chunk_overlap)
    return IngestCorpus(
        _embedder(settings), _store(settings), token_counter(settings.embedding_model), config
    )


def build_search_chunks(settings: Settings) -> SearchChunks:
    return SearchChunks(_embedder(settings), _store(settings))


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


def _store(settings: Settings) -> ChromaStore:
    return ChromaStore(settings.index_path, collection_name(settings))
