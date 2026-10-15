from rag.application.use_cases.search_chunks import SearchChunks
from rag.domain.models import Chunk
from tests.fakes import FakeEmbedder, FakeVectorStore


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source="doc.md", page=None, section=None, position=0)


def test_search_returns_the_chunks_closest_to_the_query():
    embedder, store = FakeEmbedder(), FakeVectorStore()
    chunks = [
        make_chunk("query", "los parámetros de query son opcionales"),
        make_chunk("password", "cambiar la contraseña del usuario"),
        make_chunk("path", "los parámetros de path son obligatorios"),
    ]
    store.add(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))

    results = SearchChunks(embedder, store)("parámetros de query opcionales", k=2)

    assert [result.chunk.chunk_id for result in results] == ["query", "path"]
