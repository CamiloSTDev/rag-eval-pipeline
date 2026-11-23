import pytest

from rag.application.retrieval import Retriever
from rag.application.use_cases.search_chunks import SearchChunks
from rag.domain.models import Chunk
from tests.fakes import FakeEmbedder, FakeKeywordIndex, FakeVectorStore


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source="doc.md", page=None, section=None, position=0)


CHUNKS = [
    make_chunk("query", "los parámetros de query son opcionales"),
    make_chunk("password", "cambiar la contraseña del usuario"),
    make_chunk("path", "los parámetros de path son obligatorios"),
]


def make_search(hybrid: bool) -> SearchChunks:
    embedder, store = FakeEmbedder(), FakeVectorStore()
    store.add(CHUNKS, embedder.embed_documents([chunk.text for chunk in CHUNKS]))
    keywords = None
    if hybrid:
        keywords = FakeKeywordIndex()
        keywords.build(CHUNKS)
    return SearchChunks(Retriever(embedder, store, keywords))


def test_vector_search_returns_the_chunks_closest_to_the_query():
    results = make_search(hybrid=False)("parámetros de query opcionales", k=2)

    assert [result.chunk.chunk_id for result in results] == ["query", "path"]


def test_bm25_mode_returns_only_chunks_that_share_words():
    results = make_search(hybrid=True)("contraseña", k=5, mode="bm25")

    assert [result.chunk.chunk_id for result in results] == ["password"]


def test_hybrid_mode_fuses_both_branches():
    results = make_search(hybrid=True)("parámetros de query opcionales", k=3, mode="hybrid")

    assert results[0].chunk.chunk_id == "query"
    assert results[0].score == pytest.approx(2 / 61)  # primero en las dos ramas


def test_bm25_mode_without_a_keyword_index_is_an_error():
    with pytest.raises(ValueError, match="BM25"):
        make_search(hybrid=False)("contraseña", k=5, mode="bm25")
