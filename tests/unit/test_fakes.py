from rag.domain.models import Chunk, RetrievedChunk
from tests.fakes import (
    FakeEmbedder,
    FakeKeywordIndex,
    FakeReranker,
    FakeVectorStore,
    cosine,
)


def make_chunk(chunk_id: str, text: str = "") -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source="doc.md", page=None, section=None, position=0)


def test_embedder_is_stable_and_puts_related_texts_closer():
    embedder = FakeEmbedder()

    query = embedder.embed_query("reembolso del pago")
    related, unrelated = embedder.embed_documents(
        ["el reembolso del pago tarda cinco días", "cambiar la contraseña de la cuenta"]
    )

    assert embedder.embed_query("reembolso del pago") == query
    assert cosine(query, related) > cosine(query, unrelated)


def test_vector_store_returns_nearest_first_and_replaces_by_chunk_id():
    store = FakeVectorStore()
    near, far = make_chunk("near"), make_chunk("far")

    store.add([near, far], [[1.0, 0.0], [0.0, 1.0]])
    store.add([near], [[1.0, 0.1]])
    results = store.search([1.0, 0.0], k=2)

    assert [result.chunk for result in results] == [near, far]
    assert len(store.rows) == 2


def test_keyword_index_ranks_by_shared_words_and_skips_non_matches():
    index = FakeKeywordIndex()
    two = make_chunk("two", "parámetros de query opcionales")
    one = make_chunk("one", "parámetros de path")
    none = make_chunk("none", "dependencias con yield")
    index.build([one, none, two])

    results = index.search("query opcionales", k=5)

    assert [result.chunk for result in results] == [two]
    assert [result.chunk for result in index.search("parámetros query", k=5)] == [two, one]


def test_reranker_reorders_candidates_and_keeps_top_n():
    candidates = [
        RetrievedChunk(make_chunk("weak", "habla sobre otra cosa"), score=0.9),
        RetrievedChunk(make_chunk("strong", "cómo declarar parámetros de query"), score=0.1),
        RetrievedChunk(make_chunk("medium", "parámetros en general"), score=0.5),
    ]

    results = FakeReranker().rerank("parámetros de query", candidates, top_n=2).ranking

    assert [result.chunk.chunk_id for result in results] == ["strong", "medium"]
