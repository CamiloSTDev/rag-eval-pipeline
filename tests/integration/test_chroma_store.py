import pytest

from rag.domain.models import Chunk
from rag.infrastructure.stores.chroma_store import ChromaStore


def make_chunk(chunk_id: str, page: int | None = None, section: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=f"texto {chunk_id}",
        source="doc.md",
        page=page,
        section=section,
        position=0,
    )


def test_search_returns_nearest_first_with_cosine_similarity_and_metadata(tmp_path):
    store = ChromaStore(tmp_path, "prueba")
    near = make_chunk("near", page=3, section="Guía > Uso")
    far = make_chunk("far")

    store.add([near, far], [[1.0, 0.0], [0.0, 1.0]])
    results = store.search([1.0, 0.0], k=2)

    assert [result.chunk for result in results] == [near, far]
    assert results[0].score == pytest.approx(1.0, abs=1e-6)
    assert results[1].score == pytest.approx(0.0, abs=1e-6)


def test_adding_an_existing_chunk_id_replaces_it(tmp_path):
    store = ChromaStore(tmp_path, "prueba")
    chunk = make_chunk("a")

    store.add([chunk], [[1.0, 0.0]])
    store.add([chunk], [[0.0, 1.0]])
    results = store.search([0.0, 1.0], k=5)

    assert [result.chunk for result in results] == [chunk]
    assert results[0].score == pytest.approx(1.0, abs=1e-6)
