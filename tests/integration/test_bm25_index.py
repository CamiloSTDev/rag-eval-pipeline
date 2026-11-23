from rag.domain.models import Chunk
from rag.infrastructure.stores.bm25_index import Bm25Index, tokenize


def make_chunk(chunk_id: str, text: str, section: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, text=text, source="doc.md", page=None, section=section, position=0
    )


CHUNKS = [
    make_chunk("schema", "Establece include_in_schema en False para ocultar el parámetro."),
    make_chunk("query", "Los parámetros de query son opcionales con un valor por defecto."),
    make_chunk("deploy", "Despliega la aplicación con varios workers.", section="Despliegue"),
    make_chunk("path", "Los parámetros de path forman parte de la ruta."),
]


def make_index() -> Bm25Index:
    index = Bm25Index()
    index.build(CHUNKS)
    return index


def test_tokenize_lowercases_removes_accents_and_keeps_identifiers():
    assert tokenize("Parámetro include_in_schema, AÑO") == ["parametro", "include_in_schema", "ano"]


def test_an_exact_identifier_finds_its_chunk_first():
    results = make_index().search("¿qué hace include_in_schema?", k=3)

    assert results[0].chunk.chunk_id == "schema"


def test_search_ignores_accents_and_case():
    results = make_index().search("PARAMETROS QUERY", k=2)

    assert results[0].chunk.chunk_id == "query"


def test_the_section_title_is_searchable_too():
    results = make_index().search("despliegue", k=3)

    assert [result.chunk.chunk_id for result in results] == ["deploy"]


def test_chunks_without_any_shared_word_are_not_returned():
    assert make_index().search("contraseña usuario", k=5) == []


def test_an_empty_index_returns_nothing():
    index = Bm25Index()
    index.build([])

    assert index.search("parámetros", k=5) == []
