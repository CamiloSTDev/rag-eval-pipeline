from rag.domain.citations import resolve_citations
from rag.domain.models import Chunk, Citation


def make_chunk(chunk_id: str, section: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="texto",
        source=f"{chunk_id}.md",
        page=None,
        section=section,
        position=0,
    )


SOURCES = [make_chunk("a", "Guía"), make_chunk("b"), make_chunk("c")]


def test_valid_citations_point_to_their_source_chunk():
    valid, invalid = resolve_citations("Se usa Query [3]. También Path [1].", SOURCES)

    assert valid == [
        Citation(n=3, chunk_id="c", source="c.md", page=None, section=None),
        Citation(n=1, chunk_id="a", source="a.md", page=None, section="Guía"),
    ]
    assert invalid == []


def test_numbers_without_a_source_are_reported_as_invalid():
    valid, invalid = resolve_citations("Dato real [2]. Dato inventado [7]. Otro [0].", SOURCES)

    assert [citation.n for citation in valid] == [2]
    assert invalid == [7, 0]


def test_grouped_and_repeated_citations_are_counted_once_in_order():
    valid, _ = resolve_citations("Primero [2][1]. Luego [1, 3]. Otra vez [2].", SOURCES)

    assert [citation.n for citation in valid] == [2, 1, 3]


def test_code_indexes_are_not_citations():
    valid, invalid = resolve_citations("Accede con `items[0]` o lista[2] [1].", SOURCES)

    assert [citation.n for citation in valid] == [1]
    assert invalid == []
