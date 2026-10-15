import pytest

from rag.domain.chunking import chunk_document
from rag.domain.models import Chunk, Document


def count_words(text: str) -> int:
    """Contador falso para los tests: un token por palabra, fácil de calcular a mano."""
    return len(text.split())


SENTENCES = [f"Oración número {i} del texto." for i in range(20)]  # 5 palabras cada una

CODE_BLOCK = """\
```python
# esto es un comentario, no un título
def saludar(nombre):
    return f"hola {nombre}"
```"""

LONG_DOCUMENT = Document(
    source="guia.md",
    text="\n\n".join(
        [
            "# Guía",
            " ".join(SENTENCES),
            "## Sin puntuación",
            " ".join(f"palabra{i}" for i in range(60)),
            "## Código y listas",
            CODE_BLOCK,
            "\n".join(f"* elemento {i} de la lista" for i in range(10)),
        ]
    ),
)


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens"),
    [(20, 0), (20, 5), (50, 10), (7, 3)],
)
def test_no_chunk_exceeds_max_tokens(max_tokens, overlap_tokens):
    chunks = chunk_document(
        LONG_DOCUMENT,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        count_tokens=count_words,
    )

    assert chunks
    assert all(count_words(chunk.text) <= max_tokens for chunk in chunks)


def test_each_chunk_starts_with_the_end_of_the_previous_one():
    document = Document(text=" ".join(SENTENCES), source="doc.md")

    chunks = chunk_document(document, max_tokens=20, overlap_tokens=10, count_tokens=count_words)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_ten_words = " ".join(previous.text.split()[-10:])
        assert current.text.startswith(last_ten_words)
    assert all(any(sentence in chunk.text for chunk in chunks) for sentence in SENTENCES)


def chunk_small(document: Document) -> list[Chunk]:
    return chunk_document(document, max_tokens=20, overlap_tokens=5, count_tokens=count_words)


def test_same_document_gives_same_chunks_with_unique_ids():
    first = chunk_small(LONG_DOCUMENT)
    second = chunk_small(LONG_DOCUMENT)

    assert first == second
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert [chunk.position for chunk in first] == list(range(len(first)))


def test_chunk_id_depends_on_the_source():
    text = " ".join(SENTENCES)

    in_a = chunk_small(Document(text=text, source="a.md"))
    in_b = chunk_small(Document(text=text, source="b.md"))

    assert [chunk.text for chunk in in_a] == [chunk.text for chunk in in_b]
    assert {chunk.chunk_id for chunk in in_a}.isdisjoint(chunk.chunk_id for chunk in in_b)


def test_sections_follow_headings_and_ignore_comments_inside_code():
    document = Document(
        source="guia.md",
        text="\n\n".join(
            [
                "Texto previo al primer título.",
                "# Guía",
                "Introducción de la guía.",
                "## Instalación",
                "Ejecuta el instalador.",
                CODE_BLOCK,
                "## Uso",
                "Llama a la función.",
            ]
        ),
    )

    chunks = chunk_document(document, max_tokens=100, overlap_tokens=10, count_tokens=count_words)

    assert [chunk.section for chunk in chunks] == [
        None,
        "Guía",
        "Guía > Instalación",
        "Guía > Uso",
    ]
    assert chunks[2].text == "Ejecuta el instalador.\n\n" + CODE_BLOCK


def test_code_block_that_fits_is_never_split():
    document = Document(
        source="doc.md",
        text="\n\n".join([" ".join(SENTENCES[:3]), CODE_BLOCK, " ".join(SENTENCES[3:6])]),
    )
    assert count_words(CODE_BLOCK) == 15

    chunks = chunk_document(document, max_tokens=25, overlap_tokens=5, count_tokens=count_words)

    assert len(chunks) > 1
    assert sum(CODE_BLOCK in chunk.text for chunk in chunks) == 1


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens"),
    [(0, 0), (10, 10), (10, 11), (10, -1)],
)
def test_invalid_sizes_are_rejected(max_tokens, overlap_tokens):
    document = Document(text="Hola.", source="doc.md")

    with pytest.raises(ValueError):
        chunk_document(
            document,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            count_tokens=count_words,
        )
