from rag.domain.corpus import corpus_version
from rag.domain.models import Document

GUIDE = Document(text="Instala con uv.", source="guia.md")
API = Document(text="Usa FastAPI.", source="api.md")


def test_the_version_does_not_depend_on_the_loading_order():
    assert corpus_version([GUIDE, API]) == corpus_version([API, GUIDE])


def test_editing_adding_or_removing_a_document_changes_the_version():
    base = corpus_version([GUIDE, API])
    edited = Document(text="Instala con pip.", source="guia.md")

    assert corpus_version([edited, API]) != base
    assert corpus_version([GUIDE]) != base
    assert corpus_version([GUIDE, API, Document(text="Nuevo.", source="nuevo.md")]) != base


def test_moving_text_between_fields_changes_the_version():
    assert corpus_version([Document(text="c", source="ab")]) != corpus_version(
        [Document(text="bc", source="a")]
    )


def test_the_version_is_short_and_hexadecimal():
    version = corpus_version([GUIDE])

    assert len(version) == 12
    int(version, 16)
