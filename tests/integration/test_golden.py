import json

import pytest

from rag import bootstrap
from rag.domain.models import GoldenQuestion, SourceRef
from rag.infrastructure.golden import GoldenSetError, load_golden
from rag.settings import Settings

VALID = """\
{"id": "q001", "question": "¿Cómo hago opcional un query?", "reference_answer": "Con None.", \
"relevant_sources": [{"source": "query.md", "section": "Query > Opcionales"}], "type": "single"}

{"id": "q002", "question": "¿Corre en un mainframe?", "reference_answer": "No hay información.", \
"type": "unanswerable", "holdout": true}
"""


def test_valid_golden_set_is_read_into_questions(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(VALID, encoding="utf-8")

    questions = load_golden(path)

    assert questions == [
        GoldenQuestion(
            id="q001",
            question="¿Cómo hago opcional un query?",
            reference_answer="Con None.",
            relevant_sources=(SourceRef("query.md", section="Query > Opcionales"),),
            kind="single",
        ),
        GoldenQuestion(
            id="q002",
            question="¿Corre en un mainframe?",
            reference_answer="No hay información.",
            relevant_sources=(),
            kind="unanswerable",
            holdout=True,
        ),
    ]


@pytest.mark.parametrize(
    ("line", "problem"),
    [
        ('{"id": "q1", "question": "?", "reference_answer": "r", "type": "single"}', "falta"),
        (
            '{"id": "q1", "question": "?", "reference_answer": "r", "type": "unanswerable", '
            '"relevant_sources": [{"source": "a.md"}]}',
            "no lleva",
        ),
        (
            '{"id": "q1", "question": "?", "reference_answer": "r", "type": "single", '
            '"relevant_sources": [{"source": "a.md", "seccion": "x"}]}',
            "seccion",
        ),
        ('{"id": "q1", "question": "?", "reference_answer": "r", "type": "facil"}', "type"),
    ],
)
def test_errors_point_to_the_line_and_the_problem(tmp_path, line, problem):
    path = tmp_path / "golden.jsonl"
    path.write_text(f"\n{line}\n", encoding="utf-8")

    with pytest.raises(GoldenSetError, match=f"(?s)línea 2.*{problem}"):
        load_golden(path)


CORPUS_PAGE = """\
# Parámetros de Query { #query-parameters }

## Parámetros opcionales { #optional }

Puedes declarar parámetros opcionales estableciendo su valor por defecto en `None`.
"""


def check_against_corpus(tmp_path, *sources: dict) -> list[GoldenQuestion]:
    corpus = tmp_path / "raw"
    corpus.mkdir()
    (corpus / "query.md").write_text(CORPUS_PAGE, encoding="utf-8")
    row = {
        "id": "q1",
        "question": "?",
        "reference_answer": "r",
        "type": "single",
        "relevant_sources": list(sources),
    }
    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    # Los valores explícitos tienen prioridad sobre .env, así el test no depende de él
    settings = Settings(corpus_path=corpus, corpus_exclude=[], golden_path=golden)
    return bootstrap.load_golden_set(settings)


def test_sources_sections_and_quotes_that_match_the_clean_corpus_pass(tmp_path):
    [question] = check_against_corpus(
        tmp_path,
        {
            "source": "query.md",
            "section": "Parámetros de Query > Parámetros opcionales",
            "evidence": "estableciendo su valor  por defecto en `None`",  # espacios: da igual
        },
    )

    assert question.relevant_sources[0].evidence is not None


@pytest.mark.parametrize(
    ("ref", "problem"),
    [
        ({"source": "otro.md"}, "no es un documento"),
        ({"source": "query.md", "section": "Parámetros opcionales"}, "no existe la sección"),
        (
            {
                "source": "query.md",
                "section": "Parámetros de Query > Parámetros opcionales",
                "evidence": "con un valor por defecto de cero",
            },
            "la cita no está",
        ),
    ],
)
def test_mismatches_with_the_corpus_are_reported(tmp_path, ref, problem):
    with pytest.raises(GoldenSetError, match=problem):
        check_against_corpus(tmp_path, ref)


def test_repeated_ids_are_rejected(tmp_path):
    first_line = VALID.splitlines()[0]
    path = tmp_path / "golden.jsonl"
    path.write_text(f"{first_line}\n{first_line}\n", encoding="utf-8")

    with pytest.raises(GoldenSetError, match="repetido"):
        load_golden(path)
