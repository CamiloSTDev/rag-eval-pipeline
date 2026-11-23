import pytest

from rag.domain.models import GoldenQuestion, SourceRef
from rag.infrastructure.golden import GoldenSetError, load_golden

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


def test_repeated_ids_are_rejected(tmp_path):
    first_line = VALID.splitlines()[0]
    path = tmp_path / "golden.jsonl"
    path.write_text(f"{first_line}\n{first_line}\n", encoding="utf-8")

    with pytest.raises(GoldenSetError, match="repetido"):
        load_golden(path)
