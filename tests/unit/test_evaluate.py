import pytest

from rag.application.judge import Judgement
from rag.application.ports import ProviderError
from rag.application.prompts import REFUSAL, InvalidLLMOutput
from rag.application.use_cases.evaluate import Evaluate, QuestionResult, summarize
from rag.domain.models import Answer, Chunk, GoldenQuestion, RetrievedChunk, SourceRef


def ranking(*sources: str) -> tuple[RetrievedChunk, ...]:
    """Un ranking de búsqueda con un chunk por fuente, en ese orden."""
    return tuple(
        RetrievedChunk(
            Chunk(chunk_id=s, text="t", source=s, page=None, section=None, position=0), 0.5
        )
        for s in sources
    )


def answer(retrieved: tuple[RetrievedChunk, ...], refused: bool = False) -> Answer:
    return Answer(
        text=REFUSAL if refused else "Respuesta [1].",
        citations=(),
        invalid_citations=(),
        refused=refused,
        retrieved=retrieved,
        context=tuple(result.chunk for result in retrieved[:5]),
        top_similarity=0.5,
        latency_ms={"retrieve": 100.0, "generate": 900.0},
        tokens_in=2000,
        tokens_out=50,
    )


def question(id: str, kind: str = "single", source: str = "ok.md") -> GoldenQuestion:
    sources = () if kind == "unanswerable" else (SourceRef(source),)
    return GoldenQuestion(id, f"pregunta {id}", "referencia", sources, kind)  # type: ignore[arg-type]


OTHERS = [f"otro{i}.md" for i in range(9)]
ANSWERS = {
    "pregunta first": answer(ranking("ok.md", *OTHERS)),
    "pregunta third": answer(ranking(*OTHERS[:2], "ok.md", *OTHERS[2:])),
    "pregunta seventh": answer(ranking(*OTHERS[:6], "ok.md", *OTHERS[6:])),
    "pregunta missing": answer(ranking(*OTHERS)),
    "pregunta refused-ok": answer(ranking(*OTHERS), refused=True),
    "pregunta answered-bad": answer(ranking(*OTHERS)),
}


def evaluate(questions: list[GoldenQuestion]) -> list[QuestionResult]:
    return Evaluate(lambda text: ANSWERS[text])(questions)


def test_retrieval_metrics_depend_on_where_the_relevant_chunk_ranks():
    results = evaluate([question("first"), question("third"), question("seventh")])

    assert [(r.hit_at_5, r.hit_at_10) for r in results] == [(1, 1), (1, 1), (0, 1)]
    assert [r.reciprocal_rank for r in results] == pytest.approx([1.0, 1 / 3, 1 / 7])


def test_refusal_is_correct_only_for_questions_without_an_answer():
    results = evaluate(
        [
            question("refused-ok", kind="unanswerable"),
            question("answered-bad", kind="unanswerable"),
            question("first"),
        ]
    )

    assert [r.refusal_correct for r in results] == [True, False, True]
    assert results[0].hit_at_5 is None  # sin respuesta, no hay nada que recuperar


def test_summary_averages_retrieval_over_answerable_questions_only():
    results = evaluate(
        [
            question("first"),
            question("missing"),
            question("refused-ok", kind="unanswerable"),
            question("answered-bad", kind="unanswerable"),
        ]
    )

    summary = summarize(results)

    assert summary.questions == 4
    assert summary.hit_at_5 == 0.5
    assert summary.mrr == 0.5
    assert summary.refusal_accuracy == 0.75
    assert (summary.false_refusals, summary.missed_refusals) == (0, 1)
    assert summary.latency_p50_ms == 1000.0
    assert summary.tokens_in_mean == 2000.0


def test_an_unreadable_model_output_is_recorded_and_left_out_of_the_metrics():
    def broken(text: str) -> Answer:
        if text == "pregunta broken":
            raise InvalidLLMOutput("esto no es JSON")
        return ANSWERS[text]

    results = Evaluate(broken)([question("broken"), question("first")])
    summary = summarize(results)

    assert results[0].error is not None
    assert (summary.questions, summary.errors) == (2, 1)
    assert summary.hit_at_5 == 1.0


def test_a_provider_failure_is_recorded_like_an_unreadable_output():
    def down(text: str) -> Answer:
        raise ProviderError("sin conexión")

    [result] = Evaluate(down)([question("first")])

    assert result.error == "sin conexión"


def test_the_judge_scores_answers_and_its_failures_keep_the_other_metrics():
    def judge(text: str, answer: Answer) -> Judgement | None:
        if answer.refused:
            return None
        if text == "pregunta third":
            raise InvalidLLMOutput("el juez no devolvió JSON")
        return Judgement(faithfulness=0.5, relevant=True, unsupported_claims=("dato inventado",))

    results = Evaluate(lambda text: ANSWERS[text], judge=judge)(
        [question("first"), question("third"), question("refused-ok", kind="unanswerable")]
    )
    summary = summarize(results)

    assert (results[0].faithfulness, results[0].unsupported_claims) == (0.5, "dato inventado")
    assert results[1].judge_error is not None and results[1].hit_at_5 == 1
    assert results[2].faithfulness is None  # un rechazo no se juzga
    assert (summary.faithfulness, summary.relevance, summary.judged) == (0.5, 1.0, 1)
    assert summary.judge_errors == 1


def test_each_result_is_reported_as_soon_as_it_is_ready():
    seen: list[str] = []

    Evaluate(lambda text: ANSWERS[text])(
        [question("first"), question("third")], on_result=lambda result: seen.append(result.id)
    )

    assert seen == ["first", "third"]
