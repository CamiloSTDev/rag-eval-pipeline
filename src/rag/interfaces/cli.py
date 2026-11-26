"""Línea de comandos: `ingest` construye el índice, `search` lo consulta, `ask` responde con
citas y `eval` mide el sistema con el golden set."""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

from rag import bootstrap
from rag.application.ports import ProviderError
from rag.application.prompts import InvalidLLMOutput
from rag.application.use_cases.evaluate import QuestionResult, Summary, summarize
from rag.application.use_cases.search_chunks import SearchMode
from rag.domain.models import Answer, RetrievedChunk
from rag.settings import Settings


class CommandError(Exception):
    """Un error de uso del comando; el mensaje dice qué hacer."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Pipeline RAG con evaluación")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="lee el corpus y construye el índice vectorial")
    search = commands.add_parser("search", help="muestra los chunks más parecidos a una consulta")
    search.add_argument("query", help="la pregunta o búsqueda, entre comillas")
    search.add_argument("-k", type=int, default=5, help="cuántos chunks mostrar (por defecto 5)")
    search.add_argument(
        "--mode",
        choices=["vector", "bm25", "hybrid"],
        default="vector",
        help="rama vectorial, por palabras (BM25) o las dos fusionadas con RRF",
    )
    ask = commands.add_parser("ask", help="responde una pregunta citando la documentación")
    ask.add_argument("question", help="la pregunta, entre comillas")
    evaluate = commands.add_parser("eval", help="mide el sistema con el golden set")
    evaluate.add_argument("name", help="nombre de la corrida, por ejemplo E0")
    evaluate.add_argument("--golden", type=Path, help="otro archivo de golden set")
    evaluate.add_argument(
        "--holdout", action="store_true", help="solo las preguntas de hold-out (úsalo al final)"
    )
    golden = commands.add_parser(
        "golden", help="valida el golden set contra el corpus (no necesita claves)"
    )
    golden.add_argument("--golden", type=Path, help="otro archivo de golden set")
    golden.add_argument(
        "--show", action="store_true", help="muestra cada pregunta con su respuesta y sus citas"
    )
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        if args.command == "ingest":
            _ingest(settings)
        elif args.command == "search":
            _search(settings, args.query, args.k, args.mode)
        elif args.command == "ask":
            _ask(settings, args.question)
        elif args.command == "eval":
            _eval(settings, args.name, args.golden, args.holdout)
        else:
            _golden(settings, args.golden, args.show)
    except (bootstrap.ConfigurationError, bootstrap.GoldenSetError, CommandError) as error:
        print(error, file=sys.stderr)
        return 2
    except (InvalidLLMOutput, ProviderError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


def _ingest(settings: Settings) -> None:
    ingest = bootstrap.build_ingest_corpus(settings)
    start = time.perf_counter()
    report = ingest(bootstrap.load_documents(settings))
    print(
        f"{report.documents} documentos -> {report.chunks} chunks en la colección "
        f"{bootstrap.collection_name(settings)} ({time.perf_counter() - start:.1f} s)"
    )


def _search(settings: Settings, query: str, k: int, mode: SearchMode) -> None:
    results = bootstrap.build_search_chunks(settings, mode)(query, k, mode)
    if not results:
        print(
            "Ningún chunk comparte palabras con la consulta."
            if mode == "bm25"
            else "El índice está vacío: corre primero `rag ingest`."
        )
    for rank, result in enumerate(results, start=1):
        print(_format_result(rank, result))


def _ask(settings: Settings, question: str) -> None:
    print(_format_answer(bootstrap.build_answer_question(settings)(question)))


def _eval(settings: Settings, name: str, golden: Path | None, holdout: bool) -> None:
    csv_path = settings.results_path / f"{name}.csv"
    json_path = settings.results_path / f"{name}.json"
    if csv_path.exists() or json_path.exists():
        # Sobrescribir borraría en silencio un experimento anterior
        raise CommandError(
            f"Ya existe la corrida {name} en {settings.results_path}: usa otro nombre."
        )
    evaluate = bootstrap.build_evaluate(settings)
    questions = [q for q in bootstrap.load_golden_set(settings, golden) if q.holdout == holdout]
    if not questions:
        group = "de hold-out" if holdout else "fuera del hold-out"
        raise CommandError(f"El golden set no tiene preguntas {group}.")
    unverified = sum(not question.verified for question in questions)
    if unverified:
        print(f"Atención: {unverified} preguntas sin revisar (verified=false).")
    if settings.judge_model:
        print(f"Juez de fidelidad: {settings.judge_provider} / {settings.judge_model}")
    else:
        print("Sin juez: configura JUDGE_MODEL para medir fidelidad y relevancia.")

    def progress(result: QuestionResult) -> None:
        status = "error" if result.error else ("ok" if result.refusal_correct else "rechazo mal")
        judged = "" if result.faithfulness is None else f" fidelidad={result.faithfulness:.2f}"
        print(f"  {result.id}: hit@5={result.hit_at_5} {status}{judged}")

    results = evaluate(questions, on_result=progress)
    summary = summarize(results)

    settings.results_path.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:  # -sig: Excel lee las tildes
        writer = csv.DictWriter(file, fieldnames=[field.name for field in fields(QuestionResult)])
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    run = {
        "name": name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "split": "holdout" if holdout else "dev",
        "config": bootstrap.run_config(settings),
        "summary": asdict(summary),
    }
    json_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{_format_summary(summary)}\n\nGuardado en {csv_path} y {json_path}")


def _golden(settings: Settings, golden: Path | None, show: bool) -> None:
    questions = bootstrap.load_golden_set(settings, golden)
    for question in questions if show else []:
        flags = [question.kind] + ["hold-out"] * question.holdout + ["revisada"] * question.verified
        print(f"{question.id} ({', '.join(flags)}) {question.question}")
        print(f"  Respuesta: {question.reference_answer}")
        for ref in question.relevant_sources:
            print(
                f"  Fuente: {ref.source.split('/', 1)[-1]} | {ref.section or '(todo el archivo)'}"
            )
            if ref.evidence:
                print(f'    "{ref.evidence}"')
        print()
    kinds = Counter(question.kind for question in questions)
    holdout = sum(question.holdout for question in questions)
    unverified = sum(not question.verified for question in questions)
    print(
        f"Golden set válido: {len(questions)} preguntas ({kinds['single']} de un chunk, "
        f"{kinds['multi']} de varios, {kinds['unanswerable']} sin respuesta), "
        f"{holdout} de hold-out, {unverified} sin revisar."
    )


def _format_result(rank: int, result: RetrievedChunk) -> str:
    chunk = result.chunk
    preview = " ".join(chunk.text.split())
    if len(preview) > 200:
        preview = preview[:200] + "…"
    location = _location(chunk.source, chunk.page)
    section = chunk.section or "(sin sección)"
    return f"{rank}. [{result.score:.3f}] {location}\n   {section}\n   {preview}\n"


def _format_answer(answer: Answer) -> str:
    lines = [answer.text]
    if answer.citations:
        lines += ["", "Fuentes:"]
        for citation in answer.citations:
            section = f" — {citation.section}" if citation.section else ""
            lines.append(f"  [{citation.n}] {_location(citation.source, citation.page)}{section}")
    if answer.invalid_citations:
        numbers = ", ".join(f"[{n}]" for n in answer.invalid_citations)
        lines.append(f"Atención: citas sin fuente {numbers}")
    best = "sin resultados" if answer.top_similarity is None else f"{answer.top_similarity:.3f}"
    stages = " · ".join(f"{stage} {ms:.0f} ms" for stage, ms in answer.latency_ms.items())
    lines += [
        "",
        f"Mejor similitud vectorial: {best} · tokens {answer.tokens_in} entrada / "
        f"{answer.tokens_out} salida · {stages}",
    ]
    return "\n".join(lines)


def _format_summary(summary: Summary) -> str:
    def value(number: float | None, digits: int = 2) -> str:
        return "—" if number is None else f"{number:.{digits}f}"

    return "\n".join(
        [
            f"Preguntas      {summary.questions} (errores: {summary.errors})",
            f"Recuperación   Hit@5 {value(summary.hit_at_5)} · Hit@10 {value(summary.hit_at_10)}"
            f" · MRR {value(summary.mrr)}",
            f"Rechazo        aciertos {value(summary.refusal_accuracy)} · de más "
            f"{summary.false_refusals} · de menos {summary.missed_refusals}",
            f"Citas          respuestas con citas sin fuente: "
            f"{summary.answers_with_invalid_citations}",
            f"Generación     fidelidad {value(summary.faithfulness)} · relevancia "
            f"{value(summary.relevance)} ({summary.judged} juzgadas, "
            f"{summary.judge_errors} errores del juez)",
            f"Operación      latencia p50 {value(summary.latency_p50_ms, 0)} ms · p95 "
            f"{value(summary.latency_p95_ms, 0)} ms · tokens promedio "
            f"{value(summary.tokens_in_mean, 0)} entrada / "
            f"{value(summary.tokens_out_mean, 0)} salida",
        ]
    )


def _location(source: str, page: int | None) -> str:
    return source + (f", p. {page}" if page is not None else "")
