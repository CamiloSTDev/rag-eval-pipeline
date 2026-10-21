"""Línea de comandos: `rag ingest` construye el índice, `rag search` lo consulta y `rag ask`
responde con citas."""

import argparse
import sys
import time
from collections.abc import Sequence

from rag import bootstrap
from rag.application.prompts import InvalidLLMOutput
from rag.domain.models import Answer, RetrievedChunk
from rag.settings import Settings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Pipeline RAG con evaluación")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="lee el corpus y construye el índice vectorial")
    search = commands.add_parser("search", help="muestra los chunks más parecidos a una consulta")
    search.add_argument("query", help="la pregunta o búsqueda, entre comillas")
    search.add_argument("-k", type=int, default=5, help="cuántos chunks mostrar (por defecto 5)")
    ask = commands.add_parser("ask", help="responde una pregunta citando la documentación")
    ask.add_argument("question", help="la pregunta, entre comillas")
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        if args.command == "ingest":
            _ingest(settings)
        elif args.command == "search":
            _search(settings, args.query, args.k)
        else:
            _ask(settings, args.question)
    except bootstrap.ConfigurationError as error:
        print(error, file=sys.stderr)
        return 2
    except InvalidLLMOutput as error:
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


def _search(settings: Settings, query: str, k: int) -> None:
    results = bootstrap.build_search_chunks(settings)(query, k)
    if not results:
        print("El índice está vacío: corre primero `rag ingest`.")
    for rank, result in enumerate(results, start=1):
        print(_format_result(rank, result))


def _ask(settings: Settings, question: str) -> None:
    print(_format_answer(bootstrap.build_answer_question(settings)(question)))


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
    best = f"{answer.retrieved[0].score:.3f}" if answer.retrieved else "sin resultados"
    stages = " · ".join(f"{stage} {ms:.0f} ms" for stage, ms in answer.latency_ms.items())
    lines += [
        "",
        f"Mejor puntaje de búsqueda: {best} · tokens {answer.tokens_in} entrada / "
        f"{answer.tokens_out} salida · {stages}",
    ]
    return "\n".join(lines)


def _location(source: str, page: int | None) -> str:
    return source + (f", p. {page}" if page is not None else "")
