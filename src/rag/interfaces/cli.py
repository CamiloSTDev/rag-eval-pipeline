"""Línea de comandos: `rag ingest` construye el índice y `rag search` lo consulta."""

import argparse
import sys
import time
from collections.abc import Sequence

from rag import bootstrap
from rag.domain.models import RetrievedChunk
from rag.settings import Settings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Pipeline RAG con evaluación")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="lee el corpus y construye el índice vectorial")
    search = commands.add_parser("search", help="muestra los chunks más parecidos a una consulta")
    search.add_argument("query", help="la pregunta o búsqueda, entre comillas")
    search.add_argument("-k", type=int, default=5, help="cuántos chunks mostrar (por defecto 5)")
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        if args.command == "ingest":
            _ingest(settings)
        else:
            _search(settings, args.query, args.k)
    except bootstrap.ConfigurationError as error:
        print(error, file=sys.stderr)
        return 2
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
        print(_format(rank, result))


def _format(rank: int, result: RetrievedChunk) -> str:
    chunk = result.chunk
    location = chunk.source + (f", p. {chunk.page}" if chunk.page is not None else "")
    preview = " ".join(chunk.text.split())
    if len(preview) > 200:
        preview = preview[:200] + "…"
    section = chunk.section or "(sin sección)"
    return f"{rank}. [{result.score:.3f}] {location}\n   {section}\n   {preview}\n"
