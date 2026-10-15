"""Lector de PDF: un Document por página, sin encabezados ni pies de página repetidos."""

import re
from collections import Counter
from pathlib import Path

import pymupdf

from rag.domain.models import Document
from rag.infrastructure.loaders.text import normalize

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_DIGITS = re.compile(r"\d+")
_EDGE_BLOCKS = 2  # bloques que se revisan al principio y al final de cada página
_REPEAT_SHARE = 0.6  # un borde que aparece en el 60 % de las páginas es encabezado o pie


def load_pdf(path: Path, source: str) -> list[Document]:
    with pymupdf.open(path) as pdf:
        title = (pdf.metadata or {}).get("title") or None
        # get_text("blocks") entrega párrafos; el tipo 0 es texto y el 1, imagen
        pages = [[block[4] for block in page.get_text("blocks") if block[6] == 0] for page in pdf]

    repeated = _repeated_edges(pages)
    documents: list[Document] = []
    for number, blocks in enumerate(pages, start=1):
        kept = [
            block
            for index, block in enumerate(blocks)
            if not (_is_edge(index, len(blocks)) and _key(block) in repeated)
        ]
        text = normalize("\n\n".join(_join_lines(block) for block in kept))
        if text:
            documents.append(Document(text=text, source=source, title=title, page=number))
    return documents


def _repeated_edges(pages: list[list[str]]) -> set[str]:
    """Bloques que se repiten en los bordes de la mayoría de las páginas."""
    if len(pages) < 3:
        return set()
    counts = Counter(
        key
        for blocks in pages
        for key in {
            _key(block) for index, block in enumerate(blocks) if _is_edge(index, len(blocks))
        }
    )
    return {key for key, count in counts.items() if count >= _REPEAT_SHARE * len(pages)}


def _is_edge(index: int, total: int) -> bool:
    # En páginas con pocos bloques los bordes se achican: el bloque central nunca es borde
    per_side = min(_EDGE_BLOCKS, (total - 1) // 2)
    return index < per_side or index >= total - per_side


def _key(block: str) -> str:
    # "Página 3" y "Página 4" cuentan como el mismo pie
    return _DIGITS.sub("#", " ".join(block.lower().split()))


def _join_lines(block: str) -> str:
    """Une las líneas de un párrafo y repara las palabras cortadas con guion al final de línea."""
    return " ".join(_HYPHEN_BREAK.sub(r"\1\2", block).split())
