"""Citas [n] dentro de una respuesta: cuáles apuntan a una fuente real y cuáles no."""

import re

from rag.domain.models import Chunk, Citation

# "[2]", "[1, 3]" o "[1][2]". El (?<!\w) descarta índices de código como items[0]
_CITATION = re.compile(r"(?<!\w)\[(\d+(?:\s*,\s*\d+)*)\]")


def resolve_citations(text: str, sources: list[Chunk]) -> tuple[list[Citation], list[int]]:
    """Separa las citas del texto en válidas (con su chunk) e inválidas (número sin fuente).

    Las fuentes se numeran desde 1 en el orden de `sources`. Cada número aparece una sola
    vez, en el orden en que se citó por primera vez.
    """
    valid: list[Citation] = []
    invalid: list[int] = []
    for n in _cited_numbers(text):
        if 1 <= n <= len(sources):
            chunk = sources[n - 1]
            valid.append(Citation(n, chunk.chunk_id, chunk.source, chunk.page, chunk.section))
        else:
            invalid.append(n)
    return valid, invalid


def _cited_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for group in _CITATION.findall(text):
        for part in group.split(","):
            n = int(part)
            if n not in numbers:
                numbers.append(n)
    return numbers
