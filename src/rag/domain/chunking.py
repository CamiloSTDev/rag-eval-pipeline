"""División de documentos en chunks. Función pura: mismo documento, mismos chunks."""

import hashlib
import re
from collections.abc import Callable, Iterator

from rag.domain.models import Chunk, Document

# El dominio no conoce ningún tokenizador: quien llama decide cómo se cuentan los tokens.
TokenCounter = Callable[[str], int]

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST_ITEM = re.compile(r"^\s*(?:[*+-]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# De más grueso a más fino: por líneas y luego por palabras
_FINER_SPLITS = (re.compile(r"[^\n]*\n+|[^\n]+"), re.compile(r"\S+\s*"))


def chunk_document(
    document: Document,
    *,
    max_tokens: int,
    overlap_tokens: int,
    count_tokens: TokenCounter,
) -> list[Chunk]:
    """Parte un documento Markdown en chunks de hasta `max_tokens`.

    Los encabezados son fronteras: ningún chunk mezcla secciones, y el encabezado no va en
    el texto sino en `section`. Dentro de una sección, cada chunk empieza repitiendo hasta
    `overlap_tokens` del final del anterior.
    """
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("Se requiere 0 <= overlap_tokens < max_tokens")

    chunks: list[Chunk] = []
    for section, body in _sections(document.text):
        # Cada unidad cabe en max - overlap: así, sumada al solape heredado, nunca pasa de max
        units = [
            piece
            for unit in _units(body)
            for piece in _fit(unit, max_tokens - overlap_tokens, count_tokens)
        ]
        for text in _pack(units, max_tokens, overlap_tokens, count_tokens):
            position = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(document.source, document.page, position, text),
                    text=text,
                    source=document.source,
                    page=document.page,
                    section=section,
                    position=position,
                )
            )
    return chunks


def _sections(text: str) -> Iterator[tuple[str | None, list[str]]]:
    """Entrega (ruta de encabezados, líneas del cuerpo) por cada sección del texto."""
    path: list[tuple[int, str]] = []
    body: list[str] = []
    fence = ""
    for line in text.splitlines():
        marker = _FENCE.match(line)
        if marker and not fence:
            fence = marker.group(1)
        elif marker and marker.group(1) == fence:
            fence = ""
        # Dentro de un bloque de código, "# algo" es un comentario y no un encabezado
        heading = None if fence or marker else _HEADING.match(line)
        if heading is None:
            body.append(line)
            continue
        yield _breadcrumb(path), body
        level, title = len(heading.group(1)), heading.group(2)
        path = [(lvl, name) for lvl, name in path if lvl < level] + [(level, title)]
        body = []
    yield _breadcrumb(path), body


def _breadcrumb(path: list[tuple[int, str]]) -> str | None:
    return " > ".join(name for _, name in path) or None


def _blocks(lines: list[str]) -> list[tuple[bool, str]]:
    """Agrupa líneas en bloques (indivisible, texto). Código, tablas y listas son indivisibles."""
    blocks: list[tuple[bool, str]] = []
    kind = ""
    current: list[str] = []
    fence = ""

    def close() -> None:
        nonlocal kind, current
        if current:
            blocks.append((kind != "prose", "\n".join(current)))
        kind, current = "", []

    for line in lines:
        marker = _FENCE.match(line)
        if fence:
            current.append(line)
            if marker and marker.group(1) == fence:
                fence = ""
                close()
        elif marker:
            close()
            kind, current, fence = "code", [line], marker.group(1)
        elif not line.strip():
            close()
        else:
            line_kind = _line_kind(line, kind)
            if line_kind != kind:
                close()
                kind = line_kind
            current.append(line)
    close()
    return blocks


def _line_kind(line: str, current: str) -> str:
    if line.lstrip().startswith("|"):
        return "table"
    if _LIST_ITEM.match(line) or (current == "list" and line[:1].isspace()):
        return "list"
    return "prose"


def _units(lines: list[str]) -> Iterator[str]:
    """Piezas que se empaquetan: bloques indivisibles enteros y prosa oración por oración.

    Cada unidad lleva su separador al final, así que unirlas reconstruye el texto.
    """
    for atomic, block in _blocks(lines):
        if atomic:
            yield block + "\n\n"
            continue
        sentences = _SENTENCE_END.split(block)
        for sentence in sentences[:-1]:
            yield sentence + " "
        yield sentences[-1] + "\n\n"


def _fit(piece: str, limit: int, count_tokens: TokenCounter) -> list[str]:
    """Parte `piece` en trozos cada vez más finos hasta que cada uno quepa en `limit`."""
    if count_tokens(piece) <= limit or len(piece) <= 1:
        return [piece]
    for pattern in _FINER_SPLITS:
        parts = pattern.findall(piece)
        if len(parts) > 1:
            return [fitted for part in parts for fitted in _fit(part, limit, count_tokens)]
    # Una sola "palabra" que no cabe (una URL enorme, por ejemplo): se corta por la mitad
    middle = len(piece) // 2
    return _fit(piece[:middle], limit, count_tokens) + _fit(piece[middle:], limit, count_tokens)


def _pack(
    units: list[str], max_tokens: int, overlap_tokens: int, count_tokens: TokenCounter
) -> list[str]:
    """Llena cada chunk hasta el máximo y siembra el siguiente con el final del anterior."""
    texts: list[str] = []
    window: list[str] = []
    size = 0
    for unit in units:
        unit_size = count_tokens(unit)
        if window and size + unit_size > max_tokens:
            texts.append("".join(window).strip())
            window, size = _tail(window, overlap_tokens, count_tokens)
        window.append(unit)
        size += unit_size
    if window:
        texts.append("".join(window).strip())
    return texts


def _tail(window: list[str], budget: int, count_tokens: TokenCounter) -> tuple[list[str], int]:
    """Últimas unidades enteras de `window` que caben en `budget` tokens."""
    size = 0
    kept = 0
    for unit in reversed(window):
        unit_size = count_tokens(unit)
        if size + unit_size > budget:
            break
        size += unit_size
        kept += 1
    return window[len(window) - kept :], size


def _chunk_id(source: str, page: int | None, position: int, text: str) -> str:
    """Identificador derivado del contenido: reingerir el mismo documento no crea duplicados."""
    key = "\x1f".join((source, str(page), str(position), text))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
