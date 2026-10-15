"""Lector de Markdown, con limpieza de la sintaxis de MkDocs que usa la documentación de FastAPI."""

import html
import re
from collections.abc import Iterator
from pathlib import Path

from rag.domain.models import Document
from rag.infrastructure.loaders.text import normalize

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_FENCE = re.compile(r"^\s*(```|~~~)")
# Inclusión de archivos externos: {* ../../docs_src/app.py hl[3] *} y {!../../docs_src/app.py!}
_INCLUDE = re.compile(r"^\s*\{[*!].*[*!]\}\s*$")
_HEADING_ANCHOR = re.compile(r"^(#{1,6}\s.*?)\s*\{\s*#[^}]*\}\s*$")
# Bloques de MkDocs: "/// tip | Consejo" abre, "///" cierra; "//// tab | Python 3.10+" son pestañas
_BLOCK_OPEN = re.compile(r"^/{3,}\s*(\w+)(?:\s*\|\s*(.*?))?\s*$")
_BLOCK_CLOSE = re.compile(r"^/{3,}\s*$")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_STYLE_OR_SCRIPT = re.compile(r"<(style|script)\b.*?</\1>", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TITLED_TERM = re.compile(r'<(abbr|dfn)\s+title="([^"]*)"\s*>(.*?)</\1>', re.S)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_STASHED = re.compile(r"\x00(\d+)\x00")


def load_markdown(path: Path, source: str) -> list[Document]:
    text = clean_markdown(path.read_text(encoding="utf-8"))
    if not text:
        return []
    heading = re.search(r"^#\s+(.+)$", text, re.M)
    return [Document(text=text, source=source, title=heading.group(1) if heading else None)]


def clean_markdown(text: str) -> str:
    """Quita lo que pertenece a MkDocs y no al contenido. El código se deja intacto."""
    parts: list[str] = []
    for is_code, lines in _segments(_FRONT_MATTER.sub("", text).splitlines()):
        if is_code:
            fenced = _clean_code(lines)
            if fenced:
                parts.append(fenced)
        else:
            parts.append(_clean_prose(lines))
    return normalize("\n".join(parts))


def _segments(lines: list[str]) -> Iterator[tuple[bool, list[str]]]:
    """Separa las líneas en tramos de prosa y bloques de código (con sus vallas ```)."""
    current: list[str] = []
    fence = ""
    for line in lines:
        marker = _FENCE.match(line)
        if fence:
            current.append(line)
            if marker and marker.group(1) == fence:
                yield True, current
                current, fence = [], ""
        elif marker:
            if current:
                yield False, current
            current, fence = [line], marker.group(1)
        else:
            current.append(line)
    if current:
        yield bool(fence), current


def _clean_code(lines: list[str]) -> str:
    """Quita las inclusiones de archivos externos. Si el bloque queda vacío, desaparece."""
    closed = len(lines) > 1 and _FENCE.match(lines[-1]) is not None
    body = lines[1:-1] if closed else lines[1:]
    kept = [line for line in body if not _INCLUDE.match(line)]
    if not any(line.strip() for line in kept):
        return ""
    return "\n".join([lines[0], *kept, *lines[-1:]] if closed else [lines[0], *kept])


def _clean_prose(lines: list[str]) -> str:
    kept: list[str] = []
    for line in lines:
        if _INCLUDE.match(line) or _BLOCK_CLOSE.match(line):
            continue
        block = _BLOCK_OPEN.match(line)
        if block:
            kind, title = block.groups()
            # El título de un aviso ("Consejo", "Advertencia") da contexto; el de una pestaña, no
            if title and kind != "tab":
                kept.append(f"**{title}**")
            continue
        kept.append(_HEADING_ANCHOR.sub(r"\1", line))

    # El código en línea (`<form>`) se aparta para que la limpieza de HTML no lo toque
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"\x00{len(spans) - 1}\x00"

    prose = _INLINE_CODE.sub(stash, "\n".join(kept))
    prose = _STYLE_OR_SCRIPT.sub("", prose)
    prose = _COMMENT.sub("", prose)
    # <abbr title="Asynchronous Server Gateway Interface">ASGI</abbr> → "ASGI (Asynchronous...)"
    prose = _TITLED_TERM.sub(r"\3 (\2)", prose)
    prose = _TAG.sub("", prose)
    prose = _IMAGE.sub("", prose)
    prose = _LINK.sub(r"\1", prose)
    prose = html.unescape(prose)
    return _STASHED.sub(lambda match: spans[int(match.group(1))], prose)
