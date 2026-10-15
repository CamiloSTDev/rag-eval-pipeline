"""Lector de HTML: extrae el contenido principal como Markdown y descarta menús y pies."""

from pathlib import Path

import trafilatura

from rag.domain.models import Document
from rag.infrastructure.loaders.text import normalize


def load_html(path: Path, source: str) -> list[Document]:
    page = path.read_bytes()  # en bytes: trafilatura detecta la codificación
    # Salida en Markdown para que los títulos lleguen como "#" y el chunker los use como secciones
    text = trafilatura.extract(
        page,
        output_format="markdown",
        include_formatting=True,
        include_tables=True,
        include_links=False,
        include_images=False,
        include_comments=False,
    )
    if not text:
        return []
    metadata = trafilatura.extract_metadata(page)
    title = metadata.title if metadata else None
    return [Document(text=normalize(text), source=source, title=title)]
