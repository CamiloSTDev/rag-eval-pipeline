"""Lectores del corpus: convierten archivos en objetos Document con el texto ya limpio."""

from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

from rag.domain.models import Document
from rag.infrastructure.loaders.html import load_html
from rag.infrastructure.loaders.markdown import load_markdown
from rag.infrastructure.loaders.pdf import load_pdf

_LOADERS: dict[str, Callable[[Path, str], list[Document]]] = {
    ".md": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}


def load_corpus(root: Path, exclude: Sequence[str] = ()) -> list[Document]:
    """Lee los archivos soportados bajo `root`, en orden alfabético de ruta.

    `exclude` son patrones glob comparados desde la derecha de la ruta relativa: "_*.md"
    excluye cualquier archivo que empiece con "_" en cualquier carpeta. Los formatos no
    soportados (la licencia, por ejemplo) se saltan sin error.
    """
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        loader = _LOADERS.get(path.suffix.lower())
        source = path.relative_to(root).as_posix()
        if loader is None or not path.is_file():
            continue
        if any(PurePosixPath(source).match(pattern) for pattern in exclude):
            continue
        documents.extend(loader(path, source))
    return documents
