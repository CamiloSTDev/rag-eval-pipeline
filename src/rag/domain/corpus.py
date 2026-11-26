"""Versión del corpus: una huella corta de los documentos limpios que se ingieren."""

import hashlib
from collections.abc import Iterable

from rag.domain.models import Document


def corpus_version(documents: Iterable[Document]) -> str:
    """Hash de (source, page, text) de cada documento, en orden de source y página.

    Cambia si se agrega, quita o edita un documento, y también si cambia la limpieza de los
    loaders, porque se calcula sobre el texto ya limpio: el que de verdad se indexa.
    """
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda d: (d.source, d.page or 0)):
        for part in (document.source, str(document.page), document.text):
            # El separador evita que ("ab", "c") y ("a", "bc") den el mismo hash
            digest.update(part.encode() + b"\0")
    return digest.hexdigest()[:12]
