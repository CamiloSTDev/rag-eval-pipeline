"""Normalización de texto común a todos los formatos."""

import re
import unicodedata


def normalize(text: str) -> str:
    """Unifica la forma Unicode, quita espacios al final de línea y colapsa líneas en blanco."""
    # NFC: "é" como un solo carácter, para que la búsqueda por palabras no vea dos "é" distintas
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
