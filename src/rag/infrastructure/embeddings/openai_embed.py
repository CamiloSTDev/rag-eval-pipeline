"""Embeddings con la API de OpenAI, y conteo de tokens con el tokenizador del mismo modelo."""

import openai
import tiktoken
from openai import OpenAI

from rag.application.ports import ProviderError
from rag.domain.chunking import TokenCounter


class OpenAIEmbedder:
    def __init__(self, client: OpenAI, model: str) -> None:
        # El cliente llega hecho (con su clave) desde bootstrap.py; aquí no se lee configuración
        self._client = client
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            # El SDK ya reintenta con espera creciente los errores transitorios (red, 429, 5xx)
            response = self._client.embeddings.create(model=self._model, input=texts)
        except openai.APIError as error:
            raise ProviderError(f"Falló el servicio de embeddings: {error}") from error
        # Cada vector trae el índice de su texto; ordenar por él garantiza el emparejamiento
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def token_counter(model: str) -> TokenCounter:
    """Cuenta tokens igual que el modelo, para que el límite del chunker sea exacto."""
    encoding = tiktoken.encoding_for_model(model)
    # disallowed_special=(): un texto que mencione "<|endoftext|>" se cuenta en vez de fallar
    return lambda text: len(encoding.encode(text, disallowed_special=()))
