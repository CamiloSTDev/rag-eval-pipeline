from types import SimpleNamespace
from typing import cast

import httpx2
import openai
import pytest
from openai import OpenAI

from rag.application.ports import ProviderError
from rag.infrastructure.embeddings.openai_embed import OpenAIEmbedder, token_counter


class StubEmbeddingsAPI:
    """Imita client.embeddings: anota cada pedido y responde sin red."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, list[str]]] = []

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.requests.append((model, list(input)))
        data = [SimpleNamespace(index=i, embedding=[float(i)]) for i in range(len(input))]
        return SimpleNamespace(data=list(reversed(data)))  # en desorden a propósito


def test_embedder_sends_the_model_and_keeps_the_input_order():
    api = StubEmbeddingsAPI()
    # cast: el falso solo imita client.embeddings, que es lo único que usa el adaptador
    client = cast(OpenAI, SimpleNamespace(embeddings=api))
    embedder = OpenAIEmbedder(client, "text-embedding-3-small")

    vectors = embedder.embed_documents(["a", "b", "c"])

    assert vectors == [[0.0], [1.0], [2.0]]
    assert api.requests == [("text-embedding-3-small", ["a", "b", "c"])]
    assert embedder.embed_query("d") == [0.0]


class FailingEmbeddingsAPI:
    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        raise openai.APIConnectionError(request=httpx2.Request("POST", "https://api.example.com"))


def test_errors_of_the_sdk_become_a_provider_error():
    client = cast(OpenAI, SimpleNamespace(embeddings=FailingEmbeddingsAPI()))

    with pytest.raises(ProviderError, match="embeddings"):
        OpenAIEmbedder(client, "text-embedding-3-small").embed_query("hola")


def test_token_counter_uses_the_tokenizer_of_the_model():
    count = token_counter("text-embedding-3-small")

    assert count("¿Cómo declaro un parámetro de query en FastAPI?") == 14
    assert count("<|endoftext|>") > 0
