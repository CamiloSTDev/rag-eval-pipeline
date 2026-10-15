from types import SimpleNamespace

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
    embedder = OpenAIEmbedder(SimpleNamespace(embeddings=api), "text-embedding-3-small")

    vectors = embedder.embed_documents(["a", "b", "c"])

    assert vectors == [[0.0], [1.0], [2.0]]
    assert api.requests == [("text-embedding-3-small", ["a", "b", "c"])]
    assert embedder.embed_query("d") == [0.0]


def test_token_counter_uses_the_tokenizer_of_the_model():
    count = token_counter("text-embedding-3-small")

    assert count("¿Cómo declaro un parámetro de query en FastAPI?") == 14
    assert count("<|endoftext|>") > 0
