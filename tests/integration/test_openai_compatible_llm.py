from types import SimpleNamespace
from typing import Any, cast

import httpx2
import openai
import pytest
from openai import OpenAI

from rag.application.ports import ProviderError
from rag.domain.models import LLMResult
from rag.infrastructure.llm.openai_compatible import OpenAICompatibleLLM


class StubChatCompletions:
    """Imita client.chat.completions: anota el pedido y responde sin red."""

    def __init__(self, content: str | None) -> None:
        self.content = content
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        message = SimpleNamespace(content=self.content)
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def make_llm(completions: StubChatCompletions) -> OpenAICompatibleLLM:
    client = cast(OpenAI, SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    return OpenAICompatibleLLM(client, "modelo-de-prueba")


def test_generate_sends_system_and_user_in_json_mode_and_reports_tokens():
    completions = StubChatCompletions('{"answer": "Hola [1].", "refused": false}')

    result = make_llm(completions).generate("reglas", "pregunta")

    assert result == LLMResult(
        text='{"answer": "Hola [1].", "refused": false}', tokens_in=120, tokens_out=30
    )
    [request] = completions.requests
    assert request["model"] == "modelo-de-prueba"
    assert request["messages"] == [
        {"role": "system", "content": "reglas"},
        {"role": "user", "content": "pregunta"},
    ]
    assert request["response_format"] == {"type": "json_object"}
    assert request["temperature"] == 0.0


class FailingChatCompletions:
    def create(self, **request: Any) -> SimpleNamespace:
        raise openai.APIConnectionError(request=httpx2.Request("POST", "https://api.example.com"))


def test_errors_of_the_sdk_become_a_provider_error():
    client = cast(
        OpenAI, SimpleNamespace(chat=SimpleNamespace(completions=FailingChatCompletions()))
    )

    with pytest.raises(ProviderError, match="modelo-de-prueba"):
        OpenAICompatibleLLM(client, "modelo-de-prueba").generate("reglas", "pregunta")


def test_empty_content_becomes_an_empty_text():
    result = make_llm(StubChatCompletions(None)).generate("reglas", "pregunta")

    assert result.text == ""
