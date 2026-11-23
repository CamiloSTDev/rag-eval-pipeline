"""Cliente de chat con el formato de la API de OpenAI: sirve para OpenAI y para DeepSeek."""

import openai
from openai import OpenAI

from rag.application.ports import ProviderError
from rag.domain.models import LLMResult


class OpenAICompatibleLLM:
    def __init__(self, client: OpenAI, model: str, temperature: float = 0.0) -> None:
        # El cliente llega con su clave y su base_url: eso es lo único que cambia entre proveedores
        self._client = client
        self._model = model
        self._temperature = temperature

    def generate(self, system: str, user: str) -> LLMResult:
        try:
            # El SDK ya reintenta con espera creciente los errores transitorios (red, 429, 5xx)
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._temperature,
                # Modo JSON: la respuesta es un JSON válido (DeepSeek pide "json" en el prompt)
                response_format={"type": "json_object"},
            )
        except openai.APIError as error:
            raise ProviderError(f"Falló el proveedor del LLM ({self._model}): {error}") from error
        usage = response.usage
        return LLMResult(
            text=response.choices[0].message.content or "",
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
        )
