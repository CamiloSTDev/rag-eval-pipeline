"""Configuración leída de las variables de entorno y del archivo .env."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env trae variables de etapas futuras (LLM_MODEL...) que aún no se leen.
    # env_ignore_empty: "OPENAI_API_KEY=" vacío cuenta como ausente, no como clave vacía.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    # Los valores por defecto son la configuración elegida con los experimentos (ver el README):
    # embeddings large (E5), búsqueda híbrida (E3), reranker (E4) y umbral recalibrado para large.
    # El baseline E0 es text-embedding-3-small, USE_BM25=false, USE_RERANKER=false y
    # REFUSAL_THRESHOLD=0 (sin umbral).
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-large"
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    corpus_path: Path = Path("data/raw")
    corpus_exclude: list[str] = ["_llm-test.md", "translation-banner.md"]
    index_path: Path = Path("data/index")
    golden_path: Path = Path("data/eval/golden.jsonl")
    results_path: Path = Path("results")
    top_k_retrieve: int = 30
    top_k_final: int = 5
    use_bm25: bool = True  # búsqueda híbrida (E3)
    rrf_k: int = 60
    # Reranker con un LLM (E4)
    use_reranker: bool = True
    reranker_provider: Literal["openai", "deepseek"] = "openai"
    reranker_model: str = "gpt-4.1-mini"
    reranker_temperature: float | Literal["default"] = 0.0
    rerank_candidates: int = 20
    # Calibrado con large, entre la pregunta sin respuesta de menor similitud (0,380) y la con
    # respuesta de menor similitud (0,499). Cada modelo de embeddings tiene su escala: con small
    # el umbral seguro era 0,50, que con large rechazaría una pregunta con respuesta
    refusal_threshold: float | None = 0.45

    # Generación: OpenAI usa OPENAI_API_KEY; DeepSeek, DEEPSEEK_API_KEY
    llm_provider: Literal["openai", "deepseek"] = "openai"
    llm_model: str | None = None
    llm_base_url: str | None = None  # solo si quieres otro servidor compatible
    # "default" = no enviar temperatura: los modelos de razonamiento solo aceptan la suya
    llm_temperature: float | Literal["default"] = 0.0
    deepseek_api_key: SecretStr | None = None

    # Juez de la evaluación: un modelo distinto del que genera. Sin JUDGE_MODEL no hay juez
    judge_provider: Literal["openai", "deepseek"] = "deepseek"
    judge_model: str | None = None
    judge_temperature: float | Literal["default"] = 0.0

    # API: sin clave de administrador, POST /ingest queda desactivado
    admin_api_key: SecretStr | None = None
