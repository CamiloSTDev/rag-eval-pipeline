"""Configuración leída de las variables de entorno y del archivo .env."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env trae variables de etapas futuras (LLM_MODEL...) que aún no se leen.
    # env_ignore_empty: "OPENAI_API_KEY=" vacío cuenta como ausente, no como clave vacía.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    corpus_path: Path = Path("data/raw")
    corpus_exclude: list[str] = ["_llm-test.md", "translation-banner.md"]
    index_path: Path = Path("data/index")
