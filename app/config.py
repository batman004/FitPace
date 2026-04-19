from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+asyncpg://fitpace:fitpace@db:5432/fitpace",
        description="Async SQLAlchemy connection URL.",
    )
    api_key: str = Field(
        default="dev-secret-key",
        description="Static key required in the X-API-Key header.",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="If set, /chat uses OpenAI; otherwise falls back to Ollama.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama HTTP endpoint used when openai_api_key is not set.",
    )
    scheduler_enabled: bool = Field(
        default=True,
        description="Whether to start the APScheduler nightly recompute job.",
    )
    log_level: LogLevel = Field(
        default="INFO",
        description="Root logger level.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance for dependency injection."""
    return Settings()
