import pytest

from app.config import Settings, get_settings


def test_defaults_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "DATABASE_URL",
        "API_KEY",
        "OPENAI_API_KEY",
        "OLLAMA_BASE_URL",
        "SCHEDULER_ENABLED",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.database_url.startswith("postgresql+asyncpg://")
    assert settings.api_key == "dev-secret-key"
    assert settings.openai_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.scheduler_enabled is True
    assert settings.log_level == "INFO"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
    monkeypatch.setenv("API_KEY", "super-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.database_url == "postgresql+asyncpg://u:p@h/db"
    assert settings.api_key == "super-secret"
    assert settings.openai_api_key == "sk-test"
    assert settings.scheduler_enabled is False
    assert settings.log_level == "DEBUG"


def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_LEVEL")

    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
