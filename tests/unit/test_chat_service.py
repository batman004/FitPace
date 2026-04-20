"""Unit tests for app/services/chat_service.py."""
from __future__ import annotations

import uuid

import pytest

from app.services.chat_service import UnsafeSQLError, _strip_fences, validate_sql


def test_validate_sql_accepts_plain_select() -> None:
    out = validate_sql("SELECT * FROM goals WHERE user_id = :user_id LIMIT 10")
    assert out.lower().startswith("select")


def test_validate_sql_accepts_cte() -> None:
    out = validate_sql(
        "WITH latest AS (SELECT * FROM progress_logs LIMIT 5) SELECT * FROM latest"
    )
    assert out.lower().startswith("with")


def test_validate_sql_strips_markdown_fences() -> None:
    out = validate_sql(
        "```sql\nSELECT id FROM users WHERE id = :user_id\n```"
    )
    assert "```" not in out
    assert out.lower().startswith("select")


def test_validate_sql_strips_trailing_semicolon() -> None:
    out = validate_sql("SELECT 1;")
    assert out == "SELECT 1"


@pytest.mark.parametrize(
    "bad",
    [
        "DELETE FROM users",
        "DROP TABLE goals",
        "UPDATE users SET email='x'",
        "INSERT INTO goals VALUES (1)",
        "ALTER TABLE goals ADD COLUMN x INT",
        "TRUNCATE progress_logs",
        "SELECT 1; DROP TABLE users",
    ],
)
def test_validate_sql_rejects_non_select(bad: str) -> None:
    with pytest.raises(UnsafeSQLError):
        validate_sql(bad)


def test_validate_sql_rejects_empty() -> None:
    with pytest.raises(UnsafeSQLError):
        validate_sql("   ")


def test_strip_fences_handles_plain_sql() -> None:
    assert _strip_fences("SELECT 1") == "SELECT 1"


@pytest.mark.asyncio
async def test_answer_question_end_to_end(monkeypatch) -> None:
    """Full pipeline against an in-memory SQLite DB with a stubbed LLM."""
    from app.database import Base, dispose_engine, get_sessionmaker, init_engine
    import app.models  # noqa: F401  (registers ORM metadata)
    from app.models.user import User
    from app.services.chat_service import answer_question

    await dispose_engine()
    engine = init_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    user_id = uuid.uuid4()
    factory = get_sessionmaker()
    async with factory() as db:
        db.add(
            User(
                id=user_id,
                first_name="Ada",
                last_name="Lovelace",
                email="ada@example.com",
                password_hash="x",
            )
        )
        await db.commit()

    sql_from_llm = "SELECT first_name FROM users WHERE id = :user_id LIMIT 1"
    call_log: list[tuple[str, str]] = []

    async def fake_llm(system: str, user: str) -> str:
        call_log.append((system[:20], user[:40]))
        # First call asks for SQL, second asks for the answer.
        if "SELECT" in system or "SQL" in system:
            return sql_from_llm
        return "You're Ada."

    async with factory() as db:
        result = await answer_question(
            "What's my first name?", user_id, db, llm_complete=fake_llm
        )

    assert result.sql == sql_from_llm
    assert result.rows == [{"first_name": "Ada"}]
    assert "Ada" in result.answer
    assert len(call_log) == 2

    await dispose_engine()
