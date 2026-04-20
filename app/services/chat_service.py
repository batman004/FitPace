"""Natural-language chat over the fitness database.

Two-step pipeline per user question:

    1. `generate_sql`  – ask the LLM for a single, SELECT-only SQL statement
                         scoped to the current user's data.
    2. `execute_sql`   – validate + run against the DB, cap rows.
    3. `compose_answer`– ask the LLM again with the rows as context to produce
                         a friendly natural-language reply.

The LLM interface is a simple `Callable[[str, str], str]` that takes a
(system_prompt, user_prompt) pair and returns the raw completion text. This
keeps the service trivially mockable in unit tests and lets us swap OpenAI
for another provider later.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings

LLMComplete = Callable[[str, str], Awaitable[str]]

MAX_ROWS = 50

SCHEMA_DOC = """\
You are querying a fitness-tracking database with the following tables:

users(
    id UUID PRIMARY KEY,
    first_name TEXT, last_name TEXT, email TEXT,
    date_of_birth DATE, height_cm FLOAT, weight_kg FLOAT,
    sex TEXT,            -- 'male' | 'female' | 'other' | 'prefer_not_to_say'
    created_at TIMESTAMP
)

goals(
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    goal_type TEXT,      -- 'weight_loss' | 'strength_gain' | 'step_goal'
    start_value FLOAT, target_value FLOAT, unit TEXT,
    start_date DATE, target_date DATE,
    current_state TEXT,  -- 'ON_TRACK' | 'AT_RISK' | 'OFF_TRACK' | 'RECOVERED'
    created_at TIMESTAMP
)

progress_logs(
    id UUID PRIMARY KEY,
    goal_id UUID REFERENCES goals(id),
    logged_at TIMESTAMP, value FLOAT, notes TEXT
)

goal_state_events(
    id UUID PRIMARY KEY,
    goal_id UUID REFERENCES goals(id),
    from_state TEXT, to_state TEXT, pace_score FLOAT,
    occurred_at TIMESTAMP, reason TEXT
)
"""

SQL_SYSTEM_PROMPT = (
    SCHEMA_DOC
    + """

Rules for the SQL you emit:
- Output ONLY a single SQL statement. No markdown, no comments, no prose.
- The statement MUST be a SELECT. Never INSERT/UPDATE/DELETE/DROP/ALTER/CREATE.
- Filter to the current user by joining on goals.user_id = :user_id or users.id = :user_id.
- Use the bind parameter :user_id - never inline the UUID.
- Limit to at most {max_rows} rows with `LIMIT {max_rows}`.
- Prefer aggregating (MIN/MAX/AVG/COUNT) or ordering + limiting when the question asks
  for a single value ("current weight", "latest log", "how many goals").
- When the question concerns body composition, health, BMI, fitness level, diet,
  or "how am I doing", select the full relevant profile at once -
  height_cm, weight_kg, date_of_birth, and sex from users - plus, when useful,
  the user's active goals and recent progress_logs. A richer row lets the
  answer reference multiple signals instead of a single number in isolation.
"""
).replace("{max_rows}", str(MAX_ROWS))

ANSWER_SYSTEM_PROMPT = """\
You are FitPace, a concise and encouraging fitness coach.

You will receive the user's question plus a JSON array of rows retrieved from
their own fitness database. Answer using ONLY those rows. If the rows are empty
or insufficient, say so honestly rather than speculating. Keep the reply to 1-4
short sentences unless the user explicitly asked for a list.

If the rows contain several profile fields (e.g. height_cm + weight_kg,
date_of_birth, sex, goal progress, recent logs) use them together:
- Compute BMI (weight_kg / (height_cm/100)^2) when both are present and the
  question is about body composition or health; call out the standard bands
  (<18.5 underweight, 18.5-24.9 normal, 25-29.9 overweight, >=30 obese) while
  reminding the user BMI alone doesn't capture muscle mass or body composition.
- Derive age from date_of_birth if provided.
- Cross-reference active goals and recent progress logs when explaining pace
  or progress ("you're on track for your weight-loss goal, logging X kg last
  week").
Never invent data that isn't in the rows; if a field needed for the answer is
missing, tell the user which data would help.
"""


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|attach|"
    r"detach|pragma|vacuum|replace|merge)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    """Raised when the LLM-generated SQL fails validation."""


@dataclass
class ChatResult:
    answer: str
    sql: str
    rows: list[dict[str, Any]]


def _strip_fences(sql: str) -> str:
    s = sql.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"```\s*$", "", s)
    return s.strip().rstrip(";").strip()


def validate_sql(sql: str) -> str:
    """Enforce SELECT-only, single-statement SQL scoped to the current user.

    Checks applied, in order:
      1. Non-empty after fence stripping.
      2. No multiple statements (no interior semicolons).
      3. First token is SELECT or WITH.
      4. No forbidden keywords (INSERT/UPDATE/DELETE/DROP/ALTER/...).
      5. MUST reference the :user_id bind parameter so every query is scoped
         to the caller. Without this a compliant SELECT could still leak
         another user's rows (e.g. `SELECT weight_kg FROM users LIMIT 1`).
    """
    cleaned = _strip_fences(sql)
    if not cleaned:
        raise UnsafeSQLError("empty SQL")
    if ";" in cleaned:
        raise UnsafeSQLError("multiple statements are not allowed")
    first = cleaned.split(None, 1)[0].lower()
    if first not in {"select", "with"}:
        raise UnsafeSQLError(f"only SELECT/WITH queries are allowed, got '{first}'")
    if _FORBIDDEN.search(cleaned):
        raise UnsafeSQLError("SQL contains a forbidden keyword")
    if ":user_id" not in cleaned:
        raise UnsafeSQLError(
            "query must be scoped to the caller via the :user_id bind parameter"
        )
    return cleaned


async def execute_sql(
    db: AsyncSession, sql: str, user_id: UUID
) -> list[dict[str, Any]]:
    """Run the validated SQL with :user_id bound. Returns up to MAX_ROWS rows.

    `validate_sql` guarantees `:user_id` is present, so we can always bind it
    with the PG UUID type (works on Postgres native uuid + SQLite char(32)).
    """
    statement = text(sql).bindparams(
        bindparam("user_id", type_=PGUUID(as_uuid=True))
    )
    result = await db.execute(statement, {"user_id": user_id})
    rows: list[dict[str, Any]] = []
    for row in result.mappings():
        rows.append({k: _jsonable(v) for k, v in row.items()})
        if len(rows) >= MAX_ROWS:
            break
    return rows


def _jsonable(value: Any) -> Any:
    """Coerce DB values (UUID, datetime, Decimal, ...) into JSON-safe types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _openai_complete_factory() -> LLMComplete:
    """Build an async LLMComplete that calls OpenAI chat completions."""
    from openai import AsyncOpenAI

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def _complete(system: str, user: str) -> str:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    return _complete


def default_llm() -> LLMComplete:
    """Return the configured LLM callable, or raise if no provider is set."""
    settings = get_settings()
    if settings.openai_api_key:
        return _openai_complete_factory()
    raise RuntimeError(
        "No LLM provider configured. Set OPENAI_API_KEY to enable /chat."
    )


async def answer_question(
    question: str,
    user_id: UUID,
    db: AsyncSession,
    llm_complete: LLMComplete | None = None,
) -> ChatResult:
    """End-to-end: NL question -> SQL -> rows -> NL answer."""
    complete = llm_complete or default_llm()

    sql_raw = await complete(
        SQL_SYSTEM_PROMPT,
        f"User id: {user_id}\nQuestion: {question}\nSQL:",
    )
    sql = validate_sql(sql_raw)

    rows = await execute_sql(db, sql, user_id)

    answer = await complete(
        ANSWER_SYSTEM_PROMPT,
        (
            f"Question: {question}\n\n"
            f"Data rows (JSON):\n{json.dumps(rows, default=str)}\n\n"
            "Answer:"
        ),
    )
    return ChatResult(answer=answer.strip(), sql=sql, rows=rows)
