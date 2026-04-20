"""Create the ORM schema against $DATABASE_URL (no Alembic).

Used by the `make db-init` target for local SQLite dev; the Docker path uses
`alembic upgrade head` against Postgres instead.
"""
from __future__ import annotations

import asyncio
import os

import app.models  # noqa: F401  -- registers tables on Base.metadata
from app.database import Base, dispose_engine, init_engine


async def main() -> None:
    url = os.environ.get("DATABASE_URL") or "sqlite+aiosqlite:///./fitpace.db"
    engine = init_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await dispose_engine()
    print(f"schema created at {url}")


if __name__ == "__main__":
    asyncio.run(main())
