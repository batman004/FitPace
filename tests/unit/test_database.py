"""Smoke tests for the async engine + session factory against aiosqlite."""
from __future__ import annotations

import pytest
from sqlalchemy import Integer, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app import database
from app.database import Base, dispose_engine, get_db, get_sessionmaker, init_engine


class _Widget(Base):
    """Throwaway model defined only inside this test module."""

    __tablename__ = "_test_widgets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)


@pytest.fixture(autouse=True)
async def _fresh_engine() -> None:
    await dispose_engine()
    engine = init_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await dispose_engine()


async def test_sessionmaker_roundtrip() -> None:
    sm = get_sessionmaker()
    async with sm() as session:
        session.add(_Widget(name="alpha"))
        await session.commit()

    async with sm() as session:
        result = await session.execute(select(_Widget).where(_Widget.name == "alpha"))
        widget = result.scalar_one()
        assert widget.name == "alpha"
        assert widget.id == 1


async def test_get_db_yields_session() -> None:
    gen = get_db()
    session = await anext(gen)
    assert isinstance(session, AsyncSession)
    session.add(_Widget(name="beta"))
    await session.commit()

    with pytest.raises(StopAsyncIteration):
        await anext(gen)


async def test_init_engine_replaces_previous() -> None:
    first = database.get_engine()
    second = init_engine("sqlite+aiosqlite:///:memory:")
    assert second is not first
    assert database.get_engine() is second
