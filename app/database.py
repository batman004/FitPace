"""Async SQLAlchemy engine, session factory, and FastAPI dependency."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(url: str, **engine_kwargs: Any) -> AsyncEngine:
    """Build the global async engine + session factory. Safe to call multiple times."""
    global _engine, _sessionmaker
    _engine = create_async_engine(url, **engine_kwargs)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        from app.config import get_settings

        init_engine(get_settings().database_url)
    assert _engine is not None
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a short-lived AsyncSession."""
    sm = get_sessionmaker()
    async with sm() as session:
        yield session


async def dispose_engine() -> None:
    """Dispose the engine (call from app shutdown and from tests)."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
