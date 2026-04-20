"""Tests for GET /health."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import MODEL_PATH, app


@pytest.mark.asyncio
async def test_health_returns_ok_and_model_flag() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is MODEL_PATH.is_file()
