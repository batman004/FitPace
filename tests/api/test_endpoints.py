"""End-to-end API tests against an in-memory SQLite database."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

# Import so Base.metadata is populated before create_all.
import app.models  # noqa: F401
from app.database import Base, dispose_engine, init_engine
from app.main import app as fastapi_app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    await dispose_engine()
    engine = init_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    fastapi_app.dependency_overrides = {}
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await dispose_engine()


async def _seed_goal_on_pace(client: AsyncClient) -> tuple[str, str]:
    user_resp = await client.post(
        "/users", json={"name": "Ada", "email": "ada@example.com"}
    )
    assert user_resp.status_code == 201, user_resp.text
    user_id = user_resp.json()["id"]

    start_date = date(2026, 1, 1)
    target_date = start_date + timedelta(days=60)
    goal_resp = await client.post(
        "/goals",
        json={
            "user_id": user_id,
            "goal_type": "weight_loss",
            "start_value": 85.0,
            "target_value": 80.0,
            "unit": "kg",
            "start_date": start_date.isoformat(),
            "target_date": target_date.isoformat(),
        },
    )
    assert goal_resp.status_code == 201, goal_resp.text
    return user_id, goal_resp.json()["id"]


async def test_create_user_and_fetch(client: AsyncClient) -> None:
    resp = await client.post(
        "/users", json={"name": "Grace", "email": "grace@example.com"}
    )
    assert resp.status_code == 201
    user = resp.json()

    fetched = await client.get(f"/users/{user['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["email"] == "grace@example.com"


async def test_create_goal_defaults_on_track(client: AsyncClient) -> None:
    _, goal_id = await _seed_goal_on_pace(client)
    resp = await client.get(f"/goals/{goal_id}")
    assert resp.status_code == 200
    assert resp.json()["current_state"] == "ON_TRACK"


async def test_progress_post_triggers_recompute_and_trajectory(
    client: AsyncClient,
) -> None:
    _, goal_id = await _seed_goal_on_pace(client)

    start_date = date(2026, 1, 1)
    for day in range(10):
        value = 85.0 - (5.0 * day / 60.0)
        logged_at = datetime.combine(
            start_date + timedelta(days=day),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        resp = await client.post(
            "/progress",
            json={
                "goal_id": goal_id,
                "logged_at": logged_at.isoformat(),
                "value": value,
            },
        )
        assert resp.status_code == 201, resp.text

    logs_resp = await client.get(f"/progress/{goal_id}")
    assert logs_resp.status_code == 200
    assert len(logs_resp.json()) == 10

    traj_resp = await client.get(f"/goals/{goal_id}/trajectory")
    assert traj_resp.status_code == 200
    traj = traj_resp.json()
    assert 0.0 <= traj["pace_score"] <= 100.0
    assert traj["goal_id"] == goal_id

    history_resp = await client.get(f"/goals/{goal_id}/history")
    assert history_resp.status_code == 200
    assert isinstance(history_resp.json(), list)


async def test_unknown_goal_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/goals/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
