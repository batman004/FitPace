"""End-to-end demo of the FitPace API.

Walks through every public endpoint in a realistic order so a new user can
see exactly how to drive the service:

    1.  GET  /health                        (is the API up + model loaded?)
    2.  POST /users                         (signup with profile fields)
    3.  POST /users/login                   (verify password hashing works)
    4.  GET  /users/{id}                    (read back the profile)
    5.  POST /goals                         (weight-loss goal, 60-day window)
    6.  GET  /goals/{id}/trajectory         (pre-log: neutral fallback)
    7.  POST /progress    x N               (seed 10 on-pace daily logs)
    8.  GET  /progress/{goal_id}            (list logs)
    9.  GET  /goals/{id}/trajectory         (post-log: real ML projection)
    10. GET  /goals/{id}/history            (state-machine transitions)

Run the API first (e.g. `make run`) and then `make demo` or
`python scripts/demo.py`. Re-runnable: each execution uses a randomized
email so signup never collides.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import date, datetime, timedelta, timezone

import httpx

BASE_URL = os.getenv("FITPACE_URL", "http://127.0.0.1:8001")
PASSWORD = "correct horse battery staple"


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _call(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    expected: int = 200,
) -> dict | list:
    print(f"\n>>> {method} {path}")
    if json_body is not None:
        print("    payload:", json.dumps(json_body, default=str))
    resp = client.request(method, path, json=json_body)
    body: dict | list
    try:
        body = resp.json()
    except ValueError:
        body = {"raw": resp.text}
    print(f"    <- {resp.status_code}")
    print("    " + json.dumps(body, indent=2, default=str).replace("\n", "\n    "))
    if resp.status_code != expected:
        raise SystemExit(
            f"expected HTTP {expected}, got {resp.status_code} for {method} {path}"
        )
    return body


def main() -> None:
    suffix = uuid.uuid4().hex[:8]
    email = f"demo+{suffix}@fitpace.dev"

    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        _section("1. Health check")
        _call(client, "GET", "/health")

        _section("2. Sign up a new user")
        user = _call(
            client,
            "POST",
            "/users",
            json_body={
                "first_name": "Ada",
                "last_name": "Lovelace",
                "email": email,
                "password": PASSWORD,
                "date_of_birth": "1990-12-10",
                "height_cm": 168.0,
                "weight_kg": 82.0,
                "sex": "female",
            },
            expected=201,
        )
        assert isinstance(user, dict)
        user_id = user["id"]

        _section("3. Log in with that user")
        _call(
            client,
            "POST",
            "/users/login",
            json_body={"email": email, "password": PASSWORD},
        )

        _section("4. Fetch the user profile")
        _call(client, "GET", f"/users/{user_id}")

        _section("5. Create a realistic 60-day weight-loss goal")
        start = date.today()
        target = start + timedelta(days=60)
        goal = _call(
            client,
            "POST",
            "/goals",
            json_body={
                "user_id": user_id,
                "goal_type": "weight_loss",
                "start_value": 82.0,
                "target_value": 76.0,
                "unit": "kg",
                "start_date": start.isoformat(),
                "target_date": target.isoformat(),
            },
            expected=201,
        )
        assert isinstance(goal, dict)
        goal_id = goal["id"]

        _section("6. Trajectory with zero logs (neutral fallback)")
        print(
            "    Expect pace_score=50, eta_date=target_date: the service needs "
            ">=2 logs\n    before it can fit a slope."
        )
        _call(client, "GET", f"/goals/{goal_id}/trajectory")

        _section("7. Post 10 on-pace daily progress logs")
        total_delta = 76.0 - 82.0  # -6 kg over 60 days
        for i in range(10):
            logged_at = datetime.combine(
                start + timedelta(days=i),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
            ideal = 82.0 + total_delta * (i / 60.0)
            _call(
                client,
                "POST",
                "/progress",
                json_body={
                    "goal_id": goal_id,
                    "logged_at": logged_at.isoformat().replace("+00:00", "Z"),
                    "value": round(ideal, 2),
                    "notes": f"day {i + 1}",
                },
                expected=201,
            )

        _section("8. List all logs for this goal")
        _call(client, "GET", f"/progress/{goal_id}")

        _section("9. Trajectory after 10 logs (ML-backed projection)")
        print(
            "    Now pace_score should be >50 (on pace), eta_date should be a real\n"
            "    date, and days_ahead reflects how much sooner/later than the target."
        )
        _call(client, "GET", f"/goals/{goal_id}/trajectory")

        _section("10. Goal state history")
        print(
            "    Each row here is a transition emitted by the state machine when a\n"
            "    new log moves the goal between ON_TRACK / AT_RISK / OFF_TRACK /\n"
            "    RECOVERED. On a clean on-pace seed you may see zero transitions."
        )
        _call(client, "GET", f"/goals/{goal_id}/history")

        _section("11. Natural-language chat (/chat)")
        print(
            "    The LLM first generates a SELECT against the database, then uses\n"
            "    the returned rows as context to answer in plain English.\n"
            "    Requires OPENAI_API_KEY in the server's .env; otherwise returns 503."
        )
        chat_resp = client.post(
            "/chat",
            json={
                "user_id": user_id,
                "question": "Am I on track for my weight-loss goal? How many logs do I have?",
            },
        )
        print(f"    <- {chat_resp.status_code}")
        if chat_resp.status_code == 503:
            print("    (skipped: set OPENAI_API_KEY in .env and restart the API)")
        else:
            print(
                "    "
                + json.dumps(chat_resp.json(), indent=2, default=str).replace(
                    "\n", "\n    "
                )
            )

        _section("Done")
        print(f"    User:  {user_id}")
        print(f"    Email: {email}  (password: {PASSWORD!r})")
        print(f"    Goal:  {goal_id}")
        print(f"    Explore interactively: {BASE_URL}/docs  or  {BASE_URL}/scalar")


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError as exc:
        print(f"\nCould not reach {BASE_URL}. Is the API running? ({exc})")
        sys.exit(1)
