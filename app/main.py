"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import get_settings
from app.database import dispose_engine, init_engine
from app.logging_config import configure_logging
from app.routers import chat, goals, progress, users

# Register ORM models on Base.metadata.
import app.models  # noqa: F401

MODEL_PATH = Path(__file__).resolve().parent / "ml" / "model.pkl"
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    settings = get_settings()
    # Redact credentials from the DB URL before logging. "driver://user:pass@host/db"
    # -> "driver://host/db".
    safe_url = settings.database_url
    if "@" in safe_url:
        scheme, _, rest = safe_url.partition("://")
        _, _, host_and_path = rest.partition("@")
        safe_url = f"{scheme}://{host_and_path}"
    logger.info(
        "FitPace starting: db={} model_loaded={} scheduler_enabled={} chat_llm={}",
        safe_url,
        MODEL_PATH.is_file(),
        settings.scheduler_enabled,
        "openai" if settings.openai_api_key else "unset",
    )
    init_engine(settings.database_url)
    yield
    logger.info("FitPace shutting down")
    await dispose_engine()


app = FastAPI(title="FitPace API", lifespan=lifespan)

app.include_router(users.router)
app.include_router(goals.router)
app.include_router(progress.router)
app.include_router(chat.router)


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.is_file(),
    }


# Lightweight demo UI bundled under /ui. Mounted last so it never shadows the
# API routers. `html=True` lets StaticFiles serve index.html for '/' in the
# mount and for unknown subpaths, which keeps deep-linking simple.
if WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/")
