"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.config import get_settings
from app.database import dispose_engine, init_engine
from app.routers import goals, progress, users

# Register ORM models on Base.metadata.
import app.models  # noqa: F401

MODEL_PATH = Path(__file__).resolve().parent / "ml" / "model.pkl"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_engine(get_settings().database_url)
    yield
    await dispose_engine()


app = FastAPI(title="FitPace API", lifespan=lifespan)

app.include_router(users.router)
app.include_router(goals.router)
app.include_router(progress.router)


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.is_file(),
    }
