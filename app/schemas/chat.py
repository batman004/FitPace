"""Pydantic schemas for the /chat endpoint."""
from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: UUID = Field(..., description="Whose data to query.")
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural-language question, e.g. 'Am I on track for my weight goal?'",
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Natural-language reply from the coach.")
    sql: str = Field(..., description="The SELECT statement the LLM generated.")
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rows that SQL returned, used as context for the answer.",
    )
