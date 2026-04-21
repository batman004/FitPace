"""Natural-language chat endpoint backed by an LLM + SQL retrieval."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from openai import OpenAIError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import (
    LLMComplete,
    UnsafeSQLError,
    answer_question,
    default_llm,
)

router = APIRouter(prefix="/chat", tags=["chat"])


def get_llm() -> LLMComplete:
    """Resolve the LLM callable. Tests override this to inject a stub."""
    try:
        return default_llm()
    except RuntimeError as exc:
        logger.warning("chat unavailable: {}", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    llm: LLMComplete = Depends(get_llm),
) -> ChatResponse:
    try:
        result = await answer_question(payload.question, payload.user_id, db, llm_complete=llm)
    except UnsafeSQLError as exc:
        # chat_service.answer_question already logs the raw SQL before raising;
        # the router just translates to a 422.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"generated SQL was rejected: {exc}",
        )
    except SQLAlchemyError as exc:
        logger.exception("chat SQL failed to execute user_id={}", payload.user_id)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"generated SQL failed to execute: {exc}",
        )
    except OpenAIError as exc:
        # Rate limits, auth errors, connection failures, timeouts, etc.
        logger.exception("chat LLM provider error user_id={}", payload.user_id)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM provider error: {exc}",
        )
    return ChatResponse(answer=result.answer, sql=result.sql, rows=result.rows)
