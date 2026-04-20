"""User endpoints."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserRead

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=UserRead)
async def create_user(
    payload: UserCreate, db: AsyncSession = Depends(get_db)
) -> User:
    user = User(name=payload.name, email=payload.email)
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="email already registered"
        )
    await db.refresh(user)
    return user


@router.get("/{user_id}", response_model=UserRead)
async def get_user(user_id: UUID, db: AsyncSession = Depends(get_db)) -> User:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="user not found")
    return user
