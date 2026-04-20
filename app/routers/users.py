"""User endpoints: signup, login, fetch."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, UserRead
from app.security import hash_password, verify_password

router = APIRouter(prefix="/users", tags=["users"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=UserRead,
    summary="Sign up a new user",
)
async def create_user(
    payload: UserCreate, db: AsyncSession = Depends(get_db)
) -> User:
    user = User(
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        date_of_birth=payload.date_of_birth,
        height_cm=payload.height_cm,
        weight_kg=payload.weight_kg,
        sex=payload.sex,
    )
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


@router.post(
    "/login",
    response_model=UserRead,
    summary="Verify credentials and return the user profile",
)
async def login(payload: UserLogin, db: AsyncSession = Depends(get_db)) -> User:
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()
    # Run verify_password either way to make response time uniform whether the
    # email exists or not (mitigates trivial email enumeration).
    dummy_hash = "$".join(["pbkdf2_sha256", "1", "AA==", "AA=="])
    stored = user.password_hash if user else dummy_hash
    valid = verify_password(payload.password, stored)
    if user is None or not valid:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="invalid email or password"
        )
    return user


@router.get("/{user_id}", response_model=UserRead)
async def get_user(user_id: UUID, db: AsyncSession = Depends(get_db)) -> User:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="user not found")
    return user
