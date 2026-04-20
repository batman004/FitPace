"""User request/response schemas."""
from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.enums import Sex


class UserCreate(BaseModel):
    """Signup payload. `password` is plain text in transit only; the server
    immediately hashes it before persisting."""

    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    date_of_birth: date | None = None
    height_cm: float | None = Field(default=None, gt=0, lt=300)
    weight_kg: float | None = Field(default=None, gt=0, lt=500)
    sex: Sex | None = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRead(BaseModel):
    """Public user representation — never includes the password hash."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    first_name: str
    last_name: str
    email: EmailStr
    date_of_birth: date | None
    height_cm: float | None
    weight_kg: float | None
    sex: Sex | None
    created_at: datetime
