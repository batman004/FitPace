"""Tests for PBKDF2 password hashing."""
from __future__ import annotations

import pytest

from app.security import hash_password, verify_password


def test_hash_and_verify_roundtrip() -> None:
    stored = hash_password("correct-horse-battery-staple")
    assert verify_password("correct-horse-battery-staple", stored) is True


def test_verify_rejects_wrong_password() -> None:
    stored = hash_password("letmein-please")
    assert verify_password("nope-nope-nope", stored) is False


def test_hash_is_unique_per_call() -> None:
    assert hash_password("hello-world") != hash_password("hello-world")


def test_hash_rejects_empty_password() -> None:
    with pytest.raises(ValueError):
        hash_password("")


def test_verify_rejects_malformed_stored_value() -> None:
    assert verify_password("anything", "not-a-valid-format") is False
    assert verify_password("anything", "pbkdf2_sha256$bad$parts") is False
