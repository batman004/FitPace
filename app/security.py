"""Password hashing using stdlib PBKDF2-SHA256.

No third-party deps. Storage format:
    pbkdf2_sha256$<iterations>$<base64 salt>$<base64 derived key>
"""
from __future__ import annotations

import hashlib
import hmac
import os
from base64 import b64decode, b64encode

_ALGO = "pbkdf2_sha256"
# OWASP 2023 minimum recommendation for PBKDF2-SHA256.
_ITERATIONS = 600_000
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("password must be non-empty")
    salt = os.urandom(_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return "$".join(
        [
            _ALGO,
            str(_ITERATIONS),
            b64encode(salt).decode("ascii"),
            b64encode(derived).decode("ascii"),
        ]
    )


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verify. Returns False for any malformed input."""
    try:
        algo, iters_str, salt_b64, key_b64 = stored.split("$")
    except ValueError:
        return False
    if algo != _ALGO:
        return False
    try:
        iterations = int(iters_str)
        salt = b64decode(salt_b64)
        expected = b64decode(key_b64)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return hmac.compare_digest(expected, actual)
