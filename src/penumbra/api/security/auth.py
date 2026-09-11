"""Authentication.

JWT bearer tokens over a small user store. Deliberately simple: the interesting access-control work
is in `rbac.py`, and an elaborate auth implementation here would add risk without adding evidence.

Passwords are hashed with PBKDF2-HMAC-SHA256 from the standard library rather than bcrypt, to keep
the dependency surface small. The iteration count is set high enough to matter; for a production
deployment this would be argon2id, and saying so is more useful than pretending the choice was
free.

The demo users exist so the console has something to log into. They are created only when
`PENUMBRA_ALLOW_DEMO_USERS` is set, which the Docker image does not set.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from penumbra.api.security.rbac import Principal, Role

ALGORITHM = "HS256"
TOKEN_TTL = timedelta(hours=8)  # one shift
_PBKDF2_ROUNDS = 240_000


def _secret() -> str:
    """JWT signing secret.

    Generated per-process if unset, which means tokens do not survive a restart. That is the right
    default for a dev server and the wrong one for a deployment, so production must set
    PENUMBRA_JWT_SECRET explicitly rather than inheriting a value that silently works.
    """
    return os.environ.get("PENUMBRA_JWT_SECRET") or _EPHEMERAL_SECRET


_EPHEMERAL_SECRET = secrets.token_urlsafe(48)


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        _, rounds, salt_hex, digest_hex = encoded.split("$")
        expected = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        )
    except (ValueError, TypeError):
        return False
    # compare_digest rather than == so a wrong password does not leak its prefix length via timing.
    return hmac.compare_digest(expected.hex(), digest_hex)


@dataclass
class User:
    username: str
    password_hash: str
    role: Role
    segments: frozenset[str] = frozenset()

    def principal(self) -> Principal:
        return Principal(username=self.username, role=self.role, segments=self.segments)


class UserStore:
    """In-memory user store. A real deployment swaps this for a directory integration."""

    def __init__(self) -> None:
        self._users: dict[str, User] = {}

    def add(self, username: str, password: str, role: Role, segments: frozenset[str] = frozenset()) -> User:
        user = User(username, hash_password(password), role, segments)
        self._users[username] = user
        return user

    def authenticate(self, username: str, password: str) -> User | None:
        user = self._users.get(username)
        if user is None:
            # Hash anyway so a missing user and a wrong password take the same time. Otherwise the
            # endpoint is a username oracle.
            hash_password(password)
            return None
        return user if verify_password(password, user.password_hash) else None

    def get(self, username: str) -> User | None:
        return self._users.get(username)

    def seed_demo_users(self) -> None:
        """Three users, one per role, for the console demo.

        Gated on an environment variable so an image built from this repo does not ship with known
        credentials.
        """
        if not os.environ.get("PENUMBRA_ALLOW_DEMO_USERS"):
            return
        self.add("analyst", "analyst", Role.ANALYST, frozenset({"dmz", "corp"}))
        self.add("senior", "senior", Role.SENIOR)
        self.add("admin", "admin", Role.ADMIN)


def issue_token(user: User, *, ttl: timedelta = TOKEN_TTL) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": user.username,
        "role": user.role.value,
        "segments": sorted(user.segments),
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


class InvalidToken(ValueError):
    pass


def decode_token(token: str) -> Principal:
    try:
        payload: dict[str, Any] = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise InvalidToken("token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise InvalidToken("invalid token") from exc

    try:
        role = Role(payload["role"])
    except (KeyError, ValueError) as exc:
        raise InvalidToken("token carries no valid role") from exc

    return Principal(
        username=str(payload.get("sub", "")),
        role=role,
        segments=frozenset(payload.get("segments") or []),
    )
