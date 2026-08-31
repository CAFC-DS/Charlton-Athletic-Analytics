"""Authentication helpers for the Streamlit analyst login wall."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import unicodedata
from typing import Any


AUTH_BUILD = "auth-v2-2026-08-31"


class AuthConfigError(ValueError):
    """Raised when the app's authentication secrets are missing or invalid."""


@dataclass(frozen=True)
class AuthConfig:
    """Validated credentials loaded from Streamlit secrets."""

    username: str
    password: str = field(repr=False)
    revision: str = field(repr=False)


def _normalize_username(value: str) -> str:
    """Make usernames forgiving without weakening password matching."""

    return unicodedata.normalize("NFKC", value).strip().casefold()


def _normalize_password(value: str) -> str:
    """Ignore accidental outer whitespace while preserving password content."""

    return value.strip()


def _required_string(section: Any, key: str) -> str:
    try:
        value = section[key]
    except Exception as exc:
        raise AuthConfigError(f"missing auth.{key}") from exc

    if not isinstance(value, str):
        raise AuthConfigError(f"auth.{key} must be a quoted TOML string")

    value = value.strip()
    if not value:
        raise AuthConfigError(f"auth.{key} cannot be empty")
    return value


def load_auth_config(secrets: Any) -> AuthConfig:
    """Load and validate the single analyst account from Streamlit secrets."""

    try:
        section = secrets["auth"]
    except Exception as exc:
        raise AuthConfigError("missing [auth] section") from exc

    username = _required_string(section, "username")
    password = _required_string(section, "password")
    revision_material = (
        f"{_normalize_username(username)}\0{_normalize_password(password)}".encode("utf-8")
    )
    revision = hashlib.sha256(revision_material).hexdigest()
    return AuthConfig(username=username, password=password, revision=revision)


def credentials_match(config: AuthConfig, username: str, password: str) -> bool:
    """Compare submitted credentials in constant time."""

    submitted_username = _normalize_username(username).encode("utf-8")
    expected_username = _normalize_username(config.username).encode("utf-8")
    submitted_password = _normalize_password(password).encode("utf-8")
    expected_password = _normalize_password(config.password).encode("utf-8")

    return hmac.compare_digest(submitted_username, expected_username) and hmac.compare_digest(
        submitted_password,
        expected_password,
    )
