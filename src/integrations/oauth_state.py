"""Helpers for signed OAuth state payloads."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict


class OAuthStateError(ValueError):
    """Raised when OAuth state is invalid or expired."""


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def sign_state(payload: Dict[str, Any], secret: str) -> str:
    """Sign an OAuth state payload and return the compact string."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return f"{_b64encode(body)}.{_b64encode(sig)}"


def verify_state(state: str, secret: str, ttl_seconds: int) -> Dict[str, Any]:
    """Verify signed OAuth state and return the payload."""
    if not state or "." not in state:
        raise OAuthStateError("Missing or malformed state")

    body_b64, sig_b64 = state.split(".", 1)
    try:
        body = _b64decode(body_b64)
        sig = _b64decode(sig_b64)
    except Exception as exc:
        raise OAuthStateError("Invalid state encoding") from exc

    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise OAuthStateError("Invalid state signature")

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise OAuthStateError("Invalid state payload") from exc

    ts = payload.get("ts")
    if ts is None:
        raise OAuthStateError("State missing timestamp")

    age = time.time() - float(ts)
    if age < 0 or age > ttl_seconds:
        raise OAuthStateError("State expired")

    return payload
