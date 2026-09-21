"""Allowlist checks for hosted Streamlit access. Identity values are untrusted input."""

from __future__ import annotations

from phd_agent.config import Settings


def normalize_email(value: str | None) -> str | None:
    email = (value or "").strip().lower()
    return email or None


def email_allowed(email: str | None, settings: Settings) -> bool:
    normalized = normalize_email(email)
    return bool(normalized and normalized in set(settings.allowed_emails))


def access_state(*, authenticated: bool, email: str | None, settings: Settings) -> str:
    if settings.auth_bypass:
        return "allowed"
    if not authenticated:
        return "unauthenticated"
    if email_allowed(email, settings):
        return "allowed"
    return "unauthorized"
