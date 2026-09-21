"""Deterministic settings for the existing Streamlit process."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from phd_agent.paths import APP_ROOT


# Review intervals are prompts to recheck sources, not guarantees of truth.
FRESHNESS_DAYS: Mapping[str, int] = {
    "FACULTY_AFFILIATION": 45,
    "FACULTY_EMAIL": 45,
    "OPPORTUNITY_OPENING": 3,
    "DEADLINE": 7,
    "PROGRAMME_REQUIREMENT": 14,
    "CONTACT_POLICY": 14,
    "PUBLICATION": 30,
}

MATCH_WEIGHTS: Mapping[str, float] = {
    "topic": 0.30,
    "method": 0.20,
    "recent_work": 0.20,
    "experience": 0.20,
    "proposed_direction": 0.10,
}

GOOGLE_OIDC_METADATA = "https://accounts.google.com/.well-known/openid-configuration"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _emails(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in (value or "").split(",") if part.strip())


def hosted_from_environ(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    if (env.get("PHD_AGENT_ENV") or "").strip().lower() == "production":
        return True
    return bool(
        (env.get("RAILWAY_ENVIRONMENT") or "").strip()
        or (env.get("RAILWAY_PROJECT_ID") or "").strip()
        or (env.get("RAILWAY_SERVICE_ID") or "").strip()
    )


def refuse_hosted_scripts(environ: Mapping[str, str] | None = None) -> None:
    if hosted_from_environ(environ):
        raise SystemExit("Demo and local setup scripts cannot run in production or on Railway")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    environment: str = "development"
    railway: bool = False
    allowed_emails: tuple[str, ...] = ()
    auth_disabled_requested: bool = False

    @property
    def database_path(self) -> Path:
        return self.data_dir / "phd_outreach.db"

    @property
    def cv_path(self) -> Path:
        return self.data_dir / "documents" / "uploaded_cv.pdf"

    @property
    def profile_path(self) -> Path:
        return self.data_dir / "research_profile.txt"

    @property
    def credentials_path(self) -> Path:
        return self.data_dir / "credentials.json"

    @property
    def gmail_token_path(self) -> Path:
        return self.data_dir / "gmail_token.json"

    @property
    def targets_path(self) -> Path:
        return self.data_dir / "PhD_Targets.csv"

    @property
    def results_path(self) -> Path:
        return self.data_dir / "PhD_Results.csv"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "phd_outreach.log"

    @property
    def hosted(self) -> bool:
        return self.railway or self.environment == "production"

    @property
    def auth_bypass(self) -> bool:
        return self.auth_disabled_requested and not self.hosted

    @property
    def require_auth(self) -> bool:
        return not self.auth_bypass

    @property
    def interactive_oauth_allowed(self) -> bool:
        return not self.hosted

    @property
    def auto_send_enabled(self) -> bool:
        # Campaign automation needs the later quality gate and contact history.
        # Do not enable the legacy bulk send paths through an environment flag.
        return False


def load_settings(root: Path = APP_ROOT, environ: Mapping[str, str] | None = None) -> Settings:
    if environ is None:
        load_dotenv(root / ".env", override=False)
        env: Mapping[str, str] = os.environ
    else:
        env = environ
    environment = (env.get("PHD_AGENT_ENV") or "development").strip().lower()
    if environment not in {"development", "production"}:
        environment = "development"
    railway = bool(
        (env.get("RAILWAY_ENVIRONMENT") or "").strip()
        or (env.get("RAILWAY_PROJECT_ID") or "").strip()
        or (env.get("RAILWAY_SERVICE_ID") or "").strip()
    )
    hosted = railway or environment == "production"
    configured = (env.get("PHD_AGENT_DATA_DIR") or "").strip()
    if hosted:
        if not configured:
            raise ValueError("Hosted deployments require an absolute PHD_AGENT_DATA_DIR")
        data_dir = Path(configured).expanduser()
        if not data_dir.is_absolute():
            raise ValueError("PHD_AGENT_DATA_DIR must be an absolute persistent directory")
    else:
        data_dir = Path(configured).expanduser() if configured else root / "data"
        if not data_dir.is_absolute():
            data_dir = root / data_dir
    resolved = data_dir.resolve()
    if resolved.is_relative_to(root.resolve()) and not resolved.is_relative_to((root / "data").resolve()):
        raise ValueError("PHD_AGENT_DATA_DIR inside the repository must be under ignored data/")
    return Settings(
        data_dir=resolved,
        environment=environment,
        railway=railway,
        allowed_emails=_emails(env.get("PHD_AGENT_ALLOWED_EMAILS")),
        auth_disabled_requested=_truthy(env.get("PHD_AGENT_AUTH_DISABLED")),
    )
