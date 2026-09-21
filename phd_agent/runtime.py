"""Hosted startup checks. Do not create the production database during image build."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Mapping

from phd_agent.config import GOOGLE_OIDC_METADATA, Settings, load_settings
from phd_agent.db import connect, migrate
from phd_agent.paths import APP_ROOT, ensure_data_layout
from phd_agent.security import GitInspectionFailed, tracked_private_paths, write_restricted_file


RUNTIME_SUBDIRS = ("documents", "backups", "exports")
OIDC_ENV = (
    "PHD_AGENT_OIDC_CLIENT_ID",
    "PHD_AGENT_OIDC_CLIENT_SECRET",
    "PHD_AGENT_OIDC_COOKIE_SECRET",
    "PHD_AGENT_OIDC_REDIRECT_URI",
)


class StartupError(RuntimeError):
    """The process cannot start safely with the current host configuration."""


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _writable(directory: Path) -> None:
    probe = directory / ".phd_agent_write_check"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        raise StartupError("Persistent data directory is not writable") from error


LOG_FILE_HANDLER = "phd_agent.runtime.file"


def volume_unavailable(data_dir: Path, environ: Mapping[str, str]) -> bool:
    if not data_dir.exists() or not data_dir.is_dir():
        return True
    railway = bool(
        (environ.get("RAILWAY_ENVIRONMENT") or "").strip()
        or (environ.get("RAILWAY_PROJECT_ID") or "").strip()
    )
    if not railway:
        return False
    resolved = data_dir.resolve()
    if resolved != Path("/data"):
        return False
    mounts = Path("/proc/mounts")
    if not mounts.is_file():
        return False
    target = str(resolved).rstrip("/")
    for line in mounts.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].rstrip("/") == target:
            return False
    return True


def materialize_gmail_files(settings: Settings, environ: Mapping[str, str]) -> None:
    mapping = (
        ("PHD_AGENT_GMAIL_CREDENTIALS_JSON", settings.credentials_path),
        ("PHD_AGENT_GMAIL_TOKEN_JSON", settings.gmail_token_path),
    )
    for key, destination in mapping:
        payload = environ.get(key)
        if not payload or not payload.strip():
            continue
        if destination.exists():
            continue
        write_restricted_file(destination, payload)


def write_oidc_secrets(destination: Path, environ: Mapping[str, str]) -> None:
    values = {
        "redirect_uri": (environ.get("PHD_AGENT_OIDC_REDIRECT_URI") or "").strip(),
        "cookie_secret": (environ.get("PHD_AGENT_OIDC_COOKIE_SECRET") or "").strip(),
        "client_id": (environ.get("PHD_AGENT_OIDC_CLIENT_ID") or "").strip(),
        "client_secret": (environ.get("PHD_AGENT_OIDC_CLIENT_SECRET") or "").strip(),
        "server_metadata_url": (
            (environ.get("PHD_AGENT_OIDC_SERVER_METADATA_URL") or "").strip()
            or GOOGLE_OIDC_METADATA
        ),
    }
    lines = ["[auth]"]
    for key, value in values.items():
        lines.append(f"{key} = {json.dumps(value)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_auth_config(settings: Settings, environ: Mapping[str, str]) -> None:
    if settings.auth_bypass:
        return
    if not settings.allowed_emails:
        raise StartupError("PHD_AGENT_ALLOWED_EMAILS is required when authentication is mandatory")
    missing = [key for key in OIDC_ENV if not (environ.get(key) or "").strip()]
    if missing:
        raise StartupError("OIDC configuration is incomplete")


def check_sqlite_integrity(database_path: Path) -> None:
    with connect(database_path) as db:
        row = db.execute("PRAGMA integrity_check").fetchone()
    result = row[0] if row else ""
    if str(result).lower() != "ok":
        raise StartupError("SQLite integrity check failed")


def _configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    if settings.hosted:
        root.setLevel(logging.WARNING)
    wanted = os.path.abspath(str(settings.log_path))
    existing = None
    for handler in list(root.handlers):
        if getattr(handler, "name", "") != LOG_FILE_HANDLER:
            continue
        if os.path.abspath(getattr(handler, "baseFilename", "")) == wanted:
            existing = handler
            continue
        root.removeHandler(handler)
        handler.close()
    if existing is not None:
        return
    try:
        handler = logging.FileHandler(settings.log_path, encoding="utf-8")
        handler.set_name(LOG_FILE_HANDLER)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root.addHandler(handler)
    except OSError:
        pass


def prepare_runtime(
    root: Path = APP_ROOT,
    environ: Mapping[str, str] | None = None,
    *,
    secrets_path: Path | None = None,
) -> Settings:
    env = _environ(environ)
    try:
        settings = load_settings(root, env if environ is not None else None)
    except ValueError as error:
        raise StartupError(str(error)) from error
    if settings.hosted:
        if volume_unavailable(settings.data_dir, env):
            raise StartupError(
                "Persistent data directory is unavailable. Mount the Railway volume before starting."
            )
        _writable(settings.data_dir)
        for name in RUNTIME_SUBDIRS:
            (settings.data_dir / name).mkdir(mode=0o700, exist_ok=True)
    else:
        ensure_data_layout(settings.data_dir, root)
        (settings.data_dir / "exports").mkdir(exist_ok=True)
        _writable(settings.data_dir)
    materialize_gmail_files(settings, env)
    migrate(settings.database_path)
    check_sqlite_integrity(settings.database_path)
    vault = settings.data_dir / "documents"
    if not vault.is_dir():
        raise StartupError("Document Vault directory is not accessible")
    _writable(vault)
    require_auth_config(settings, env)
    if settings.require_auth:
        write_oidc_secrets(secrets_path or (root / ".streamlit" / "secrets.toml"), env)
    try:
        tracked = tracked_private_paths(root)
    except GitInspectionFailed:
        tracked = []
    if tracked and settings.hosted:
        raise StartupError("Tracked runtime credentials must not ship with the deployment")
    if settings.auto_send_enabled:
        raise StartupError("Automatic sending cannot be enabled")
    _configure_logging(settings)
    return settings
