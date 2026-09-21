"""Railway deployment-readiness checks. No Google, Gmail, OpenAI, or Railway network calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gmail_manager
from phd_agent.access import access_state
from phd_agent.config import Settings, hosted_from_environ, load_settings, refuse_hosted_scripts
from phd_agent.db import connect
from phd_agent.documents import DocumentVault
from phd_agent.runtime import StartupError, prepare_runtime, write_oidc_secrets
from phd_agent.security import tracked_private_paths


ROOT = Path(__file__).resolve().parents[1]


def _hosted_env(data_dir: Path, **extra) -> dict[str, str]:
    env = {
        "PHD_AGENT_ENV": "production",
        "PHD_AGENT_DATA_DIR": str(data_dir),
        "PHD_AGENT_AUTH_DISABLED": "true",
        "PHD_AGENT_ALLOWED_EMAILS": "operator@example.com",
        "PHD_AGENT_OIDC_CLIENT_ID": "test-client-id",
        "PHD_AGENT_OIDC_CLIENT_SECRET": "test-client-secret",
        "PHD_AGENT_OIDC_COOKIE_SECRET": "test-cookie-secret",
        "PHD_AGENT_OIDC_REDIRECT_URI": "https://example.invalid/oauth2callback",
        "AUTO_SEND_ENABLED": "true",
    }
    env.update(extra)
    return env


def _prepare(tmp_path: Path, environ: dict[str, str], **kwargs):
    return prepare_runtime(ROOT, environ, secrets_path=tmp_path / "secrets.toml", **kwargs)


def test_production_auth_is_required(tmp_path):
    settings = load_settings(ROOT, _hosted_env(tmp_path))
    assert settings.require_auth is True
    assert settings.auth_bypass is False
    assert settings.hosted is True


def test_allowed_user_is_accepted(tmp_path):
    settings = load_settings(ROOT, _hosted_env(tmp_path))
    assert access_state(authenticated=True, email="Operator@example.com", settings=settings) == "allowed"


def test_unauthorized_user_is_rejected(tmp_path):
    settings = load_settings(ROOT, _hosted_env(tmp_path))
    assert access_state(authenticated=True, email="other@example.com", settings=settings) == "unauthorized"
    assert access_state(authenticated=False, email=None, settings=settings) == "unauthenticated"


def test_production_cannot_silently_bypass_authentication(tmp_path):
    env = _hosted_env(tmp_path)
    env["PHD_AGENT_AUTH_DISABLED"] = "true"
    settings = load_settings(ROOT, env)
    assert settings.auth_bypass is False
    railway = dict(env)
    railway.pop("PHD_AGENT_ENV")
    railway["RAILWAY_ENVIRONMENT"] = "production"
    assert hosted_from_environ(railway) is True
    assert load_settings(ROOT, railway).auth_bypass is False


def test_railway_data_directory_must_be_absolute(tmp_path):
    env = _hosted_env(tmp_path)
    env["PHD_AGENT_DATA_DIR"] = "data"
    with pytest.raises(ValueError, match="absolute"):
        load_settings(ROOT, env)
    env.pop("PHD_AGENT_DATA_DIR")
    with pytest.raises(ValueError, match="PHD_AGENT_DATA_DIR"):
        load_settings(ROOT, env)


def test_persistent_path_configuration(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    settings = load_settings(ROOT, _hosted_env(volume))
    assert settings.data_dir == volume.resolve()
    assert settings.database_path == volume.resolve() / "phd_outreach.db"


def test_startup_without_openai_or_gmail(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    env = _hosted_env(volume)
    env.pop("OPENAI_API_KEY", None)
    settings = _prepare(tmp_path, env)
    assert not settings.credentials_path.exists()
    assert not settings.gmail_token_path.exists()
    assert settings.database_path.is_file()


def test_gmail_oauth_not_automatically_launched(tmp_path, monkeypatch):
    env = _hosted_env(tmp_path)
    monkeypatch.setenv("PHD_AGENT_ENV", "production")
    monkeypatch.setenv("PHD_AGENT_DATA_DIR", str(tmp_path))
    (tmp_path / "credentials.json").write_text('{"installed":{"client_id":"test"}}', encoding="utf-8")
    called = []

    class FakeFlow:
        @staticmethod
        def from_client_secrets_file(*args, **kwargs):
            called.append(True)
            raise AssertionError("InstalledAppFlow must not start on Railway")

    monkeypatch.setattr(gmail_manager, "InstalledAppFlow", FakeFlow)
    monkeypatch.setattr(gmail_manager, "load_settings", lambda: load_settings(ROOT, env))
    manager = gmail_manager.GmailManager()
    assert manager.service is None
    assert called == []


def test_production_database_migration_and_integrity(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    settings = _prepare(tmp_path, _hosted_env(volume))
    with connect(settings.database_path) as db:
        versions = {row[0] for row in db.execute("SELECT version FROM schema_migrations")}
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    assert {1, 2, 3, 4, 5, 6, 7} <= versions
    assert str(integrity).lower() == "ok"


def test_empty_volume_initializes_and_survives_restart(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    assert not any(volume.iterdir())
    first = _prepare(tmp_path, _hosted_env(volume))
    vault = DocumentVault(first.database_path)
    uploaded = vault.upload(b"persist-me", "note.txt", "SOURCE", "OTHER", "Note")
    marker = volume / "operator-note.txt"
    marker.write_text("keep", encoding="utf-8")
    second = _prepare(tmp_path, _hosted_env(volume))
    restored = DocumentVault(second.database_path)
    assert restored.storage.get(uploaded["storage_key"]) == b"persist-me"
    assert marker.read_text(encoding="utf-8") == "keep"
    assert second.database_path.is_file()


def test_missing_production_volume_is_not_created(tmp_path):
    missing = tmp_path / "not-mounted"
    with pytest.raises(StartupError, match="unavailable"):
        _prepare(tmp_path, _hosted_env(missing))
    assert not missing.exists()


def test_gmail_env_materializes_restricted_files_without_oauth(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    env = _hosted_env(volume)
    env["PHD_AGENT_GMAIL_CREDENTIALS_JSON"] = '{"installed":{"client_id":"rotated"}}'
    env["PHD_AGENT_GMAIL_TOKEN_JSON"] = '{"token":"hosted","client_id":"rotated"}'
    settings = _prepare(tmp_path, env)
    assert json.loads(settings.credentials_path.read_text(encoding="utf-8"))["installed"]["client_id"] == "rotated"
    settings = _prepare(tmp_path, {**env, "PHD_AGENT_GMAIL_CREDENTIALS_JSON": '{"installed":{"client_id":"other"}}'})
    assert json.loads(settings.credentials_path.read_text(encoding="utf-8"))["installed"]["client_id"] == "rotated"


def test_tracked_credentials_stay_out_of_git():
    assert tracked_private_paths(ROOT) == []


def test_auto_send_remains_false_in_production(tmp_path):
    settings = load_settings(ROOT, _hosted_env(tmp_path, AUTO_SEND_ENABLED="true"))
    assert settings.auto_send_enabled is False
    local = Settings(data_dir=tmp_path)
    assert local.auto_send_enabled is False


def test_demo_scripts_refuse_production():
    with pytest.raises(SystemExit, match="cannot run"):
        refuse_hosted_scripts({"PHD_AGENT_ENV": "production"})
    with pytest.raises(SystemExit, match="cannot run"):
        refuse_hosted_scripts({"RAILWAY_PROJECT_ID": "proj_test"})


def test_start_command_uses_port_env_and_health_route():
    text = (ROOT / "railway.toml").read_text(encoding="utf-8")
    assert "--server.port=$PORT" in text
    assert "8501" not in text
    assert 'healthcheckPath = "/_stcore/health"' in text
    assert "--server.address=0.0.0.0" in text
    assert "--server.headless=true" in text


def test_oidc_secrets_are_written_without_network(tmp_path):
    dest = tmp_path / "secrets.toml"
    write_oidc_secrets(dest, _hosted_env(tmp_path))
    text = dest.read_text(encoding="utf-8")
    assert "test-client-id" in text
    assert "[auth]" in text
    assert "accounts.google.com" in text
