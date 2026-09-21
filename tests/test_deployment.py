"""Railway deployment-readiness checks. No Google, Gmail, OpenAI, or Railway network calls."""

from __future__ import annotations

import json
import logging
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
import authlib
import streamlit

import gmail_manager
import streamlit_app
from phd_agent import launch
from phd_agent.access import access_state
from phd_agent.auth_dependencies import (
    AUTHLIB_VERSION, STREAMLIT_VERSION, AuthDependencyError,
    validate_auth_dependencies,
)
from phd_agent.config import GOOGLE_OIDC_METADATA, Settings, hosted_from_environ, load_settings, refuse_hosted_scripts
from phd_agent.db import connect
from phd_agent.documents import DocumentVault
import phd_agent.runtime as runtime
from phd_agent.runtime import LOG_FILE_HANDLER, StartupError, prepare_runtime, write_oidc_secrets
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


def test_production_auth_dependencies_are_installed_and_pinned():
    installed = validate_auth_dependencies()
    assert installed == {"streamlit": STREAMLIT_VERSION, "Authlib": AUTHLIB_VERSION}
    assert streamlit.__version__ == STREAMLIT_VERSION
    assert version("Authlib") == AUTHLIB_VERSION
    assert callable(streamlit.login)
    assert authlib is not None
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert f"streamlit[auth]=={STREAMLIT_VERSION}" in requirements
    assert f"Authlib=={AUTHLIB_VERSION}" in requirements


def test_missing_auth_dependency_stops_before_database_or_secrets(tmp_path, monkeypatch):
    volume = tmp_path / "volume"
    volume.mkdir()
    secrets = tmp_path / "secrets.toml"

    def missing():
        raise AuthDependencyError("Streamlit authentication dependencies are missing")

    monkeypatch.setattr(runtime, "validate_auth_dependencies", missing)
    with pytest.raises(StartupError, match="dependencies are missing"):
        prepare_runtime(ROOT, _hosted_env(volume), secrets_path=secrets)
    assert not (volume / "phd_outreach.db").exists()
    assert not secrets.exists()


def test_allowed_user_is_accepted(tmp_path):
    settings = load_settings(ROOT, _hosted_env(tmp_path))
    assert access_state(authenticated=True, email="Operator@example.com", settings=settings) == "allowed"


def test_unauthorized_user_is_rejected(tmp_path):
    settings = load_settings(ROOT, _hosted_env(tmp_path))
    assert access_state(authenticated=True, email="other@example.com", settings=settings) == "unauthorized"
    assert access_state(authenticated=False, email=None, settings=settings) == "unauthenticated"


@pytest.mark.parametrize(
    ("logged_in", "email", "heading"),
    ((False, None, "This application is private."),
     (True, "other@example.com", "Not authorized")),
)
def test_operator_gate_stops_before_sensitive_ui(tmp_path, monkeypatch, logged_in, email, heading):
    settings = load_settings(ROOT, _hosted_env(tmp_path))

    class StopExecution(Exception):
        pass

    class GateUI:
        user = type("User", (), {"is_logged_in": logged_in, "email": email})()

        def __init__(self):
            self.headings = []

        def header(self, value):
            self.headings.append(value)

        @staticmethod
        def write(_value):
            return None

        @staticmethod
        def button(*_args, **_kwargs):
            return False

        @staticmethod
        def stop():
            raise StopExecution

    gate = GateUI()
    monkeypatch.setattr(streamlit_app, "st", gate)
    with pytest.raises(StopExecution):
        streamlit_app._operator_allowed(settings)
    assert gate.headings == [heading]


def test_production_cannot_silently_bypass_authentication(tmp_path):
    env = _hosted_env(tmp_path)
    env["PHD_AGENT_AUTH_DISABLED"] = "true"
    settings = load_settings(ROOT, env)
    assert settings.auth_bypass is False
    railway = dict(env)
    railway.pop("PHD_AGENT_ENV")
    railway["RAILWAY_ENVIRONMENT"] = "production"
    railway["PHD_AGENT_DATA_DIR"] = "/data"
    assert hosted_from_environ(railway) is True
    assert load_settings(ROOT, railway).auth_bypass is False


def test_invalid_environment_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="development or production"):
        load_settings(ROOT, {
            "PHD_AGENT_ENV": "prodution",
            "PHD_AGENT_DATA_DIR": str(tmp_path),
            "PHD_AGENT_AUTH_DISABLED": "true",
        })


def test_railway_data_directory_must_be_absolute(tmp_path):
    env = _hosted_env(tmp_path)
    env["PHD_AGENT_DATA_DIR"] = "data"
    with pytest.raises(ValueError, match="absolute"):
        load_settings(ROOT, env)
    env.pop("PHD_AGENT_DATA_DIR")
    with pytest.raises(ValueError, match="PHD_AGENT_DATA_DIR"):
        load_settings(ROOT, env)


def test_railway_requires_data_volume_path(tmp_path):
    env = _hosted_env(tmp_path)
    env.pop("PHD_AGENT_ENV")
    env["RAILWAY_ENVIRONMENT"] = "production"
    with pytest.raises(ValueError, match="PHD_AGENT_DATA_DIR=/data"):
        load_settings(ROOT, env)
    env["PHD_AGENT_DATA_DIR"] = "/data"
    settings = load_settings(ROOT, env)
    assert settings.railway is True
    assert str(settings.data_dir).replace("\\", "/").endswith("/data")


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


def test_local_data_dir_is_created_when_missing(tmp_path):
    data = tmp_path / "data"
    assert not data.exists()
    settings = prepare_runtime(ROOT, {
        "PHD_AGENT_ENV": "development",
        "PHD_AGENT_DATA_DIR": str(data),
        "PHD_AGENT_AUTH_DISABLED": "true",
    }, secrets_path=tmp_path / "secrets.toml")
    assert data.is_dir()
    assert (data / "documents").is_dir()
    assert settings.database_path.is_file()
    assert not (tmp_path / "secrets.toml").exists()


def test_hosted_startup_requires_allowlist(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    env = _hosted_env(volume)
    env["PHD_AGENT_ALLOWED_EMAILS"] = ""
    with pytest.raises(StartupError, match="ALLOWED_EMAILS"):
        _prepare(tmp_path, env)


def test_prepare_runtime_reuses_log_handler(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    env = _hosted_env(volume)
    _prepare(tmp_path, env)
    root = logging.getLogger()
    first = [handler for handler in root.handlers if getattr(handler, "name", "") == LOG_FILE_HANDLER]
    _prepare(tmp_path, env)
    second = [handler for handler in root.handlers if getattr(handler, "name", "") == LOG_FILE_HANDLER]
    assert len(first) == 1
    assert first == second


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
    assert "python -m phd_agent.launch" in text
    assert "8501" not in text
    assert 'healthcheckPath = "/_stcore/health"' in text
    argv = launch.streamlit_command("8080")
    assert "--server.port=8080" in argv
    assert "--server.address=0.0.0.0" in argv
    assert "--server.headless=true" in argv
    assert "8501" not in argv


def test_streamlit_cors_and_websocket_hosts_are_restricted():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    server = config["server"]
    assert server["enableCORS"] is True
    assert server["enableXsrfProtection"] is True
    assert server["corsAllowedOrigins"] == ["https://phd-agent-production.up.railway.app"]
    assert "*" not in server["corsAllowedOrigins"]
    assert "phd-agent-production.up.railway.app" in server["allowedHosts"]
    assert "localhost" in server["allowedHosts"]
    assert "127.0.0.1" in server["allowedHosts"]
    assert "*" not in server["allowedHosts"]
    assert streamlit.config.get_option("server.enableCORS") is True
    assert streamlit.config.get_option("server.enableXsrfProtection") is True
    assert streamlit.config.get_option("server.corsAllowedOrigins") == server["corsAllowedOrigins"]
    assert streamlit.config.get_option("server.allowedHosts") == server["allowedHosts"]


def test_launch_writes_oidc_secrets_before_exec(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    env = _hosted_env(volume)
    env["PORT"] = "4321"
    executed = []

    def fake_exec(file, args):
        parsed = tomllib.loads((tmp_path / "secrets.toml").read_text(encoding="utf-8"))
        assert parsed["auth"]["client_id"] == "test-client-id"
        assert parsed["auth"]["client_secret"] == "test-client-secret"
        assert parsed["auth"]["cookie_secret"] == "test-cookie-secret"
        assert parsed["auth"]["redirect_uri"] == "https://example.invalid/oauth2callback"
        assert parsed["auth"]["server_metadata_url"] == GOOGLE_OIDC_METADATA
        assert "expose_tokens" not in parsed["auth"]
        executed.append((file, list(args)))
        raise SystemExit(0)

    with pytest.raises(SystemExit) as stopped:
        launch.main(env, root=ROOT, secrets_path=tmp_path / "secrets.toml", exec_fn=fake_exec)
    assert stopped.value.code == 0
    assert executed
    assert "--server.port=4321" in executed[0][1]
    parsed = tomllib.loads((tmp_path / "secrets.toml").read_text(encoding="utf-8"))
    assert parsed["auth"]["client_id"] == "test-client-id"


def test_launch_refuses_hosted_start_without_allowlist(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    env = _hosted_env(volume)
    env["PHD_AGENT_ALLOWED_EMAILS"] = ""
    env["PORT"] = "8080"
    executed = []
    with pytest.raises(SystemExit) as stopped:
        launch.main(env, root=ROOT, secrets_path=tmp_path / "secrets.toml",
                    exec_fn=lambda file, args: executed.append(args))
    assert stopped.value.code == 1
    assert executed == []
    assert not (tmp_path / "secrets.toml").exists()


def test_oidc_secrets_are_written_without_network(tmp_path):
    dest = tmp_path / "secrets.toml"
    write_oidc_secrets(dest, _hosted_env(tmp_path))
    parsed = tomllib.loads(dest.read_text(encoding="utf-8"))
    assert parsed["auth"]["client_id"] == "test-client-id"
    assert parsed["auth"]["server_metadata_url"] == GOOGLE_OIDC_METADATA
    assert "expose_tokens" not in parsed["auth"]


def test_production_oidc_urls_are_strict(tmp_path):
    volume = tmp_path / "volume"
    volume.mkdir()
    env = _hosted_env(volume)
    env["PHD_AGENT_OIDC_REDIRECT_URI"] = "https://phd-agent-production.up.railway.app/oauth2callback"
    secrets = tmp_path / "secrets.toml"
    prepare_runtime(ROOT, env, secrets_path=secrets)
    parsed = tomllib.loads(secrets.read_text(encoding="utf-8"))
    assert parsed["auth"]["redirect_uri"] == env["PHD_AGENT_OIDC_REDIRECT_URI"]
    assert parsed["auth"]["server_metadata_url"] == GOOGLE_OIDC_METADATA
    assert "expose_tokens" not in parsed["auth"]

    bad_redirect = {**env, "PHD_AGENT_OIDC_REDIRECT_URI": "http://example.invalid/oauth2callback"}
    with pytest.raises(StartupError, match="absolute HTTPS"):
        _prepare(tmp_path, bad_redirect)
    bad_metadata = {**env, "PHD_AGENT_OIDC_SERVER_METADATA_URL": "https://example.invalid/.well-known/openid-configuration"}
    with pytest.raises(StartupError, match="Google"):
        _prepare(tmp_path, bad_metadata)


def test_development_http_oidc_redirect_requires_loopback(tmp_path):
    data_dir = tmp_path / "data"
    env = {
        "PHD_AGENT_ENV": "development",
        "PHD_AGENT_DATA_DIR": str(data_dir),
        "PHD_AGENT_ALLOWED_EMAILS": "operator@example.com",
        "PHD_AGENT_OIDC_CLIENT_ID": "test-client-id",
        "PHD_AGENT_OIDC_CLIENT_SECRET": "test-client-secret",
        "PHD_AGENT_OIDC_COOKIE_SECRET": "test-cookie-secret",
        "PHD_AGENT_OIDC_REDIRECT_URI": "http://development.example/oauth2callback",
    }
    with pytest.raises(StartupError, match="localhost or a loopback IP"):
        prepare_runtime(ROOT, env, secrets_path=tmp_path / "secrets.toml")
    assert not data_dir.exists()


@pytest.mark.parametrize("host", ("localhost", "127.0.0.1", "[::1]"))
def test_development_http_oidc_redirect_accepts_loopback(tmp_path, host):
    data_dir = tmp_path / host.replace(":", "_").replace("[", "").replace("]", "")
    env = {
        "PHD_AGENT_ENV": "development",
        "PHD_AGENT_DATA_DIR": str(data_dir),
        "PHD_AGENT_ALLOWED_EMAILS": "operator@example.com",
        "PHD_AGENT_OIDC_CLIENT_ID": "test-client-id",
        "PHD_AGENT_OIDC_CLIENT_SECRET": "test-client-secret",
        "PHD_AGENT_OIDC_COOKIE_SECRET": "test-cookie-secret",
        "PHD_AGENT_OIDC_REDIRECT_URI": f"http://{host}:8501/oauth2callback",
    }
    settings = prepare_runtime(ROOT, env, secrets_path=data_dir / "secrets.toml")
    assert settings.data_dir == data_dir.resolve()


def test_cms_does_not_render_before_operator_authorization(tmp_path, monkeypatch):
    settings = load_settings(ROOT, _hosted_env(tmp_path))
    rendered = []

    class PageOnly:
        @staticmethod
        def set_page_config(**_kwargs):
            return None

    monkeypatch.setattr(streamlit_app, "st", PageOnly())
    monkeypatch.setattr(streamlit_app, "prepare_runtime", lambda: settings)
    monkeypatch.setattr(streamlit_app, "_operator_allowed", lambda _settings: False)
    monkeypatch.setattr(streamlit_app, "render_cms", lambda _path: rendered.append(True))
    streamlit_app.main()
    assert rendered == []
