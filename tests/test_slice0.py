"""Safety checks for the 2026–27 baseline upgrade."""

from types import SimpleNamespace

import pytest

import gmail_manager
import phd_agent.security as security
import streamlit_app
from phd_agent.config import Settings, load_settings
from phd_agent.paths import ensure_data_layout


def test_legacy_files_are_copied_and_database_backed_up(tmp_path):
    root = tmp_path / "repository"
    data = root / "data"
    root.mkdir()
    (root / "phd_outreach.db").write_bytes(b"legacy database bytes")
    (root / "uploaded_cv.pdf").write_bytes(b"private cv bytes")
    copied = ensure_data_layout(data, root)

    assert "legacy database backup" in copied
    assert (data / "backups" / "phd_outreach-legacy.db").read_bytes() == b"legacy database bytes"
    assert (data / "phd_outreach.db").read_bytes() == b"legacy database bytes"
    assert (data / "documents" / "uploaded_cv.pdf").read_bytes() == b"private cv bytes"
    assert (root / "phd_outreach.db").exists()

    (data / "phd_outreach.db").write_bytes(b"newer database bytes")
    ensure_data_layout(data, root)
    assert (data / "phd_outreach.db").read_bytes() == b"newer database bytes"


def test_auto_send_cannot_be_enabled_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_SEND_ENABLED", "true")
    settings = load_settings(tmp_path)
    assert settings.auto_send_enabled is False


def test_custom_data_dir_inside_repository_must_be_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("PHD_AGENT_DATA_DIR", "unignored-runtime")
    try:
        load_settings(tmp_path)
    except ValueError as error:
        assert "ignored data/" in str(error)
    else:
        raise AssertionError("An unignored document store was accepted")


def test_legacy_bulk_paths_never_call_gmail(monkeypatch):
    monkeypatch.setenv("AUTO_SEND_ENABLED", "true")
    orchestrator = streamlit_app.ResearchOrchestrator.__new__(streamlit_app.ResearchOrchestrator)
    orchestrator.gmail_manager = object()
    orchestrator.generate_bulk_emails_sync = lambda *_: (_ for _ in ()).throw(
        AssertionError("generation should not run")
    )

    assert orchestrator.send_bulk_emails_sync("Applicant")["success"] is False
    assert orchestrator.generate_and_send_all_sync("profile", "Applicant")["success"] is False

    manager = gmail_manager.GmailManager.__new__(gmail_manager.GmailManager)
    manager.send_email = lambda **_: (_ for _ in ()).throw(AssertionError("Gmail send called"))
    result = manager.send_bulk_emails([{"to_email": "test@example.invalid"}], "Applicant")
    assert result == [{"success": False, "error": "Bulk sending is disabled"}]


def test_tracked_private_file_scan_returns_names_only(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        stdout=b"streamlit_app.py\0credentials.json\0data/gmail_token.json\0"
        b"uploaded_cv.pdf\0phd_outreach.db\0",
    )
    monkeypatch.setattr(security.subprocess, "run", lambda *args, **kwargs: fake)
    found = security.tracked_private_paths(tmp_path)
    assert found == [
        "credentials.json",
        "data/gmail_token.json",
        "uploaded_cv.pdf",
        "phd_outreach.db",
    ]


def test_git_index_has_no_private_runtime_files():
    assert security.tracked_private_paths() == []


def test_git_inspection_failure_is_explicit(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise OSError("git missing")
    monkeypatch.setattr(security.subprocess, "run", boom)
    with pytest.raises(security.GitInspectionFailed):
        security.tracked_private_paths(tmp_path)


def test_gmail_writes_json_token_without_loading_pickle(tmp_path, monkeypatch):
    settings = Settings(tmp_path)
    settings.credentials_path.write_text("{}", encoding="utf-8")
    (tmp_path / "gmail_token.pickle").write_bytes(b"legacy token must not be read")
    monkeypatch.setattr(gmail_manager, "load_settings", lambda: settings)

    class FakeCredentials:
        valid = True

        def to_json(self):
            return '{"token":"test-only"}'

    class FakeFlow:
        def run_local_server(self, port):
            assert port == 0
            return FakeCredentials()

    class FakeInstalledAppFlow:
        @staticmethod
        def from_client_secrets_file(path, scopes):
            assert path == str(settings.credentials_path)
            assert scopes
            return FakeFlow()

    class FakeService:
        def users(self):
            return self

        def getProfile(self, userId):
            assert userId == "me"
            return self

        def execute(self):
            return {"emailAddress": "test@example.invalid"}

    monkeypatch.setattr(gmail_manager, "InstalledAppFlow", FakeInstalledAppFlow)
    monkeypatch.setattr(gmail_manager, "build", lambda *args, **kwargs: FakeService())

    manager = gmail_manager.GmailManager()
    assert manager.service is not None
    assert settings.gmail_token_path.read_text(encoding="utf-8") == '{"token":"test-only"}'
    assert (tmp_path / "gmail_token.pickle").read_bytes() == b"legacy token must not be read"
