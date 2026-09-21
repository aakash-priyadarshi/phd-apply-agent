"""Simple applicant-profile and task workspace regressions."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from phd_agent.applicant_context import ApplicantResearchContextService
from phd_agent.db import connect, migrate
from phd_agent.documents import DocumentVault
from phd_agent.profile_workspace import ProfileUpload, ProfileWorkspace


CV_TEXT = b"""Aakash Example
MSc Computer Science, University of Liverpool
Research project: evaluated long-horizon AI agents using soundness and reward-hacking checks.
Built retrieval augmented generation workflows in Python and PyTorch.
Technical skills: Python, SQL, PyTorch, LLM evaluation and RAG.
I aim to research reliable agentic AI systems.
"""


def test_one_action_builds_context_and_markdown_summary(tmp_path):
    database = tmp_path / "phd_outreach.db"
    migrate(database)
    workspace = ProfileWorkspace(database)
    result = workspace.build(
        "Aakash Example", "Reliable evaluation of agentic AI",
        [ProfileUpload("Aakash-CV.txt", CV_TEXT)],
    )
    assert result.documents_added == 1
    assert result.facts_in_profile > 0
    assert result.summary_path.is_file()
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "# Applicant profile — Aakash Example" in summary
    assert "Reliable evaluation of agentic AI" in summary
    assert "Aakash-CV.txt" in summary
    context = ApplicantResearchContextService(database).get(result.context_id)
    assert context["profile_version_id"] == result.profile_version_id
    assert context["master_cv_version_id"] == result.master_cv_version_id
    with connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM profile_versions WHERE approval_state='APPROVED'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM master_cv_versions WHERE approval_state='APPROVED'").fetchone()[0] == 1


def test_existing_vault_cv_can_finish_setup_without_reupload(tmp_path):
    database = tmp_path / "phd_outreach.db"
    migrate(database)
    vault = DocumentVault(database)
    saved = vault.upload(CV_TEXT, "existing-cv.txt", "SOURCE", "CV", "Existing CV")
    vault.set_approval(saved["version_id"], True, "Earlier review")
    result = ProfileWorkspace(database).build("Aakash Example", "Reliable AI agents")
    assert result.documents_added == 0
    assert ApplicantResearchContextService(database).available_inputs()


def test_reusing_a_document_for_a_corrected_owner_builds_the_new_profile(tmp_path):
    database = tmp_path / "phd_outreach.db"
    workspace = ProfileWorkspace(database)
    first = workspace.build("Incorrect Name", "Reliable AI agents", [ProfileUpload("cv.txt", CV_TEXT)])
    corrected = workspace.build("Aakash Example", "Reliable AI agents")
    assert corrected.profile_id != first.profile_id
    assert corrected.context_id != first.context_id
    assert corrected.facts_in_profile > 0


def test_adding_document_refreshes_profile_and_summary(tmp_path):
    database = tmp_path / "phd_outreach.db"
    workspace = ProfileWorkspace(database)
    first = workspace.build("Aakash Example", "Reliable AI agents", [ProfileUpload("cv.txt", CV_TEXT)])
    second = workspace.build("Aakash Example", "Reliable AI agents", [ProfileUpload(
        "publication.md", b"Published paper: Reliable evaluation benchmarks for autonomous retrieval agents." )])
    assert second.profile_version_id > first.profile_version_id
    assert second.context_id > first.context_id
    assert second.summary_path.name != first.summary_path.name
    assert "Published paper" in second.summary_path.read_text(encoding="utf-8")


def test_simple_workspace_renders_setup_then_task_navigation(tmp_path, monkeypatch):
    monkeypatch.setenv("PHD_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PHD_AGENT_ENV", "development")
    monkeypatch.setenv("PHD_AGENT_AUTH_DISABLED", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    setup = AppTest.from_file(app_path, default_timeout=60).run()
    assert not setup.exception
    assert any(title.value == "PhD Application Assistant" for title in setup.title)
    assert any(button.label == "Build my profile" for button in setup.button)
    ProfileWorkspace(tmp_path / "phd_outreach.db").build(
        "Aakash Example", "Reliable AI agents", [ProfileUpload("cv.txt", CV_TEXT)])
    ready = AppTest.from_file(app_path, default_timeout=60).run()
    assert not ready.exception
    navigation = next(radio for radio in ready.radio if radio.label == "Navigation")
    assert navigation.options == ["Home", "Find programmes", "Applications", "People", "My documents"]
    assert any("Welcome back" in markdown.value for markdown in ready.markdown)
