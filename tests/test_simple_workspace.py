"""Simple applicant-profile and task workspace regressions."""

import io
import zipfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import phd_agent.profile_workspace as profile_workspace
from phd_agent.applicant_context import ApplicantResearchContextService
from phd_agent.db import connect, migrate
from phd_agent.documents import DocumentVault
from phd_agent.profile_workspace import ProfileUpload, ProfileWorkspace, extract_text


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


def test_research_focus_change_preserves_the_previous_summary(tmp_path):
    workspace = ProfileWorkspace(tmp_path / "phd_outreach.db")
    first = workspace.build("Aakash Example", "Reliable AI agents", [ProfileUpload("cv.txt", CV_TEXT)])
    original = first.summary_path.read_text(encoding="utf-8")
    second = workspace.build("Aakash Example", "Multimodal agent evaluation")
    assert second.profile_version_id == first.profile_version_id
    assert second.research_track_version_id != first.research_track_version_id
    assert second.summary_path != first.summary_path
    assert first.summary_path.read_text(encoding="utf-8") == original
    assert "Multimodal agent evaluation" in second.summary_path.read_text(encoding="utf-8")


def test_deterministic_candidates_do_not_promote_historic_seeking_to_aspirations():
    line = "I was seeking to investigate reliable agent evaluation methods."

    candidate = profile_workspace._deterministic_candidates(line)[0]

    assert candidate["classification"] == "FACT"
    assert candidate["statement"] == line


def test_extract_text_rejects_oversized_files_and_expanded_docx(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_workspace, "MAX_PROFILE_DOCUMENT_BYTES", 20)
    with pytest.raises(ValueError, match="profile-document limit"):
        extract_text(b"x" * 21, "large.txt")

    monkeypatch.setattr(profile_workspace, "MAX_PROFILE_DOCUMENT_BYTES", 1024)
    monkeypatch.setattr(profile_workspace, "MAX_PROFILE_DOCX_EXPANDED_BYTES", 40)
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "x" * 41)
    with pytest.raises(ValueError, match="expanded DOCX"):
        extract_text(archive_bytes.getvalue(), "expanded.docx")

    monkeypatch.setattr(profile_workspace, "MAX_PROFILE_DOCUMENT_BYTES", 50)
    database = tmp_path / "phd_outreach.db"
    workspace = ProfileWorkspace(database)
    with pytest.raises(ValueError, match="profile-document limit"):
        workspace.build("Aakash Example", "Reliable AI", [
            ProfileUpload("valid.txt", b"Research project: tested reliable AI agents."),
            ProfileUpload("large.txt", b"x" * 51),
        ])
    with connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM applicant_profiles").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_approved_document_is_hash_checked_before_reuse(tmp_path):
    workspace = ProfileWorkspace(tmp_path / "phd_outreach.db")
    workspace.build("Aakash Example", "Reliable AI agents", [ProfileUpload("cv.txt", CV_TEXT)])
    document = workspace.source_documents()[0]
    workspace.vault.storage._path(document["storage_key"]).write_bytes(b"corrupt")
    with pytest.raises(OSError, match="missing or corrupt"):
        workspace.build("Aakash Example", "Reliable AI agents")


def test_academic_documents_wait_for_authenticity_review(tmp_path):
    workspace = ProfileWorkspace(tmp_path / "phd_outreach.db")
    result = workspace.build("Aakash Example", "Reliable AI agents", [
        ProfileUpload("cv.txt", CV_TEXT),
        ProfileUpload(
            "transcript.txt", b"Official transcript for MSc Computer Science at Example University, grade distinction."),
    ])
    document = next(item for item in workspace.source_documents()
                    if item["document_type"] == "TRANSCRIPT")
    assert document["document_type"] == "TRANSCRIPT"
    assert document["verification_state"] == "NEEDS_REVIEW"
    assert document["approval_state"] == "PENDING"
    assert any("stored and parsed" in warning for warning in result.warnings)


def test_model_aspirations_do_not_block_cv_and_sop_import(tmp_path, monkeypatch):
    workspace = ProfileWorkspace(tmp_path / "phd_outreach.db")

    def extracted_candidates(profile_id, extraction_id, **_kwargs):
        with connect(workspace.db_path) as db:
            version_id = db.execute(
                "SELECT document_version_id FROM applicant_document_extractions WHERE id=?",
                (extraction_id,),
            ).fetchone()[0]
        workspace.truth.create_claim(
            profile_id, "OTHER", "I seek to investigate reliable agent evaluation.",
            "research_goal", "ASPIRATION", source_document_version_id=version_id,
            source_location="SOP", extraction_id=extraction_id, confidence=0.9,
        )
        workspace.truth.create_claim(
            profile_id, "OTHER", "Reliable agent evaluation across long-horizon tasks.",
            "research_goal", "ASPIRATION", source_document_version_id=version_id,
            source_location="SOP", extraction_id=extraction_id, confidence=0.7,
        )
        return {"pending": 2, "rejected_unsupported": 0}

    monkeypatch.setattr(workspace.truth, "extract_candidates", extracted_candidates)
    result = workspace.build(
        "Aakash Example", "Reliable AI agents",
        [ProfileUpload("statement-of-purpose.txt", CV_TEXT)], api_key="configured",
    )
    claims = workspace.truth.list_claims(result.profile_id)
    assert any(claim["claim_text"].startswith("I seek to investigate")
               and claim["review_status"] == "APPROVED" for claim in claims)
    assert any(claim["claim_text"].startswith("Reliable agent evaluation")
               and claim["review_status"] == "REJECTED" for claim in claims)
    assert any("unclear aspiration" in warning for warning in result.warnings)


def test_retry_processes_pending_claims_after_an_earlier_partial_import(tmp_path):
    workspace = ProfileWorkspace(tmp_path / "phd_outreach.db")
    first = workspace.build(
        "Aakash Example", "Reliable AI agents", [ProfileUpload("cv.txt", CV_TEXT)])
    document = workspace.source_documents()[0]
    with connect(workspace.db_path) as db:
        extraction_id = db.execute(
            "SELECT id FROM applicant_document_extractions WHERE document_version_id=?",
            (document["version_id"],),
        ).fetchone()[0]
    pending_id = workspace.truth.create_claim(
        first.profile_id, "OTHER", "Reliable agent evaluation across long-horizon tasks.",
        "research_goal", "ASPIRATION", source_document_version_id=document["version_id"],
        source_location="SOP", extraction_id=extraction_id, confidence=0.7,
    )
    retried = workspace.build("Aakash Example", "Reliable AI agents")
    with connect(workspace.db_path) as db:
        pending = db.execute(
            "SELECT review_status FROM claim_revisions WHERE id=?", (pending_id,)
        ).fetchone()
    assert pending["review_status"] == "REJECTED"
    assert any("unclear aspiration" in warning for warning in retried.warnings)


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
