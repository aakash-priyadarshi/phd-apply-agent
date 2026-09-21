"""Slice 1 migration, ledger, and local Vault behavior without network or Gmail."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

import phd_agent.db as migrations
from phd_agent.db import connect, migrate
from phd_agent.documents import DocumentVault, LocalDocumentStorage
from phd_agent.ledger import Ledger
from phd_agent.paths import APP_ROOT


@pytest.fixture
def legacy_copy(tmp_path):
    original = tmp_path / "historical-2025.db"
    with sqlite3.connect(original) as db:
        db.execute("""CREATE TABLE professors (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, university TEXT NOT NULL,
            department TEXT, email TEXT, research_interests TEXT, profile_url TEXT,
            status TEXT DEFAULT 'pending', draft_email_body TEXT)""")
        db.execute("CREATE TABLE cost_tracking (id INTEGER PRIMARY KEY, total_cost REAL)")
        db.execute("CREATE TABLE sent_emails (id INTEGER PRIMARY KEY, professor_id INTEGER)")
        db.executemany(
            "INSERT INTO professors(id, name, university, draft_email_body) VALUES (?, ?, ?, ?)",
            [(i, f"Professor {i}", "Historical University", f"draft {i}") for i in range(1, 157)],
        )
        db.execute("INSERT INTO cost_tracking VALUES (1, 1.25)")
    upgraded = tmp_path / "upgraded.db"
    shutil.copy2(original, upgraded)
    return upgraded


def test_migration_preserves_156_legacy_professors_and_is_idempotent(legacy_copy):
    with connect(legacy_copy) as db:
        old_rows = [tuple(r) for r in db.execute("SELECT * FROM professors ORDER BY id")]
        old_schema = db.execute("SELECT sql FROM sqlite_master WHERE name = 'professors'").fetchone()[0]
    assert migrate(legacy_copy) == [1, 2, 3, 4, 5, 6]
    assert migrate(legacy_copy) == []
    with connect(legacy_copy) as db:
        assert [tuple(r) for r in db.execute("SELECT * FROM professors ORDER BY id")] == old_rows
        assert len(old_rows) == 156
        assert db.execute("SELECT sql FROM sqlite_master WHERE name = 'professors'").fetchone()[0] == old_schema
        assert db.execute("SELECT total_cost FROM cost_tracking WHERE id = 1").fetchone()[0] == 1.25
        assert db.execute("SELECT version, name, applied_at FROM schema_migrations").fetchone()[0] == 1
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        version_columns = {r["name"] for r in db.execute("PRAGMA table_info(document_versions)")}
        assert {"issue_date", "expiry_date", "sensitivity", "generation_context",
                "prompt_version", "model_used", "claim_ids_json", "source_evidence_ids_json",
                "manual_edits", "approved_by", "approved_at"} <= version_columns


def test_migration_against_ignored_historical_backup_when_available(tmp_path):
    backup = APP_ROOT / "data" / "backups" / "phd_outreach-legacy-1bcb632.db"
    if not backup.is_file():
        pytest.skip("Local ignored legacy backup is not present")
    copied = tmp_path / "actual-legacy-copy.db"
    shutil.copy2(backup, copied)
    with connect(copied) as db:
        old = db.execute("SELECT * FROM professors ORDER BY id").fetchall()
        old_hash = hashlib.sha256(json.dumps([tuple(r) for r in old]).encode()).hexdigest()
    migrate(copied)
    with connect(copied) as db:
        new = db.execute("SELECT * FROM professors ORDER BY id").fetchall()
        new_hash = hashlib.sha256(json.dumps([tuple(r) for r in new]).encode()).hexdigest()
        assert len(new) == len(old) == 156
        assert new_hash == old_hash
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migration_failure_rolls_back_whole_version(tmp_path, monkeypatch):
    path = tmp_path / "failed.db"
    monkeypatch.setattr(migrations, "MIGRATIONS", ((1, "broken", (
        "CREATE TABLE should_rollback (id INTEGER)", "THIS IS NOT SQL",
    )),))
    with pytest.raises(sqlite3.OperationalError):
        migrate(path)
    with connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name = 'should_rollback'").fetchone() is None
        assert db.execute("SELECT name FROM sqlite_master WHERE name = 'schema_migrations'").fetchone() is None


def _application(ledger: Ledger, suffix: str = "Oxford") -> int:
    programme = ledger.create_programme(
        suffix, "DPhil Computer Science", department="Computer Science",
        degree_type="DPhil", cycle="2026–27",
        programme_url="https://example.edu/programme",
        admissions_url="https://example.edu/admissions",
    )
    opportunity = ledger.create_opportunity(
        "PROGRAMME_APPLICATION", "2026–27 doctoral entry", suffix,
        programme_id=programme, canonical_url="https://example.edu/programme",
        opening_status="UNKNOWN",
    )
    return ledger.create_application(
        "2026–27", programme, opportunity,
        status="IN_PROGRESS", next_action="Confirm language evidence applicability",
    )


def test_manual_application_evidence_deadlines_requirements_and_tasks(legacy_copy):
    ledger = Ledger(legacy_copy)
    app = _application(ledger)
    source = ledger.create_evidence(
        "https://example.edu/admissions", "ADMISSIONS",
        "Deadline and documents listed on admissions page", "VERIFIED",
    )
    evidence = ledger.get("source_evidence", source)
    assert evidence["content_hash"] == hashlib.sha256(
        evidence["relevant_excerpt"].encode("utf-8")
    ).hexdigest()
    with connect(legacy_copy) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE source_evidence SET relevant_excerpt = 'changed' WHERE id = ?", (source,))
    with connect(legacy_copy) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM source_evidence WHERE id = ?", (source,))

    ledger.create_deadline(app, "APPLICATION", "2026-12-01", source, timezone="Europe/London")
    funding_deadline = ledger.create_deadline(app, "FUNDING", "2026-11-01T17:00", source, timezone="Europe/London")
    ledger.update_deadline(funding_deadline, due_at="2026-11-02T17:00", notes="Updated from new source review")
    assert len(ledger.list_deadlines(app)) == 2
    assert ledger.get("deadlines", funding_deadline)["due_at"] == "2026-11-02T17:00"
    cv_req = ledger.create_requirement(
        app, "FORMAL_APPLICATION", "REQUIRED", "Academic CV", source,
        normalized_document_type="CV", file_format="PDF", upload_field="CV upload",
    )
    ledger.create_requirement(
        app, "FORMAL_APPLICATION", "UNKNOWN", "English evidence", source,
        normalized_document_type="ENGLISH_TEST",
        condition_text="Only if waiver does not apply",
    )
    ledger.create_requirement(
        app, "FACULTY_OUTREACH", "OPTIONAL", "Short research summary", source,
        normalized_document_type="RESEARCH_STATEMENT",
    )
    assert ledger.get("requirements", cv_req)["source_evidence_id"] == source
    assert ledger.readiness(app) == {
        "required_complete": 0, "required_total": 1,
        "unknown_requirements": 1, "conditional_requirements": 1,
    }
    with pytest.raises(ValueError):
        ledger.create_deadline(app, "APPLICATION", "next Friday", source)
    outreach_req = ledger.create_requirement(
        app, "FACULTY_OUTREACH", "REQUIRED", "Outreach CV", source,
        normalized_document_type="CV",
    )
    assert ledger.get("requirements", outreach_req)["context"] == "FACULTY_OUTREACH"
    assert ledger.readiness(app)["required_total"] == 1
    assert ledger.readiness(app, "FACULTY_OUTREACH")["required_total"] == 1
    task = ledger.create_task(app, "VERIFY", "Check English waiver", due_at="2026-10-01")
    assert ledger.get("application_tasks", task)["status"] == "TODO"
    ledger.update_task(task, status="DONE")
    assert ledger.get("application_tasks", task)["completed_at"]
    ledger.delete_task(task)
    assert ledger.get("application_tasks", task) is None
    ledger.update_application(app, next_action="Upload CV")
    assert ledger.get("applications", app)["next_action"] == "Upload CV"
    referee = ledger.create_referee(app, "Dr Example", institution="Example Institute")
    ledger.update_referee(referee, submission_state="SUBMITTED", submitted_at="2026-11-01")
    assert ledger.list_referees(app)[0]["submission_state"] == "SUBMITTED"
    with connect(legacy_copy) as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    # A populated application is protected from accidental deletion.
    with pytest.raises(sqlite3.IntegrityError):
        ledger.delete_application(app)


def test_vault_duplicates_versions_local_policy_and_cross_application_reuse(tmp_path):
    path = tmp_path / "ledger.db"
    ledger = Ledger(path)
    vault = DocumentVault(path)
    app1 = _application(ledger, "Oxford")
    app2 = _application(ledger, "Cambridge")
    source = ledger.create_evidence("https://example.edu/requirements", "ADMISSIONS")
    requirement_ids = [
        ledger.create_requirement(app, "FORMAL_APPLICATION", "REQUIRED", "MSc transcript", source,
                                  normalized_document_type="TRANSCRIPT")
        for app in (app1, app2)
    ]
    original = b"first original transcript bytes"
    created = vault.upload(original, "Transcript.pdf", "SOURCE", "TRANSCRIPT", "MSc transcript",
                           expiry_date="2027-01-01")
    assert created["status"] == "created"
    assert vault.storage.get(created["storage_key"]) == original
    assert vault.storage.verify_hash(created["storage_key"], hashlib.sha256(original).hexdigest())
    duplicate = vault.upload(original, "different-name.pdf", "SOURCE", "TRANSCRIPT", "Duplicate title")
    assert duplicate["status"] == "duplicate"
    assert duplicate["version_id"] == created["version_id"]
    assert len(list((tmp_path / "documents" / "objects").rglob("*"))) >= 2
    assert len([p for p in (tmp_path / "documents" / "objects").rglob("*") if p.is_file()]) == 1

    vault.set_approval(created["version_id"], True, "Test operator")
    for app, requirement in zip((app1, app2), requirement_ids):
        vault.link_to_application(app, requirement, created["version_id"])
        assert ledger.readiness(app)["required_complete"] == 1
    assert len([p for p in (tmp_path / "documents" / "objects").rglob("*") if p.is_file()]) == 1

    replacement = vault.upload(
        b"replacement transcript bytes", "replacement.pdf", "SOURCE", "TRANSCRIPT",
        "MSc transcript", document_id=created["document_id"], expiry_date="2028-01-01",
    )
    assert replacement["version_number"] == 2
    assert len(vault.list_versions(created["document_id"])) == 2
    assert vault.storage.get(created["storage_key"]) == original
    assert vault.get_version(created["version_id"])["expiry_date"] == "2027-01-01"
    assert vault.get_version(replacement["version_id"])["expiry_date"] == "2028-01-01"
    assert ledger.requirement_rows(app1)[0]["document_version_id"] == created["version_id"]
    assert vault.get_version(replacement["version_id"])["approval_state"] == "PENDING"
    same_name_new_bytes = vault.upload(
        b"another distinct document", "Transcript.pdf", "SOURCE", "TRANSCRIPT", "Another transcript"
    )
    assert same_name_new_bytes["status"] == "created"
    assert same_name_new_bytes["storage_key"] != created["storage_key"]

    passport = vault.upload(b"private ID", "id.pdf", "SOURCE", "PASSPORT", "Passport")
    passport_doc = next(d for d in vault.list_documents() if d["id"] == passport["document_id"])
    assert passport_doc["sensitivity"] == "HIGHLY_SENSITIVE"
    assert passport_doc["permitted_storage_policy"] == "LOCAL_ONLY"
    assert passport_doc["storage_backend"] == "LOCAL"
    vault.update_document(passport["document_id"], sensitivity="NORMAL")
    passport_doc = next(d for d in vault.list_documents() if d["id"] == passport["document_id"])
    assert passport_doc["sensitivity"] == "HIGHLY_SENSITIVE"
    assert passport_doc["permitted_storage_policy"] == "LOCAL_ONLY"

    export = vault.storage.export(created["storage_key"], tmp_path / "export" / "transcript.pdf")
    assert export.read_bytes() == original
    with pytest.raises(ValueError):
        vault.storage.get("../../escape")
    with connect(path) as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_document_approval_expiry_and_missing_file_affect_counts(tmp_path):
    path = tmp_path / "ledger.db"
    ledger, vault = Ledger(path), DocumentVault(path)
    app = _application(ledger)
    source = ledger.create_evidence("https://example.edu/checklist", "ADMISSIONS")
    req = ledger.create_requirement(app, "FORMAL_APPLICATION", "REQUIRED", "CV", source,
                                    normalized_document_type="CV")
    version = vault.upload(b"CV bytes", "cv.pdf", "SOURCE", "CV", "My CV")
    vault.link_to_application(app, req, version["version_id"])
    assert ledger.requirement_rows(app)[0]["document_state"] == "NEEDS_APPROVAL"
    assert ledger.readiness(app)["required_complete"] == 0
    vault.set_approval(version["version_id"], True)
    assert ledger.requirement_rows(app)[0]["document_state"] == "AVAILABLE"
    vault.update_document(version["document_id"], expiry_date="2020-01-01")
    assert ledger.requirement_rows(app)[0]["document_state"] == "NEEDS_UPDATE"
    vault.update_document(version["document_id"], expiry_date=None)
    stored = vault.storage._path(version["storage_key"])
    stored.unlink()
    assert ledger.requirement_rows(app)[0]["document_state"] == "NEEDS_UPDATE"


def test_task_and_application_crud_on_empty_application(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    app = _application(ledger)
    task = ledger.create_task(app, "PORTAL", "Create portal account")
    ledger.update_task(task, description="Create and verify portal account", priority="HIGH")
    assert ledger.get("application_tasks", task)["priority"] == "HIGH"
    ledger.delete_task(task)
    ledger.delete_application(app)
    assert ledger.get("applications", app) is None


def test_complete_slice1_manual_acceptance_path(tmp_path):
    """One temporary application carries every approved Slice 1 acceptance item."""
    path = tmp_path / "acceptance.db"
    ledger, vault = Ledger(path), DocumentVault(path)
    source = ledger.create_evidence(
        "https://example.edu/graduate/doctoral", "ADMISSIONS",
        "Synthetic official-style checklist fixture", "VERIFIED",
    )
    programme = ledger.create_programme(
        "Example University", "PhD Computer Science",
        programme_url="https://example.edu/graduate/doctoral",
    )
    app = ledger.create_application(
        "2026–27", programme_id=programme,
        next_action="Confirm language waiver and request third reference",
    )
    ledger.create_deadline(app, "APPLICATION", "2027-01-15", source, timezone="Europe/London")
    transcript_requirement = ledger.create_requirement(
        app, "FORMAL_APPLICATION", "REQUIRED", "MSc transcript", source,
        normalized_document_type="TRANSCRIPT",
    )
    ledger.create_requirement(
        app, "FORMAL_APPLICATION", "REQUIRED", "Research proposal", source,
        normalized_document_type="RESEARCH_PROPOSAL",
    )
    ledger.create_requirement(
        app, "FORMAL_APPLICATION", "UNKNOWN", "English language evidence", source,
        normalized_document_type="ENGLISH_TEST",
        condition_text="Waiver applicability unresolved",
    )
    ledger.create_task(app, "VERIFY", "Check language waiver", due_at="2026-10-01")
    document = vault.upload(
        b"synthetic transcript fixture", "transcript.pdf", "SOURCE", "TRANSCRIPT",
        "Sample MSc transcript",
    )
    vault.set_approval(document["version_id"], True, "Test operator")
    vault.link_to_application(app, transcript_requirement, document["version_id"])
    assert ledger.get("applications", app)["next_action"]
    assert ledger.list_deadlines(app)[0]["source_url"] == "https://example.edu/graduate/doctoral"
    assert len(ledger.list_tasks(app)) == 1
    assert {r["document_state"] for r in ledger.requirement_rows(app)} == {"AVAILABLE", "MISSING"}
    assert ledger.readiness(app) == {
        "required_complete": 1, "required_total": 2,
        "unknown_requirements": 1, "conditional_requirements": 1,
    }
