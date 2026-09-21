"""Versioned, additive SQLite migrations for the local application ledger."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect(path: Path | str) -> sqlite3.Connection:
    db = sqlite3.connect(str(path), timeout=30, factory=ClosingConnection)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


@contextmanager
def transaction(path: Path | str):
    db = connect(path)
    try:
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


SLICE_1_SCHEMA = (
    """CREATE TABLE source_evidence (
        id INTEGER PRIMARY KEY,
        canonical_url TEXT NOT NULL,
        source_type TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        relevant_excerpt TEXT NOT NULL DEFAULT '',
        content_hash TEXT,
        verification_state TEXT NOT NULL DEFAULT 'UNVERIFIED'
            CHECK (verification_state IN ('UNVERIFIED','VERIFIED','NEEDS_REVIEW')),
        last_manually_verified_at TEXT,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE programmes (
        id INTEGER PRIMARY KEY,
        university TEXT NOT NULL,
        programme_name TEXT NOT NULL,
        department TEXT,
        degree_type TEXT,
        cycle TEXT,
        programme_url TEXT,
        admissions_url TEXT,
        portal_url TEXT,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE opportunities (
        id INTEGER PRIMARY KEY,
        programme_id INTEGER REFERENCES programmes(id) ON DELETE RESTRICT,
        opportunity_type TEXT NOT NULL CHECK (opportunity_type IN
            ('ADVERTISED_POSITION','PROGRAMME_APPLICATION','FACULTY_ENQUIRY')),
        title TEXT NOT NULL,
        canonical_url TEXT,
        institution TEXT NOT NULL,
        department_lab TEXT,
        supervisor_name TEXT,
        research_area TEXT,
        funding_text TEXT,
        eligibility_text TEXT,
        deadline_at TEXT,
        deadline_timezone TEXT,
        application_route TEXT,
        contact_policy TEXT,
        opening_status TEXT NOT NULL DEFAULT 'UNKNOWN'
            CHECK (opening_status IN ('OPEN','CLOSED','UNKNOWN')),
        verification_state TEXT NOT NULL DEFAULT 'UNVERIFIED'
            CHECK (verification_state IN ('UNVERIFIED','VERIFIED','NEEDS_REVIEW')),
        last_checked_at TEXT,
        source_evidence_id INTEGER REFERENCES source_evidence(id) ON DELETE RESTRICT,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE applications (
        id INTEGER PRIMARY KEY,
        programme_id INTEGER REFERENCES programmes(id) ON DELETE RESTRICT,
        opportunity_id INTEGER REFERENCES opportunities(id) ON DELETE RESTRICT,
        cycle TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PLANNING',
        portal_url TEXT,
        funding_state TEXT NOT NULL DEFAULT 'UNKNOWN',
        eligibility_state TEXT NOT NULL DEFAULT 'UNKNOWN',
        supervisor_contact_state TEXT NOT NULL DEFAULT 'UNKNOWN',
        next_action TEXT NOT NULL DEFAULT '',
        owner_notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (programme_id IS NOT NULL OR opportunity_id IS NOT NULL)
    )""",
    """CREATE TABLE deadlines (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE RESTRICT,
        deadline_type TEXT NOT NULL CHECK (deadline_type IN
            ('APPLICATION','FUNDING','SCHOLARSHIP','DOCUMENT','REFEREE','INTERVIEW','OTHER')),
        due_at TEXT NOT NULL,
        timezone TEXT,
        source_evidence_id INTEGER NOT NULL REFERENCES source_evidence(id) ON DELETE RESTRICT,
        verification_state TEXT NOT NULL DEFAULT 'UNVERIFIED'
            CHECK (verification_state IN ('UNVERIFIED','VERIFIED','NEEDS_REVIEW')),
        last_checked_at TEXT,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE requirements (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE RESTRICT,
        context TEXT NOT NULL CHECK (context IN
            ('FACULTY_OUTREACH','FORMAL_APPLICATION','REPLY_REQUEST')),
        requirement_state TEXT NOT NULL CHECK (requirement_state IN
            ('REQUIRED','OPTIONAL','NOT_REQUESTED','UNKNOWN')),
        original_label TEXT NOT NULL,
        normalized_document_type TEXT,
        condition_text TEXT,
        page_limit INTEGER CHECK (page_limit IS NULL OR page_limit > 0),
        word_limit INTEGER CHECK (word_limit IS NULL OR word_limit > 0),
        file_format TEXT,
        filename_rule TEXT,
        upload_field TEXT,
        source_evidence_id INTEGER NOT NULL REFERENCES source_evidence(id) ON DELETE RESTRICT,
        last_checked_at TEXT,
        fulfilled_at TEXT,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE application_tasks (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE RESTRICT,
        task_type TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'TODO'
            CHECK (status IN ('TODO','IN_PROGRESS','BLOCKED','DONE','CANCELLED')),
        due_at TEXT,
        priority TEXT NOT NULL DEFAULT 'MEDIUM'
            CHECK (priority IN ('LOW','MEDIUM','HIGH')),
        source_context TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        notes TEXT NOT NULL DEFAULT ''
    )""",
    """CREATE TABLE documents (
        id INTEGER PRIMARY KEY,
        document_class TEXT NOT NULL CHECK (document_class IN ('SOURCE','MASTER','GENERATED')),
        document_type TEXT NOT NULL,
        title TEXT NOT NULL,
        issuer TEXT,
        degree_programme TEXT,
        issue_date TEXT,
        expiry_date TEXT,
        sensitivity TEXT NOT NULL CHECK (sensitivity IN
            ('NORMAL','CONFIDENTIAL','HIGHLY_SENSITIVE')),
        permitted_storage_policy TEXT NOT NULL CHECK (permitted_storage_policy IN
            ('LOCAL_ONLY','LOCAL_OR_CLOUD')),
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE document_versions (
        id INTEGER PRIMARY KEY,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
        version_number INTEGER NOT NULL CHECK (version_number > 0),
        parent_version_id INTEGER REFERENCES document_versions(id) ON DELETE RESTRICT,
        original_filename TEXT NOT NULL,
        canonical_filename TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
        sha256 TEXT NOT NULL,
        verification_state TEXT NOT NULL DEFAULT 'UNVERIFIED'
            CHECK (verification_state IN ('UNVERIFIED','VERIFIED','NEEDS_REVIEW')),
        approval_state TEXT NOT NULL DEFAULT 'PENDING'
            CHECK (approval_state IN ('PENDING','APPROVED','REJECTED')),
        storage_backend TEXT NOT NULL,
        storage_key TEXT NOT NULL,
        uploaded_at TEXT NOT NULL,
        generation_context TEXT,
        prompt_version TEXT,
        model_used TEXT,
        master_document_version_id INTEGER REFERENCES document_versions(id) ON DELETE RESTRICT,
        claim_ids_json TEXT,
        source_evidence_ids_json TEXT,
        manual_edits TEXT,
        generated_at TEXT,
        approved_by TEXT,
        approved_at TEXT,
        notes TEXT NOT NULL DEFAULT '',
        UNIQUE (document_id, version_number)
    )""",
    """CREATE TABLE application_documents (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE RESTRICT,
        requirement_id INTEGER NOT NULL REFERENCES requirements(id) ON DELETE RESTRICT,
        document_version_id INTEGER NOT NULL REFERENCES document_versions(id) ON DELETE RESTRICT,
        document_state TEXT NOT NULL CHECK (document_state IN
            ('AVAILABLE','MISSING','NEEDS_UPDATE','NEEDS_APPROVAL')),
        source_evidence_id INTEGER REFERENCES source_evidence(id) ON DELETE RESTRICT,
        notes TEXT NOT NULL DEFAULT '',
        linked_at TEXT NOT NULL,
        UNIQUE (application_id, requirement_id)
    )""",
    """CREATE TABLE application_referees (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE RESTRICT,
        referee_name TEXT NOT NULL,
        institution TEXT,
        email TEXT,
        relationship TEXT,
        invitation_state TEXT NOT NULL DEFAULT 'NOT_REQUESTED',
        requested_at TEXT,
        deadline_at TEXT,
        submission_state TEXT NOT NULL DEFAULT 'PENDING',
        submitted_at TEXT,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_deadlines_application_due ON deadlines(application_id, due_at)",
    "CREATE INDEX idx_requirements_application ON requirements(application_id, context)",
    "CREATE INDEX idx_tasks_application_due ON application_tasks(application_id, due_at)",
    "CREATE INDEX idx_document_versions_hash ON document_versions(sha256)",
    "CREATE INDEX idx_document_versions_document ON document_versions(document_id, version_number)",
    "CREATE INDEX idx_application_documents_version ON application_documents(document_version_id)",
    """CREATE TRIGGER source_evidence_no_update BEFORE UPDATE ON source_evidence
       BEGIN SELECT RAISE(ABORT, 'source evidence is append-only'); END""",
    """CREATE TRIGGER source_evidence_no_delete BEFORE DELETE ON source_evidence
       BEGIN SELECT RAISE(ABORT, 'source evidence is append-only'); END""",
)

VERSION_METADATA_SCHEMA = (
    "ALTER TABLE document_versions ADD COLUMN issuer TEXT",
    "ALTER TABLE document_versions ADD COLUMN degree_programme TEXT",
    "ALTER TABLE document_versions ADD COLUMN issue_date TEXT",
    "ALTER TABLE document_versions ADD COLUMN expiry_date TEXT",
    "ALTER TABLE document_versions ADD COLUMN sensitivity TEXT",
    "ALTER TABLE document_versions ADD COLUMN permitted_storage_policy TEXT",
    """UPDATE document_versions SET
        issuer = (SELECT issuer FROM documents WHERE id = document_versions.document_id),
        degree_programme = (SELECT degree_programme FROM documents WHERE id = document_versions.document_id),
        issue_date = (SELECT issue_date FROM documents WHERE id = document_versions.document_id),
        expiry_date = (SELECT expiry_date FROM documents WHERE id = document_versions.document_id),
        sensitivity = (SELECT sensitivity FROM documents WHERE id = document_versions.document_id),
        permitted_storage_policy = (SELECT permitted_storage_policy FROM documents WHERE id = document_versions.document_id)""",
)

from phd_agent.migrations.slice2 import SCHEMA as SLICE_2_SCHEMA, INTEGRITY_TRIGGERS, seed_historical_faculty
from phd_agent.migrations.slice3 import SCHEMA as SLICE_3_SCHEMA
from phd_agent.migrations.slice4 import SCHEMA as SLICE_4_SCHEMA
from phd_agent.migrations.slice5 import SCHEMA as SLICE_5_SCHEMA


MIGRATIONS = (
    (1, "slice_1_application_ledger_and_document_vault", SLICE_1_SCHEMA),
    (2, "slice_1_version_metadata_snapshot", VERSION_METADATA_SCHEMA),
    (3, "slice_2_applicant_truth_and_discovery", (*SLICE_2_SCHEMA, seed_historical_faculty)),
    (4, "slice_2_immutable_review_history", INTEGRITY_TRIGGERS),
    (5, "slice_3_reviewed_materials_and_packages", SLICE_3_SCHEMA),
    (6, "slice_4_reviewed_outreach_and_contact_memory", SLICE_4_SCHEMA),
    (7, "slice_5_portal_followthrough_and_backup", SLICE_5_SCHEMA),
)


def migrate(path: Path | str) -> list[int]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    applied: list[int] = []
    with transaction(path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )""")
        existing = {r["version"]: r["name"] for r in db.execute(
            "SELECT version, name FROM schema_migrations"
        )}
        known = {version: name for version, name, _ in MIGRATIONS}
        if any(known.get(version) != name for version, name in existing.items()):
            raise RuntimeError("Database migration history differs from this application")
        for version, name, statements in MIGRATIONS:
            if version in existing:
                continue
            for statement in statements:
                if callable(statement):
                    statement(db)
                else:
                    db.execute(statement)
            db.execute(
                "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
                (version, name, utc_now()),
            )
            applied.append(version)
        failures = db.execute("PRAGMA foreign_key_check").fetchall()
        if failures:
            raise RuntimeError(f"Foreign-key integrity failed: {len(failures)} rows")
    return applied
