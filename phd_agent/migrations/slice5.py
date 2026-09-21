"""Slice 5: reply follow-through, portal answers, submission archive, backup runs."""

SCHEMA = (
    """CREATE TABLE reply_classifications (
        id INTEGER PRIMARY KEY,
        reply_event_id INTEGER NOT NULL UNIQUE REFERENCES reply_events(id),
        category TEXT NOT NULL CHECK(category IN (
            'POSITIVE_INTEREST','APPLICATION_REQUESTED','PROPOSAL_REQUESTED','CV_REQUESTED',
            'COVER_LETTER_REQUESTED','ADDITIONAL_INFO_REQUESTED','MEETING_REQUESTED',
            'REFERRAL','NOT_ACCEPTING','NO_FUNDING','REJECTION','OUT_OF_OFFICE','BOUNCE','OTHER'
        )),
        confidence TEXT NOT NULL CHECK(confidence IN ('HEURISTIC','OPERATOR')),
        excerpt TEXT NOT NULL DEFAULT '',
        requested_document_type TEXT,
        requested_length TEXT,
        requested_deadline TEXT,
        classified_by TEXT NOT NULL,
        classified_at TEXT NOT NULL,
        task_id INTEGER REFERENCES application_tasks(id)
    )""",
    """CREATE TABLE answer_library (
        id INTEGER PRIMARY KEY,
        field_key TEXT NOT NULL,
        label TEXT NOT NULL,
        value_text TEXT NOT NULL,
        version_number INTEGER NOT NULL,
        claim_revision_id INTEGER REFERENCES claim_revisions(id),
        source_evidence_id INTEGER REFERENCES source_evidence(id),
        approval_state TEXT NOT NULL DEFAULT 'DRAFT'
            CHECK(approval_state IN ('DRAFT','APPROVED','REJECTED')),
        approved_by TEXT,
        approved_at TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(field_key, version_number)
    )""",
    """CREATE TABLE portal_checklist_fields (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES applications(id),
        field_key TEXT NOT NULL,
        portal_label TEXT NOT NULL,
        required INTEGER NOT NULL DEFAULT 1 CHECK(required IN (0,1)),
        answer_id INTEGER REFERENCES answer_library(id),
        value_snapshot TEXT,
        source_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'EMPTY'
            CHECK(status IN ('EMPTY','FILLED','REVIEWED','EXCLUDED')),
        reviewed_by TEXT,
        reviewed_at TEXT,
        UNIQUE(application_id, portal_label)
    )""",
    """CREATE TABLE submission_archives (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL UNIQUE REFERENCES applications(id),
        document_package_id INTEGER NOT NULL REFERENCES application_packages(id),
        confirmation_number TEXT NOT NULL,
        submitted_at TEXT NOT NULL,
        submitted_by TEXT NOT NULL,
        payment_state TEXT NOT NULL CHECK(payment_state IN
            ('NOT_APPLICABLE','PENDING_USER','RECORDED')),
        payment_reference TEXT,
        payment_recorded_at TEXT,
        answers_json TEXT NOT NULL,
        referee_json TEXT NOT NULL,
        requirement_snapshot_json TEXT NOT NULL,
        package_manifest_json TEXT NOT NULL,
        package_sha256 TEXT NOT NULL,
        archive_sha256 TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE backup_runs (
        id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL CHECK(kind IN ('BACKUP','RESTORE')),
        archive_path TEXT NOT NULL,
        db_sha256 TEXT NOT NULL,
        vault_file_count INTEGER NOT NULL,
        include_credentials INTEGER NOT NULL DEFAULT 0 CHECK(include_credentials IN (0,1)),
        manifest_json TEXT NOT NULL,
        actor TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_reply_classifications_category ON reply_classifications(category)",
    "CREATE INDEX idx_portal_checklist_application ON portal_checklist_fields(application_id,status)",
    """CREATE TRIGGER approved_answer_immutable BEFORE UPDATE ON answer_library
        WHEN OLD.approval_state IN ('APPROVED','REJECTED')
        BEGIN SELECT RAISE(ABORT,'reviewed portal answer is immutable'); END""",
    """CREATE TRIGGER submission_archives_no_update BEFORE UPDATE ON submission_archives
        BEGIN SELECT RAISE(ABORT,'submission archive is immutable'); END""",
    """CREATE TRIGGER submission_archives_no_delete BEFORE DELETE ON submission_archives
        BEGIN SELECT RAISE(ABORT,'submission archive is immutable'); END""",
    """CREATE TRIGGER backup_runs_no_update BEFORE UPDATE ON backup_runs
        BEGIN SELECT RAISE(ABORT,'backup audit is append-only'); END""",
    """CREATE TRIGGER backup_runs_no_delete BEFORE DELETE ON backup_runs
        BEGIN SELECT RAISE(ABORT,'backup audit is append-only'); END""",
)
