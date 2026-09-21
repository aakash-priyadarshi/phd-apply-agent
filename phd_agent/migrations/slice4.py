"""Slice 4: reviewed outreach, contact memory, and campaign dry runs."""

SCHEMA = (
    """CREATE TABLE campaigns (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL, cycle TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE','PAUSED','STOPPED')),
        policy_json TEXT NOT NULL, policy_version TEXT NOT NULL,
        auto_send_enabled INTEGER NOT NULL DEFAULT 0 CHECK(auto_send_enabled IN (0,1)),
        emergency_stop INTEGER NOT NULL DEFAULT 0 CHECK(emergency_stop IN (0,1)),
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE campaign_dry_runs (
        id INTEGER PRIMARY KEY, campaign_id INTEGER NOT NULL REFERENCES campaigns(id),
        policy_version TEXT NOT NULL, result_json TEXT NOT NULL,
        run_at TEXT NOT NULL
    )""",
    """CREATE TABLE outreach_packages (
        id INTEGER PRIMARY KEY, outreach_key TEXT NOT NULL, version_number INTEGER NOT NULL,
        faculty_profile_id INTEGER NOT NULL REFERENCES faculty_profiles(id),
        application_id INTEGER REFERENCES applications(id),
        campaign_id INTEGER REFERENCES campaigns(id),
        document_package_id INTEGER NOT NULL REFERENCES application_packages(id),
        stage TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'DRAFT'
            CHECK(status IN ('DRAFT','NEEDS_REVIEW','BLOCKED','APPROVED','SCHEDULED',
                'SENDING','SENT','FAILED','AMBIGUOUS_SEND','BOUNCED','REPLIED',
                'FOLLOW_UP_DUE','CANCELLED','DO_NOT_CONTACT')),
        snapshot_json TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL,
        quality_json TEXT NOT NULL, quality_version TEXT NOT NULL,
        policy_version TEXT NOT NULL, created_at TEXT NOT NULL,
        approved_at TEXT, approved_by TEXT, scheduled_at TEXT, stale_at TEXT,
        UNIQUE(outreach_key,version_number)
    )""",
    """CREATE TABLE outreach_package_documents (
        id INTEGER PRIMARY KEY, package_id INTEGER NOT NULL REFERENCES outreach_packages(id),
        document_version_id INTEGER NOT NULL REFERENCES document_versions(id),
        requirement_id INTEGER REFERENCES requirements(id),
        sha256 TEXT NOT NULL, filename TEXT NOT NULL, sort_order INTEGER NOT NULL,
        inclusion_reason TEXT NOT NULL, UNIQUE(package_id,sort_order)
    )""",
    """CREATE TABLE outreach_messages (
        id INTEGER PRIMARY KEY, outreach_key TEXT NOT NULL UNIQUE,
        package_id INTEGER NOT NULL REFERENCES outreach_packages(id),
        recipient TEXT NOT NULL, subject TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('SENDING','SENT','FAILED','AMBIGUOUS_SEND','BOUNCED','REPLIED')),
        gmail_message_id TEXT, gmail_thread_id TEXT, sent_at TEXT,
        last_error_code TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE outreach_attempts (
        id INTEGER PRIMARY KEY, message_id INTEGER NOT NULL REFERENCES outreach_messages(id),
        package_id INTEGER NOT NULL REFERENCES outreach_packages(id),
        outcome TEXT NOT NULL CHECK(outcome IN ('IN_FLIGHT','SENT','FAILED','AMBIGUOUS_SEND')),
        started_at TEXT NOT NULL, completed_at TEXT,
        gmail_message_id TEXT, gmail_thread_id TEXT, error_code TEXT
    )""",
    """CREATE TABLE gmail_threads (
        id INTEGER PRIMARY KEY, faculty_profile_id INTEGER REFERENCES faculty_profiles(id),
        recipient TEXT NOT NULL, direction TEXT NOT NULL CHECK(direction IN ('OUTBOUND','INBOUND')),
        subject TEXT NOT NULL, gmail_message_id TEXT NOT NULL UNIQUE,
        gmail_thread_id TEXT NOT NULL, message_at TEXT NOT NULL,
        outreach_package_id INTEGER REFERENCES outreach_packages(id),
        application_id INTEGER REFERENCES applications(id),
        contact_stage TEXT NOT NULL DEFAULT 'INITIAL',
        match_state TEXT NOT NULL DEFAULT 'PENDING' CHECK(match_state IN ('PENDING','CONFIRMED','REJECTED')),
        match_confidence TEXT NOT NULL DEFAULT 'UNKNOWN', source TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE gmail_reconciliation_runs (
        id INTEGER PRIMARY KEY, status TEXT NOT NULL CHECK(status IN ('COMPLETE','FAILED')),
        scanned_count INTEGER NOT NULL, matched_count INTEGER NOT NULL,
        run_at TEXT NOT NULL, source TEXT NOT NULL
    )""",
    """CREATE TABLE reply_events (
        id INTEGER PRIMARY KEY, gmail_message_id TEXT NOT NULL UNIQUE,
        gmail_thread_id TEXT NOT NULL, faculty_profile_id INTEGER REFERENCES faculty_profiles(id),
        direction TEXT NOT NULL CHECK(direction IN ('INBOUND','OUTBOUND')),
        subject_raw TEXT NOT NULL, subject_normalized TEXT NOT NULL,
        message_at TEXT NOT NULL, detected_state TEXT NOT NULL DEFAULT 'NEW',
        outreach_package_id INTEGER REFERENCES outreach_packages(id),
        application_id INTEGER REFERENCES applications(id), created_at TEXT NOT NULL
    )""",
    """CREATE TABLE contact_restrictions (
        faculty_profile_id INTEGER PRIMARY KEY REFERENCES faculty_profiles(id),
        state TEXT NOT NULL CHECK(state IN ('CLEAR','DO_NOT_CONTACT','REJECTED')),
        reason TEXT NOT NULL, reviewed_by TEXT NOT NULL, updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE outreach_events (
        id INTEGER PRIMARY KEY, package_id INTEGER REFERENCES outreach_packages(id),
        message_id INTEGER REFERENCES outreach_messages(id),
        event TEXT NOT NULL, previous_status TEXT, new_status TEXT,
        policy_version TEXT, quality_version TEXT, actor TEXT NOT NULL,
        gmail_message_id TEXT, gmail_thread_id TEXT, event_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_outreach_professor ON outreach_packages(faculty_profile_id,status)",
    "CREATE INDEX idx_contact_email ON gmail_threads(recipient,match_state)",
    "CREATE INDEX idx_gmail_thread ON gmail_threads(gmail_thread_id)",
    """CREATE TRIGGER outreach_package_snapshot_immutable BEFORE UPDATE ON outreach_packages
        WHEN OLD.outreach_key != NEW.outreach_key OR OLD.version_number != NEW.version_number
          OR OLD.faculty_profile_id != NEW.faculty_profile_id
          OR COALESCE(OLD.application_id,-1) != COALESCE(NEW.application_id,-1)
          OR COALESCE(OLD.campaign_id,-1) != COALESCE(NEW.campaign_id,-1)
          OR OLD.document_package_id != NEW.document_package_id
          OR OLD.stage != NEW.stage OR OLD.snapshot_json != NEW.snapshot_json
          OR OLD.snapshot_sha256 != NEW.snapshot_sha256
          OR OLD.quality_json != NEW.quality_json OR OLD.quality_version != NEW.quality_version
          OR OLD.policy_version != NEW.policy_version
        BEGIN SELECT RAISE(ABORT,'outreach snapshot is immutable'); END""",
    """CREATE TRIGGER outreach_package_documents_no_update BEFORE UPDATE ON outreach_package_documents
        BEGIN SELECT RAISE(ABORT,'outreach attachment snapshot is immutable'); END""",
    """CREATE TRIGGER outreach_package_documents_no_delete BEFORE DELETE ON outreach_package_documents
        BEGIN SELECT RAISE(ABORT,'outreach attachment snapshot is immutable'); END""",
    """CREATE TRIGGER outreach_events_no_update BEFORE UPDATE ON outreach_events
        BEGIN SELECT RAISE(ABORT,'outreach audit is append-only'); END""",
    """CREATE TRIGGER outreach_events_no_delete BEFORE DELETE ON outreach_events
        BEGIN SELECT RAISE(ABORT,'outreach audit is append-only'); END""",
)
