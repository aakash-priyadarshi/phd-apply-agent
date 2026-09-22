"""Persistent search work, operation history, and reversible record cleanup."""

SCHEMA = (
    """CREATE TABLE search_sessions (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        intent_id INTEGER NOT NULL REFERENCES discovery_intents(id),
        context_id INTEGER NOT NULL REFERENCES applicant_research_contexts(id),
        criteria_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE','ARCHIVED')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE operations (
        id INTEGER PRIMARY KEY,
        operation_type TEXT NOT NULL CHECK(operation_type IN ('PROGRAMME_SEARCH','FACULTY_DISCOVERY')),
        title TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN
            ('QUEUED','RUNNING','PAUSED','CANCEL_REQUESTED','CANCELLED','COMPLETED','FAILED','INTERRUPTED')),
        search_id INTEGER REFERENCES search_sessions(id),
        application_id INTEGER REFERENCES applications(id),
        context_id INTEGER REFERENCES applicant_research_contexts(id),
        stage TEXT NOT NULL DEFAULT 'Queued',
        current_item TEXT,
        completed_units INTEGER NOT NULL DEFAULT 0,
        total_units INTEGER,
        results_found INTEGER NOT NULL DEFAULT 0,
        checkpoint_json TEXT NOT NULL DEFAULT '{}',
        model_route TEXT,
        error_summary TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        updated_at TEXT NOT NULL,
        finished_at TEXT,
        cancellation_at TEXT
    )""",
    """CREATE TABLE operation_events (
        id INTEGER PRIMARY KEY,
        operation_id INTEGER NOT NULL REFERENCES operations(id),
        event_text TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE record_change_events (
        id INTEGER PRIMARY KEY,
        entity_type TEXT NOT NULL,
        entity_id INTEGER NOT NULL,
        action TEXT NOT NULL,
        changes_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE record_field_reviews (
        id INTEGER PRIMARY KEY,
        entity_type TEXT NOT NULL,
        entity_id INTEGER NOT NULL,
        field_name TEXT NOT NULL,
        field_value TEXT,
        verification_state TEXT NOT NULL CHECK(verification_state IN ('OPERATOR_CONFIRMED','NEEDS_REVIEW')),
        source_url TEXT,
        recorded_at TEXT NOT NULL
    )""",
    "ALTER TABLE applications ADD COLUMN archived_at TEXT",
    "ALTER TABLE programme_candidates ADD COLUMN archived_at TEXT",
    "CREATE INDEX idx_search_intent ON search_sessions(intent_id)",
    "CREATE INDEX idx_operations_status ON operations(status,created_at)",
    "CREATE INDEX idx_operation_events ON operation_events(operation_id,id)",
    "CREATE INDEX idx_application_archive ON applications(archived_at)",
    "CREATE INDEX idx_candidate_archive ON programme_candidates(archived_at)",
    """CREATE TRIGGER operation_event_no_update BEFORE UPDATE ON operation_events
       BEGIN SELECT RAISE(ABORT,'operation events are append-only'); END""",
    """CREATE TRIGGER operation_event_no_delete BEFORE DELETE ON operation_events
       BEGIN SELECT RAISE(ABORT,'operation events are append-only'); END""",
)
