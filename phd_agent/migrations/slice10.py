"""Application rescans over the existing operation history."""


def expand_operation_types(db) -> None:
    row = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='operations'").fetchone()
    if row and "APPLICATION_DETAIL_SCAN" in (row[0] or ""):
        return
    db.execute("ALTER TABLE operations RENAME TO operations_legacy")
    db.execute("""CREATE TABLE operations (
        id INTEGER PRIMARY KEY,
        operation_type TEXT NOT NULL CHECK(operation_type IN (
            'PROGRAMME_SEARCH','FACULTY_DISCOVERY','APPLICATION_DETAIL_SCAN','APPLICATION_FULL_REFRESH')),
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
    )""")
    db.execute("""INSERT INTO operations (
        id,operation_type,title,status,search_id,application_id,context_id,stage,current_item,
        completed_units,total_units,results_found,checkpoint_json,model_route,error_summary,
        created_at,started_at,updated_at,finished_at,cancellation_at)
        SELECT id,operation_type,title,status,search_id,application_id,context_id,stage,current_item,
        completed_units,total_units,results_found,checkpoint_json,model_route,error_summary,
        created_at,started_at,updated_at,finished_at,cancellation_at FROM operations_legacy""")
    db.execute("""CREATE TABLE operation_events_next (
        id INTEGER PRIMARY KEY,
        operation_id INTEGER NOT NULL REFERENCES operations(id),
        event_text TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    db.execute("""INSERT INTO operation_events_next (id,operation_id,event_text,created_at)
        SELECT id,operation_id,event_text,created_at FROM operation_events""")
    db.execute("DROP TABLE operation_events")
    db.execute("ALTER TABLE operation_events_next RENAME TO operation_events")
    db.execute("""CREATE TRIGGER operation_event_no_update BEFORE UPDATE ON operation_events
       BEGIN SELECT RAISE(ABORT,'operation events are append-only'); END""")
    db.execute("""CREATE TRIGGER operation_event_no_delete BEFORE DELETE ON operation_events
       BEGIN SELECT RAISE(ABORT,'operation events are append-only'); END""")
    db.execute("DROP TABLE operations_legacy")
    db.execute("CREATE INDEX idx_operations_status ON operations(status,created_at)")
    db.execute("CREATE INDEX idx_operation_events ON operation_events(operation_id,id)")


SCHEMA = (
    expand_operation_types,
    """CREATE TABLE application_scan_reports (
        id INTEGER PRIMARY KEY,
        operation_id INTEGER NOT NULL UNIQUE REFERENCES operations(id),
        application_id INTEGER NOT NULL REFERENCES applications(id),
        mode TEXT NOT NULL CHECK(mode IN ('MISSING','FULL')),
        target_fields_json TEXT NOT NULL,
        requested_fields_json TEXT NOT NULL DEFAULT '[]',
        summary_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE application_field_states (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES applications(id),
        field_name TEXT NOT NULL,
        field_value TEXT,
        state TEXT NOT NULL CHECK(state IN (
            'UNKNOWN','EXTRACTED','NEEDS_REVIEW','UNVERIFIED','OPERATOR_CONFIRMED','VERIFIED','CONFLICT')),
        source_url TEXT,
        evidence_id INTEGER REFERENCES source_evidence(id),
        excerpt TEXT NOT NULL DEFAULT '',
        extraction_method TEXT,
        checked_at TEXT,
        operation_id INTEGER REFERENCES operations(id),
        model_route TEXT,
        updated_at TEXT NOT NULL,
        UNIQUE(application_id, field_name)
    )""",
    """CREATE TABLE application_scan_findings (
        id INTEGER PRIMARY KEY,
        operation_id INTEGER NOT NULL REFERENCES operations(id),
        application_id INTEGER NOT NULL REFERENCES applications(id),
        field_name TEXT NOT NULL,
        field_value TEXT,
        state TEXT NOT NULL,
        source_url TEXT,
        evidence_id INTEGER REFERENCES source_evidence(id),
        excerpt TEXT NOT NULL DEFAULT '',
        extraction_method TEXT NOT NULL,
        checked_at TEXT NOT NULL,
        UNIQUE(operation_id, field_name)
    )""",
    """CREATE TABLE application_field_conflicts (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES applications(id),
        field_name TEXT NOT NULL,
        old_value TEXT,
        new_value TEXT,
        old_source TEXT,
        new_source TEXT,
        old_checked_at TEXT,
        new_checked_at TEXT,
        status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN','ACCEPTED','KEPT','UNRESOLVED')),
        operation_id INTEGER REFERENCES operations(id),
        evidence_id INTEGER REFERENCES source_evidence(id),
        created_at TEXT NOT NULL,
        resolved_at TEXT
    )""",
    """CREATE TABLE applicant_notifications (
        id INTEGER PRIMARY KEY,
        application_id INTEGER REFERENCES applications(id),
        operation_id INTEGER REFERENCES operations(id),
        message TEXT NOT NULL,
        created_at TEXT NOT NULL,
        read_at TEXT
    )""",
    "CREATE INDEX idx_scan_reports_application ON application_scan_reports(application_id,id)",
    "CREATE INDEX idx_field_states_application ON application_field_states(application_id,field_name)",
    "CREATE INDEX idx_field_conflicts_application ON application_field_conflicts(application_id,status)",
    "CREATE INDEX idx_applicant_notifications ON applicant_notifications(read_at,id)",
)
