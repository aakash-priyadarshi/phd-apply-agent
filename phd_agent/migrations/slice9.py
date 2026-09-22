"""Applicant calendar entries alongside the existing sourced deadlines and tasks."""

SCHEMA = (
    """CREATE TABLE calendar_events (
        id INTEGER PRIMARY KEY,
        application_id INTEGER REFERENCES applications(id) ON DELETE RESTRICT,
        title TEXT NOT NULL,
        event_type TEXT NOT NULL CHECK(event_type IN
            ('PERSONAL','INTERVIEW','FUNDING','DOCUMENT','FOLLOW_UP','OTHER')),
        starts_at TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE','CANCELLED')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_calendar_events_date ON calendar_events(status,starts_at)",
)
