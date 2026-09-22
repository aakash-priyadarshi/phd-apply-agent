"""Source-backed faculty page notes and application-specific preferences."""

SCHEMA = (
    "ALTER TABLE operations ADD COLUMN faculty_profile_id INTEGER REFERENCES faculty_profiles(id)",
    "ALTER TABLE faculty_candidates ADD COLUMN faculty_profile_id INTEGER REFERENCES faculty_profiles(id)",
    """CREATE TABLE faculty_research_snapshots (
        id INTEGER PRIMARY KEY,
        candidate_id INTEGER REFERENCES faculty_candidates(id),
        faculty_profile_id INTEGER REFERENCES faculty_profiles(id),
        source_url TEXT NOT NULL,
        source_evidence_id INTEGER NOT NULL REFERENCES source_evidence(id),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        extraction_state TEXT NOT NULL DEFAULT 'NEEDS_REVIEW'
            CHECK(extraction_state IN ('NEEDS_REVIEW','VERIFIED')),
        checked_at TEXT NOT NULL,
        reviewed_by TEXT,
        reviewed_at TEXT,
        CHECK(candidate_id IS NOT NULL OR faculty_profile_id IS NOT NULL)
    )""",
    "CREATE UNIQUE INDEX idx_faculty_research_candidate_url ON faculty_research_snapshots(candidate_id,source_url) WHERE candidate_id IS NOT NULL",
    "CREATE UNIQUE INDEX idx_faculty_research_profile_url ON faculty_research_snapshots(faculty_profile_id,source_url) WHERE faculty_profile_id IS NOT NULL",
    """CREATE TABLE faculty_decisions (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES applications(id),
        candidate_id INTEGER REFERENCES faculty_candidates(id),
        faculty_profile_id INTEGER REFERENCES faculty_profiles(id),
        state TEXT NOT NULL CHECK(state IN
            ('UNDECIDED','PURSUE','REJECTED','CONTACTED','REPLIED','ARCHIVED')),
        decided_at TEXT NOT NULL,
        CHECK(candidate_id IS NOT NULL OR faculty_profile_id IS NOT NULL)
    )""",
    "CREATE UNIQUE INDEX idx_faculty_decision_candidate ON faculty_decisions(application_id,candidate_id) WHERE candidate_id IS NOT NULL",
    "CREATE UNIQUE INDEX idx_faculty_decision_profile ON faculty_decisions(application_id,faculty_profile_id) WHERE faculty_profile_id IS NOT NULL",
    """CREATE TABLE faculty_decision_events (
        id INTEGER PRIMARY KEY,
        decision_id INTEGER NOT NULL REFERENCES faculty_decisions(id),
        previous_state TEXT NOT NULL,
        new_state TEXT NOT NULL,
        changed_at TEXT NOT NULL
    )""",
)
