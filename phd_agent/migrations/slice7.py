"""Profile-build operations, context trust levels, and university ranking columns."""

from phd_agent.university_enrichment import seed_university_rankings

SCHEMA = (
    """CREATE TABLE profile_build_operations (
        id INTEGER PRIMARY KEY,
        profile_id INTEGER REFERENCES applicant_profiles(id),
        owner_name TEXT NOT NULL,
        research_focus TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'BUILDING'
            CHECK(status IN ('BUILDING','ACTIVE','FAILED')),
        trust_level TEXT NOT NULL DEFAULT 'EXPLORATION'
            CHECK(trust_level IN ('EXPLORATION','TRUSTED')),
        exploration_context_id INTEGER REFERENCES applicant_research_contexts(id),
        trusted_context_id INTEGER REFERENCES applicant_research_contexts(id),
        created_ids_json TEXT NOT NULL DEFAULT '{}',
        error_text TEXT,
        created_at TEXT NOT NULL,
        finished_at TEXT
    )""",
    """CREATE TABLE university_rankings (
        id INTEGER PRIMARY KEY,
        canonical_name TEXT NOT NULL,
        country TEXT,
        country_code TEXT,
        qs_ranking_system TEXT NOT NULL,
        qs_ranking_year INTEGER NOT NULL,
        qs_rank_display TEXT,
        qs_rank_numeric INTEGER,
        qs_rank_band_low INTEGER,
        qs_rank_band_high INTEGER,
        source_url TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(canonical_name, qs_ranking_system, qs_ranking_year)
    )""",
    """CREATE TABLE university_aliases (
        id INTEGER PRIMARY KEY,
        ranking_id INTEGER NOT NULL REFERENCES university_rankings(id),
        alias TEXT NOT NULL,
        alias_normalized TEXT NOT NULL UNIQUE
    )""",
    "ALTER TABLE applicant_research_contexts ADD COLUMN trust_level TEXT NOT NULL DEFAULT 'TRUSTED'",
    "ALTER TABLE applicant_research_contexts ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE'",
    "ALTER TABLE applicant_research_contexts ADD COLUMN build_operation_id INTEGER",
    "DROP TRIGGER IF EXISTS applicant_context_no_update",
    """CREATE TRIGGER applicant_context_no_update BEFORE UPDATE ON applicant_research_contexts
        WHEN OLD.id!=NEW.id
          OR OLD.profile_id!=NEW.profile_id
          OR OLD.profile_version_id!=NEW.profile_version_id
          OR OLD.master_cv_version_id!=NEW.master_cv_version_id
          OR OLD.research_track_version_id!=NEW.research_track_version_id
          OR OLD.version_number!=NEW.version_number
          OR OLD.context_sha256!=NEW.context_sha256
          OR OLD.context_json!=NEW.context_json
          OR OLD.approval_basis_json!=NEW.approval_basis_json
          OR OLD.provider!=NEW.provider
          OR OLD.model!=NEW.model
          OR OLD.prompt_version!=NEW.prompt_version
          OR OLD.created_at!=NEW.created_at
          OR OLD.trust_level!=NEW.trust_level
        BEGIN SELECT RAISE(ABORT,'applicant context snapshots are immutable'); END""",
    "ALTER TABLE programmes ADD COLUMN country TEXT",
    "ALTER TABLE programmes ADD COLUMN country_code TEXT",
    "ALTER TABLE programmes ADD COLUMN country_source TEXT",
    "ALTER TABLE programmes ADD COLUMN country_match_state TEXT",
    "ALTER TABLE programmes ADD COLUMN qs_ranking_system TEXT",
    "ALTER TABLE programmes ADD COLUMN qs_ranking_year INTEGER",
    "ALTER TABLE programmes ADD COLUMN qs_rank_display TEXT",
    "ALTER TABLE programmes ADD COLUMN qs_rank_numeric INTEGER",
    "ALTER TABLE programmes ADD COLUMN qs_rank_band_low INTEGER",
    "ALTER TABLE programmes ADD COLUMN qs_rank_band_high INTEGER",
    "ALTER TABLE programmes ADD COLUMN qs_source_url TEXT",
    "ALTER TABLE programmes ADD COLUMN qs_source_evidence_id INTEGER",
    "ALTER TABLE programmes ADD COLUMN qs_checked_at TEXT",
    "ALTER TABLE programmes ADD COLUMN qs_match_state TEXT",
    seed_university_rankings,
)
