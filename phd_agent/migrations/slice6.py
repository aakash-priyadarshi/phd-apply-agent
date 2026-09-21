"""Intent-first orchestration, applicant context, and supervised browser plans."""

SCHEMA = (
    """CREATE TABLE applicant_research_contexts (
        id INTEGER PRIMARY KEY,
        profile_id INTEGER NOT NULL REFERENCES applicant_profiles(id),
        profile_version_id INTEGER NOT NULL REFERENCES profile_versions(id),
        master_cv_version_id INTEGER NOT NULL REFERENCES master_cv_versions(id),
        research_track_version_id INTEGER NOT NULL REFERENCES research_track_versions(id),
        version_number INTEGER NOT NULL,
        context_sha256 TEXT NOT NULL UNIQUE,
        context_json TEXT NOT NULL,
        approval_basis_json TEXT NOT NULL,
        provider TEXT NOT NULL DEFAULT 'deterministic',
        model TEXT NOT NULL DEFAULT 'applicant-context-v1',
        prompt_version TEXT NOT NULL DEFAULT 'none',
        created_at TEXT NOT NULL,
        UNIQUE(profile_id, version_number)
    )""",
    """CREATE TABLE context_retrievals (
        id INTEGER PRIMARY KEY,
        applicant_context_id INTEGER NOT NULL REFERENCES applicant_research_contexts(id),
        task_type TEXT NOT NULL,
        query_text TEXT NOT NULL,
        selected_items_json TEXT NOT NULL,
        claim_revision_ids_json TEXT NOT NULL,
        cv_sections_json TEXT NOT NULL,
        evidence_ids_json TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE output_context_links (
        id INTEGER PRIMARY KEY,
        output_type TEXT NOT NULL CHECK(output_type IN (
            'MATCH_ASSESSMENT','GENERATED_ARTIFACT','OUTREACH_PACKAGE',
            'PORTAL_ANSWER','PROGRAMME_CANDIDATE','BROWSER_FILL_PLAN'
        )),
        output_id INTEGER NOT NULL,
        applicant_context_id INTEGER NOT NULL REFERENCES applicant_research_contexts(id),
        retrieval_id INTEGER REFERENCES context_retrievals(id),
        context_sha256 TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(output_type, output_id, applicant_context_id, retrieval_id)
    )""",
    """CREATE TABLE context_staleness (
        id INTEGER PRIMARY KEY,
        output_type TEXT NOT NULL,
        output_id INTEGER NOT NULL,
        previous_context_id INTEGER NOT NULL REFERENCES applicant_research_contexts(id),
        current_context_id INTEGER NOT NULL REFERENCES applicant_research_contexts(id),
        reason TEXT NOT NULL,
        detected_at TEXT NOT NULL,
        reviewed_at TEXT,
        reviewed_by TEXT,
        UNIQUE(output_type, output_id, current_context_id)
    )""",
    """CREATE TABLE discovery_intents (
        id INTEGER PRIMARY KEY,
        intent_text TEXT NOT NULL,
        filters_json TEXT NOT NULL DEFAULT '{}',
        applicant_context_id INTEGER NOT NULL REFERENCES applicant_research_contexts(id),
        status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE','ARCHIVED')),
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE page_ingestions (
        id INTEGER PRIMARY KEY,
        intent_id INTEGER REFERENCES discovery_intents(id),
        canonical_url TEXT NOT NULL,
        acquisition_method TEXT NOT NULL CHECK(acquisition_method IN (
            'STATIC_HTTP','PLAYWRIGHT','PASTED_TEXT','UPLOADED_HTML','UPLOADED_PDF'
        )),
        status TEXT NOT NULL CHECK(status IN ('ACQUIRED','HUMAN_INPUT_REQUIRED','FAILED')),
        content_sha256 TEXT,
        content_text TEXT NOT NULL DEFAULT '',
        failure_reason TEXT,
        source_evidence_id INTEGER REFERENCES source_evidence(id),
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE programme_candidates (
        id INTEGER PRIMARY KEY,
        intent_id INTEGER REFERENCES discovery_intents(id),
        applicant_context_id INTEGER NOT NULL REFERENCES applicant_research_contexts(id),
        ingestion_id INTEGER REFERENCES page_ingestions(id),
        source_evidence_id INTEGER REFERENCES source_evidence(id),
        canonical_url TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        field_evidence_json TEXT NOT NULL,
        confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
        review_state TEXT NOT NULL DEFAULT 'NEW'
            CHECK(review_state IN ('NEW','SHORTLISTED','REJECTED','ACCEPTED')),
        accepted_programme_id INTEGER REFERENCES programmes(id),
        accepted_application_id INTEGER REFERENCES applications(id),
        created_at TEXT NOT NULL,
        reviewed_at TEXT,
        reviewed_by TEXT
    )""",
    """CREATE TABLE browser_fill_plans (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES applications(id),
        page_url TEXT NOT NULL,
        field_count INTEGER NOT NULL,
        safe_count INTEGER NOT NULL,
        review_count INTEGER NOT NULL,
        manual_count INTEGER NOT NULL,
        plan_json TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'REVIEW_REQUIRED'
            CHECK(status IN ('REVIEW_REQUIRED','APPROVED','FILLED')),
        created_at TEXT NOT NULL,
        approved_at TEXT,
        approved_by TEXT
    )""",
    """CREATE TABLE workload_events (
        id INTEGER PRIMARY KEY,
        event_type TEXT NOT NULL,
        count_value INTEGER NOT NULL CHECK(count_value >= 0),
        entity_type TEXT,
        entity_id INTEGER,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_context_profile ON applicant_research_contexts(profile_id,version_number)",
    "CREATE INDEX idx_retrieval_context ON context_retrievals(applicant_context_id,task_type)",
    "CREATE INDEX idx_output_context ON output_context_links(output_type,output_id)",
    "CREATE INDEX idx_candidate_review ON programme_candidates(review_state,created_at)",
    "CREATE INDEX idx_ingestion_intent ON page_ingestions(intent_id,created_at)",
    "CREATE INDEX idx_workload_type ON workload_events(event_type,created_at)",
    """CREATE TRIGGER applicant_context_no_update BEFORE UPDATE ON applicant_research_contexts
        BEGIN SELECT RAISE(ABORT,'applicant context snapshots are immutable'); END""",
    """CREATE TRIGGER applicant_context_no_delete BEFORE DELETE ON applicant_research_contexts
        BEGIN SELECT RAISE(ABORT,'applicant context snapshots are immutable'); END""",
    """CREATE TRIGGER context_retrieval_no_update BEFORE UPDATE ON context_retrievals
        BEGIN SELECT RAISE(ABORT,'context retrieval provenance is immutable'); END""",
    """CREATE TRIGGER context_retrieval_no_delete BEFORE DELETE ON context_retrievals
        BEGIN SELECT RAISE(ABORT,'context retrieval provenance is immutable'); END""",
    """CREATE TRIGGER output_context_link_no_update BEFORE UPDATE ON output_context_links
        BEGIN SELECT RAISE(ABORT,'output context provenance is immutable'); END""",
    """CREATE TRIGGER output_context_link_no_delete BEFORE DELETE ON output_context_links
        BEGIN SELECT RAISE(ABORT,'output context provenance is immutable'); END""",
    """CREATE TRIGGER page_ingestion_no_update BEFORE UPDATE ON page_ingestions
        BEGIN SELECT RAISE(ABORT,'page ingestion evidence is append-only'); END""",
    """CREATE TRIGGER page_ingestion_no_delete BEFORE DELETE ON page_ingestions
        BEGIN SELECT RAISE(ABORT,'page ingestion evidence is append-only'); END""",
    """CREATE TRIGGER accepted_candidate_immutable BEFORE UPDATE ON programme_candidates
        WHEN OLD.review_state='ACCEPTED'
        BEGIN SELECT RAISE(ABORT,'accepted programme candidate is immutable'); END""",
    """CREATE TRIGGER programme_candidate_no_delete BEFORE DELETE ON programme_candidates
        BEGIN SELECT RAISE(ABORT,'programme candidate review history is retained'); END""",
    """CREATE TRIGGER browser_fill_plan_snapshot_immutable BEFORE UPDATE ON browser_fill_plans
        WHEN OLD.application_id!=NEW.application_id OR OLD.page_url!=NEW.page_url
          OR OLD.field_count!=NEW.field_count OR OLD.safe_count!=NEW.safe_count
          OR OLD.review_count!=NEW.review_count OR OLD.manual_count!=NEW.manual_count
          OR OLD.plan_json!=NEW.plan_json
        BEGIN SELECT RAISE(ABORT,'browser fill plan snapshot is immutable'); END""",
    """CREATE TRIGGER workload_event_no_update BEFORE UPDATE ON workload_events
        BEGIN SELECT RAISE(ABORT,'workload history is append-only'); END""",
    """CREATE TRIGGER workload_event_no_delete BEFORE DELETE ON workload_events
        BEGIN SELECT RAISE(ABORT,'workload history is append-only'); END""",
)
