"""Slice 3 reviewed matching, authored materials, and frozen packages."""

SCHEMA = (
    """CREATE TABLE master_cv_versions (
        id INTEGER PRIMARY KEY, profile_id INTEGER NOT NULL REFERENCES applicant_profiles(id),
        profile_version_id INTEGER NOT NULL REFERENCES profile_versions(id),
        version_number INTEGER NOT NULL, sections_json TEXT NOT NULL,
        approval_state TEXT NOT NULL DEFAULT 'DRAFT' CHECK(approval_state IN ('DRAFT','APPROVED','REJECTED')),
        created_at TEXT NOT NULL, approved_at TEXT, approved_by TEXT,
        UNIQUE(profile_id,version_number)
    )""",
    """CREATE TABLE story_module_versions (
        id INTEGER PRIMARY KEY, profile_id INTEGER NOT NULL REFERENCES applicant_profiles(id),
        module_key TEXT NOT NULL, version_number INTEGER NOT NULL,
        content TEXT NOT NULL, claim_revision_ids_json TEXT NOT NULL,
        approval_state TEXT NOT NULL DEFAULT 'DRAFT' CHECK(approval_state IN ('DRAFT','APPROVED','REJECTED')),
        created_at TEXT NOT NULL, approved_at TEXT, approved_by TEXT,
        UNIQUE(profile_id,module_key,version_number)
    )""",
    """CREATE TABLE generated_artifacts (
        id INTEGER PRIMARY KEY, document_version_id INTEGER NOT NULL UNIQUE REFERENCES document_versions(id),
        parent_artifact_id INTEGER REFERENCES generated_artifacts(id),
        kind TEXT NOT NULL CHECK(kind IN ('CV','SOP','PERSONAL_STATEMENT','RESEARCH_STATEMENT','RESEARCH_PROPOSAL','COVER_LETTER')),
        application_id INTEGER REFERENCES applications(id), faculty_profile_id INTEGER REFERENCES faculty_profiles(id),
        profile_version_id INTEGER NOT NULL REFERENCES profile_versions(id),
        research_track_version_id INTEGER NOT NULL REFERENCES research_track_versions(id),
        master_cv_version_id INTEGER REFERENCES master_cv_versions(id),
        requirement_id INTEGER REFERENCES requirements(id),
        claim_revision_ids_json TEXT NOT NULL, story_module_version_ids_json TEXT NOT NULL DEFAULT '[]',
        publication_ids_json TEXT NOT NULL DEFAULT '[]', evidence_ids_json TEXT NOT NULL DEFAULT '[]',
        content_text TEXT NOT NULL, diff_json TEXT NOT NULL DEFAULT '{}', quality_json TEXT NOT NULL DEFAULT '{}',
        template_id TEXT NOT NULL, template_version TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
        generated_at TEXT NOT NULL, approval_state TEXT NOT NULL DEFAULT 'DRAFT'
            CHECK(approval_state IN ('DRAFT','APPROVED','REJECTED')),
        approved_at TEXT, approved_by TEXT
    )""",
    """CREATE TABLE match_reviews (
        id INTEGER PRIMARY KEY, assessment_id INTEGER NOT NULL REFERENCES match_assessments(id),
        version_number INTEGER NOT NULL, reviewer TEXT NOT NULL, annotation TEXT NOT NULL,
        override_json TEXT NOT NULL DEFAULT '{}', reviewed_at TEXT NOT NULL,
        UNIQUE(assessment_id,version_number)
    )""",
    """CREATE TABLE application_packages (
        id INTEGER PRIMARY KEY, application_id INTEGER NOT NULL REFERENCES applications(id),
        version_number INTEGER NOT NULL, context TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'DRAFT' CHECK(status IN ('DRAFT','READY')),
        profile_version_id INTEGER NOT NULL REFERENCES profile_versions(id),
        research_track_version_id INTEGER NOT NULL REFERENCES research_track_versions(id),
        requirements_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
        decisions_json TEXT NOT NULL, manifest_json TEXT NOT NULL,
        package_sha256 TEXT NOT NULL, export_path TEXT NOT NULL,
        built_at TEXT NOT NULL, approved_at TEXT, approved_by TEXT,
        preflight_run_id INTEGER REFERENCES preflight_runs(id),
        UNIQUE(application_id,version_number)
    )""",
    """CREATE TABLE package_documents (
        id INTEGER PRIMARY KEY, package_id INTEGER NOT NULL REFERENCES application_packages(id),
        document_id INTEGER NOT NULL REFERENCES documents(id),
        document_version_id INTEGER NOT NULL REFERENCES document_versions(id),
        requirement_id INTEGER REFERENCES requirements(id),
        canonical_filename TEXT NOT NULL, sha256 TEXT NOT NULL,
        inclusion_reason TEXT NOT NULL, sort_order INTEGER NOT NULL,
        UNIQUE(package_id,document_version_id)
    )""",
    """CREATE TABLE preflight_runs (
        id INTEGER PRIMARY KEY, package_id INTEGER NOT NULL REFERENCES application_packages(id),
        rule_version TEXT NOT NULL, result_json TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('PASS','WARNING','BLOCK')),
        run_at TEXT NOT NULL
    )""",
    "CREATE INDEX idx_artifact_application ON generated_artifacts(application_id,kind)",
    "CREATE INDEX idx_package_application ON application_packages(application_id,version_number)",
    """CREATE TRIGGER approved_master_cv_immutable BEFORE UPDATE ON master_cv_versions
        WHEN OLD.approval_state IN ('APPROVED','REJECTED')
        BEGIN SELECT RAISE(ABORT,'reviewed master CV is immutable'); END""",
    """CREATE TRIGGER approved_story_module_immutable BEFORE UPDATE ON story_module_versions
        WHEN OLD.approval_state IN ('APPROVED','REJECTED')
        BEGIN SELECT RAISE(ABORT,'reviewed story module is immutable'); END""",
    """CREATE TRIGGER reviewed_artifact_immutable BEFORE UPDATE ON generated_artifacts
        WHEN OLD.approval_state IN ('APPROVED','REJECTED')
        BEGIN SELECT RAISE(ABORT,'reviewed generated artifact is immutable'); END""",
    """CREATE TRIGGER package_snapshot_immutable BEFORE UPDATE ON application_packages
        WHEN OLD.requirements_json != NEW.requirements_json OR OLD.evidence_json != NEW.evidence_json
          OR OLD.decisions_json != NEW.decisions_json OR OLD.manifest_json != NEW.manifest_json
          OR OLD.package_sha256 != NEW.package_sha256 OR OLD.profile_version_id != NEW.profile_version_id
          OR OLD.research_track_version_id != NEW.research_track_version_id
        BEGIN SELECT RAISE(ABORT,'package snapshot is immutable'); END""",
    """CREATE TRIGGER package_documents_immutable BEFORE UPDATE ON package_documents
        BEGIN SELECT RAISE(ABORT,'package document snapshot is immutable'); END""",
    """CREATE TRIGGER package_documents_no_delete BEFORE DELETE ON package_documents
        BEGIN SELECT RAISE(ABORT,'package document snapshot is immutable'); END""",
    """CREATE TRIGGER approved_document_version_immutable BEFORE UPDATE ON document_versions
        WHEN OLD.approval_state = 'APPROVED' AND EXISTS
          (SELECT 1 FROM documents WHERE id=OLD.document_id AND document_class='GENERATED')
        BEGIN SELECT RAISE(ABORT,'approved document version is immutable'); END""",
    """CREATE TRIGGER match_reviews_no_update BEFORE UPDATE ON match_reviews
        BEGIN SELECT RAISE(ABORT,'match review history is append-only'); END""",
    """CREATE TRIGGER preflight_runs_no_update BEFORE UPDATE ON preflight_runs
        BEGIN SELECT RAISE(ABORT,'preflight history is append-only'); END""",
)
