"""Slice 2 identity and discovery tables. Existing legacy rows are never updated."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = (
    """CREATE TABLE applicant_profiles (
        id INTEGER PRIMARY KEY,
        owner_name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE applicant_document_extractions (
        id INTEGER PRIMARY KEY,
        document_version_id INTEGER NOT NULL REFERENCES document_versions(id) ON DELETE RESTRICT,
        extraction_method TEXT NOT NULL,
        extracted_at TEXT NOT NULL,
        page_count INTEGER,
        extracted_text TEXT NOT NULL,
        text_sha256 TEXT NOT NULL,
        model_used TEXT,
        prompt_version TEXT,
        candidate_count INTEGER NOT NULL DEFAULT 0,
        notes TEXT NOT NULL DEFAULT ''
    )""",
    """CREATE TABLE claims (
        id INTEGER PRIMARY KEY,
        profile_id INTEGER NOT NULL REFERENCES applicant_profiles(id) ON DELETE RESTRICT,
        category TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE claim_revisions (
        id INTEGER PRIMARY KEY,
        claim_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE RESTRICT,
        version_number INTEGER NOT NULL CHECK(version_number > 0),
        claim_text TEXT NOT NULL,
        normalized_claim_type TEXT NOT NULL,
        structured_data_json TEXT NOT NULL DEFAULT '{}',
        source_document_version_id INTEGER REFERENCES document_versions(id) ON DELETE RESTRICT,
        source_evidence_id INTEGER REFERENCES source_evidence(id) ON DELETE RESTRICT,
        source_location TEXT,
        extraction_id INTEGER REFERENCES applicant_document_extractions(id) ON DELETE RESTRICT,
        classification TEXT NOT NULL CHECK(classification IN ('FACT','INFERENCE','ASPIRATION')),
        confidence REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
        verification_state TEXT NOT NULL DEFAULT 'UNVERIFIED'
            CHECK(verification_state IN ('UNVERIFIED','VERIFIED','CONFLICT')),
        review_status TEXT NOT NULL DEFAULT 'PENDING'
            CHECK(review_status IN ('PENDING','APPROVED','REJECTED')),
        approved_for_application INTEGER NOT NULL DEFAULT 0 CHECK(approved_for_application IN (0,1)),
        approved_for_outreach INTEGER NOT NULL DEFAULT 0 CHECK(approved_for_outreach IN (0,1)),
        created_at TEXT NOT NULL,
        approved_at TEXT,
        approved_by TEXT,
        notes TEXT NOT NULL DEFAULT '',
        UNIQUE(claim_id,version_number)
    )""",
    """CREATE TABLE profile_versions (
        id INTEGER PRIMARY KEY,
        profile_id INTEGER NOT NULL REFERENCES applicant_profiles(id) ON DELETE RESTRICT,
        version_number INTEGER NOT NULL CHECK(version_number > 0),
        approval_state TEXT NOT NULL DEFAULT 'DRAFT'
            CHECK(approval_state IN ('DRAFT','APPROVED')),
        source_document_version_ids_json TEXT NOT NULL DEFAULT '[]',
        claim_snapshot_sha256 TEXT,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        approved_at TEXT,
        approved_by TEXT,
        UNIQUE(profile_id,version_number)
    )""",
    """CREATE TABLE profile_version_claims (
        profile_version_id INTEGER NOT NULL REFERENCES profile_versions(id) ON DELETE RESTRICT,
        claim_revision_id INTEGER NOT NULL REFERENCES claim_revisions(id) ON DELETE RESTRICT,
        claim_snapshot_json TEXT NOT NULL,
        PRIMARY KEY(profile_version_id,claim_revision_id)
    )""",
    """CREATE TABLE research_tracks (
        id INTEGER PRIMARY KEY,
        profile_id INTEGER NOT NULL REFERENCES applicant_profiles(id) ON DELETE RESTRICT,
        title TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE','ARCHIVED')),
        priority INTEGER NOT NULL DEFAULT 3 CHECK(priority BETWEEN 1 AND 5),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE research_track_versions (
        id INTEGER PRIMARY KEY,
        track_id INTEGER NOT NULL REFERENCES research_tracks(id) ON DELETE RESTRICT,
        version_number INTEGER NOT NULL CHECK(version_number > 0),
        research_problem TEXT NOT NULL DEFAULT '',
        motivation TEXT NOT NULL DEFAULT '',
        research_gap TEXT NOT NULL DEFAULT '',
        research_questions TEXT NOT NULL DEFAULT '',
        hypotheses TEXT NOT NULL DEFAULT '',
        proposed_methodology TEXT NOT NULL DEFAULT '',
        evaluation_strategy TEXT NOT NULL DEFAULT '',
        possible_datasets TEXT NOT NULL DEFAULT '',
        expected_contribution TEXT NOT NULL DEFAULT '',
        supporting_claim_revision_ids_json TEXT NOT NULL DEFAULT '[]',
        related_projects TEXT NOT NULL DEFAULT '',
        prior_work TEXT NOT NULL DEFAULT '',
        limitations TEXT NOT NULL DEFAULT '',
        open_questions TEXT NOT NULL DEFAULT '',
        approval_state TEXT NOT NULL DEFAULT 'DRAFT'
            CHECK(approval_state IN ('DRAFT','APPROVED','REJECTED')),
        created_at TEXT NOT NULL,
        approved_at TEXT,
        approved_by TEXT,
        notes TEXT NOT NULL DEFAULT '',
        UNIQUE(track_id,version_number)
    )""",
    """CREATE TABLE target_institutions (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        state TEXT NOT NULL DEFAULT 'CONSIDERING'
            CHECK(state IN ('ACTIVE','CONSIDERING','PAUSED','ARCHIVED')),
        departments TEXT NOT NULL DEFAULT '',
        legacy_priority TEXT,
        origin TEXT NOT NULL DEFAULT 'MANUAL',
        last_verified_at TEXT,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE source_catalogue (
        id INTEGER PRIMARY KEY,
        target_id INTEGER REFERENCES target_institutions(id) ON DELETE RESTRICT,
        university TEXT NOT NULL,
        department TEXT,
        source_type TEXT NOT NULL CHECK(source_type IN
            ('PROGRAMME','OPPORTUNITY','FACULTY','LAB','FUNDING','OTHER')),
        canonical_url TEXT NOT NULL,
        discovery_strategy TEXT NOT NULL DEFAULT 'MANUAL'
            CHECK(discovery_strategy IN ('MANUAL','STATIC_HTML')),
        enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)),
        last_verified_at TEXT,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(canonical_url,source_type,university,department)
    )""",
    """CREATE TABLE faculty_profiles (
        id INTEGER PRIMARY KEY,
        legacy_professor_id INTEGER UNIQUE,
        name TEXT NOT NULL,
        institution TEXT NOT NULL,
        department TEXT,
        official_title TEXT,
        lab TEXT,
        official_profile_url TEXT,
        lab_url TEXT,
        email TEXT,
        email_state TEXT NOT NULL DEFAULT 'UNKNOWN'
            CHECK(email_state IN ('UNKNOWN','VERIFIED','UNVERIFIED','CONFLICT')),
        research_topics TEXT,
        verification_state TEXT NOT NULL DEFAULT 'NEEDS_REVERIFICATION'
            CHECK(verification_state IN ('VERIFIED','PARTIALLY_VERIFIED',
                'NEEDS_REVERIFICATION','CONFLICT','INACTIVE','UNKNOWN')),
        affiliation_state TEXT NOT NULL DEFAULT 'UNKNOWN'
            CHECK(affiliation_state IN ('CURRENT','MOVED','INACTIVE','UNKNOWN','CONFLICT')),
        supervision_state TEXT NOT NULL DEFAULT 'UNKNOWN'
            CHECK(supervision_state IN ('OPEN','CLOSED','UNKNOWN')),
        openalex_author_id TEXT,
        openalex_resolution_state TEXT NOT NULL DEFAULT 'UNRESOLVED'
            CHECK(openalex_resolution_state IN ('UNRESOLVED','RESOLVED','AMBIGUOUS','NO_MATCH')),
        last_checked_at TEXT,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE faculty_evidence_links (
        id INTEGER PRIMARY KEY,
        faculty_profile_id INTEGER NOT NULL REFERENCES faculty_profiles(id) ON DELETE RESTRICT,
        fact_type TEXT NOT NULL,
        source_evidence_id INTEGER NOT NULL REFERENCES source_evidence(id) ON DELETE RESTRICT,
        linked_at TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT '',
        UNIQUE(faculty_profile_id,fact_type,source_evidence_id)
    )""",
    """CREATE TABLE faculty_verification_events (
        id INTEGER PRIMARY KEY,
        faculty_profile_id INTEGER NOT NULL REFERENCES faculty_profiles(id) ON DELETE RESTRICT,
        previous_state TEXT NOT NULL,
        new_state TEXT NOT NULL,
        reason TEXT NOT NULL,
        evidence_ids_json TEXT NOT NULL DEFAULT '[]',
        changed_at TEXT NOT NULL
    )""",
    """CREATE TABLE faculty_candidates (
        id INTEGER PRIMARY KEY,
        source_catalogue_id INTEGER NOT NULL REFERENCES source_catalogue(id) ON DELETE RESTRICT,
        name TEXT NOT NULL,
        institution TEXT NOT NULL,
        department TEXT,
        profile_url TEXT,
        source_evidence_id INTEGER NOT NULL REFERENCES source_evidence(id) ON DELETE RESTRICT,
        review_state TEXT NOT NULL DEFAULT 'NEW'
            CHECK(review_state IN ('NEW','REVIEWED','DISMISSED')),
        created_at TEXT NOT NULL,
        UNIQUE(source_catalogue_id,profile_url,name)
    )""",
    """CREATE TABLE faculty_duplicate_reviews (
        id INTEGER PRIMARY KEY,
        faculty_profile_a_id INTEGER NOT NULL REFERENCES faculty_profiles(id) ON DELETE RESTRICT,
        faculty_profile_b_id INTEGER NOT NULL REFERENCES faculty_profiles(id) ON DELETE RESTRICT,
        reason TEXT NOT NULL,
        review_state TEXT NOT NULL DEFAULT 'PENDING'
            CHECK(review_state IN ('PENDING','DISTINCT','DUPLICATE')),
        created_at TEXT NOT NULL,
        resolved_at TEXT,
        UNIQUE(faculty_profile_a_id,faculty_profile_b_id)
    )""",
    """CREATE TABLE publications (
        id INTEGER PRIMARY KEY,
        faculty_profile_id INTEGER NOT NULL REFERENCES faculty_profiles(id) ON DELETE RESTRICT,
        title TEXT NOT NULL,
        publication_date TEXT,
        year INTEGER,
        doi TEXT,
        openalex_id TEXT,
        venue TEXT,
        authors_json TEXT NOT NULL DEFAULT '[]',
        abstract_text TEXT,
        topics_json TEXT NOT NULL DEFAULT '[]',
        source_evidence_id INTEGER NOT NULL REFERENCES source_evidence(id) ON DELETE RESTRICT,
        retrieved_at TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT '',
        UNIQUE(faculty_profile_id,openalex_id)
    )""",
    """CREATE TABLE application_faculty (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE RESTRICT,
        faculty_profile_id INTEGER NOT NULL REFERENCES faculty_profiles(id) ON DELETE RESTRICT,
        relationship_type TEXT NOT NULL DEFAULT 'POTENTIAL_SUPERVISOR',
        source_evidence_id INTEGER REFERENCES source_evidence(id) ON DELETE RESTRICT,
        notes TEXT NOT NULL DEFAULT '',
        linked_at TEXT NOT NULL,
        UNIQUE(application_id,faculty_profile_id)
    )""",
    """CREATE TABLE match_assessments (
        id INTEGER PRIMARY KEY,
        faculty_profile_id INTEGER NOT NULL REFERENCES faculty_profiles(id) ON DELETE RESTRICT,
        application_id INTEGER REFERENCES applications(id) ON DELETE RESTRICT,
        research_track_version_id INTEGER REFERENCES research_track_versions(id) ON DELETE RESTRICT,
        research_fit REAL CHECK(research_fit IS NULL OR research_fit BETWEEN 0 AND 10),
        application_readiness REAL CHECK(application_readiness IS NULL OR application_readiness BETWEEN 0 AND 10),
        research_fit_components_json TEXT NOT NULL DEFAULT '{}',
        application_readiness_components_json TEXT NOT NULL DEFAULT '{}',
        unknowns_json TEXT NOT NULL DEFAULT '[]',
        evidence_ids_json TEXT NOT NULL DEFAULT '[]',
        assessed_at TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT ''
    )""",
    "CREATE INDEX idx_claim_revisions_claim ON claim_revisions(claim_id,version_number)",
    "CREATE INDEX idx_faculty_profiles_institution ON faculty_profiles(institution,verification_state)",
    "CREATE INDEX idx_faculty_evidence_field ON faculty_evidence_links(faculty_profile_id,fact_type)",
    "CREATE INDEX idx_publications_faculty_date ON publications(faculty_profile_id,publication_date)",
    "CREATE INDEX idx_catalogue_target ON source_catalogue(target_id,source_type)",
    """CREATE TRIGGER approved_profile_immutable BEFORE UPDATE ON profile_versions
        WHEN OLD.approval_state = 'APPROVED'
        BEGIN SELECT RAISE(ABORT,'approved profile version is immutable'); END""",
    """CREATE TRIGGER approved_profile_claims_no_insert BEFORE INSERT ON profile_version_claims
        WHEN (SELECT approval_state FROM profile_versions WHERE id = NEW.profile_version_id) = 'APPROVED'
        BEGIN SELECT RAISE(ABORT,'approved profile snapshot is immutable'); END""",
    """CREATE TRIGGER approved_profile_claims_no_update BEFORE UPDATE ON profile_version_claims
        WHEN (SELECT approval_state FROM profile_versions WHERE id = OLD.profile_version_id) = 'APPROVED'
        BEGIN SELECT RAISE(ABORT,'approved profile snapshot is immutable'); END""",
    """CREATE TRIGGER approved_profile_claims_no_delete BEFORE DELETE ON profile_version_claims
        WHEN (SELECT approval_state FROM profile_versions WHERE id = OLD.profile_version_id) = 'APPROVED'
        BEGIN SELECT RAISE(ABORT,'approved profile snapshot is immutable'); END""",
    """CREATE TRIGGER approved_claim_revision_immutable BEFORE UPDATE ON claim_revisions
        WHEN OLD.review_status IN ('APPROVED','REJECTED')
        BEGIN SELECT RAISE(ABORT,'reviewed claim revision is immutable'); END""",
    """CREATE TRIGGER approved_track_version_immutable BEFORE UPDATE ON research_track_versions
        WHEN OLD.approval_state IN ('APPROVED','REJECTED')
        BEGIN SELECT RAISE(ABORT,'reviewed research track version is immutable'); END""",
    """CREATE TRIGGER faculty_events_no_update BEFORE UPDATE ON faculty_verification_events
        BEGIN SELECT RAISE(ABORT,'faculty verification history is append-only'); END""",
    """CREATE TRIGGER faculty_events_no_delete BEFORE DELETE ON faculty_verification_events
        BEGIN SELECT RAISE(ABORT,'faculty verification history is append-only'); END""",
)

INTEGRITY_TRIGGERS = (
    """CREATE TRIGGER reviewed_claim_revision_no_delete BEFORE DELETE ON claim_revisions
        WHEN OLD.review_status IN ('APPROVED','REJECTED')
        BEGIN SELECT RAISE(ABORT,'reviewed claim revision is immutable'); END""",
    """CREATE TRIGGER reviewed_track_version_no_delete BEFORE DELETE ON research_track_versions
        WHEN OLD.approval_state IN ('APPROVED','REJECTED')
        BEGIN SELECT RAISE(ABORT,'reviewed research track version is immutable'); END""",
    """CREATE TRIGGER approved_profile_no_delete BEFORE DELETE ON profile_versions
        WHEN OLD.approval_state = 'APPROVED'
        BEGIN SELECT RAISE(ABORT,'approved profile version is immutable'); END""",
    """CREATE TRIGGER faculty_evidence_links_no_update BEFORE UPDATE ON faculty_evidence_links
        BEGIN SELECT RAISE(ABORT,'faculty evidence links are append-only'); END""",
    """CREATE TRIGGER faculty_evidence_links_no_delete BEFORE DELETE ON faculty_evidence_links
        BEGIN SELECT RAISE(ABORT,'faculty evidence links are append-only'); END""",
)


def seed_historical_faculty(db: sqlite3.Connection) -> None:
    """Mirror legacy identities as unverified seeds without modifying professors."""
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='professors'").fetchone()
    if not exists:
        return
    now = _utc_now()
    db.execute("""INSERT INTO faculty_profiles
        (legacy_professor_id,name,institution,department,official_profile_url,email,
         research_topics,verification_state,created_at,updated_at)
        SELECT id,name,university,department,profile_url,email,research_interests,
               'NEEDS_REVERIFICATION',?,?
        FROM professors ORDER BY id""", (now, now))
