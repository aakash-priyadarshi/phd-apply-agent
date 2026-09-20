import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from phd_agent.db import connect
from phd_agent.discovery import Discovery, canonical_url, freshness
from phd_agent.ledger import Ledger
from phd_agent.openalex import OpenAlexEnrichment
from phd_agent.profile import ApplicantTruth


class Response:
    def __init__(self, data=None, text=""):
        self.data = data or {}
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class Session:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.results[url]


@pytest.fixture
def discovery(tmp_path):
    return Discovery(tmp_path / "app.db")


def test_targets_catalogue_snapshots_and_freshness(discovery, tmp_path):
    targets = tmp_path / "targets.csv"
    targets.write_text("University Name,Departments to Search,Priority\nExample University,CS,High\n", encoding="utf-8")
    assert discovery.import_targets(targets) == 1
    assert discovery.import_targets(targets) == 0
    target = discovery.list_targets()[0]
    assert target["state"] == "CONSIDERING"
    discovery.set_target_state(target["id"], "ACTIVE")
    source_id = discovery.add_source("Example University", "FACULTY", "https://example.edu/people",
                                     target_id=target["id"], strategy="STATIC_HTML")
    discovery.session = Session({"https://example.edu/people": Response(text="<nav>Ignore</nav><main>Professor Ada works on robotics.</main>")})
    first = discovery.snapshot_source(source_id)
    second = discovery.snapshot_source(source_id, excerpt="Professor Ada works on robotics.", manually_verified=True)
    assert first != second
    assert discovery.get_evidence(first)["verification_state"] == "UNVERIFIED"
    assert discovery.get_evidence(second)["verification_state"] == "VERIFIED"
    with connect(discovery.db_path) as db:
        assert db.execute("SELECT count(*) FROM source_evidence").fetchone()[0] == 2
    assert freshness((datetime.now(timezone.utc) - timedelta(days=50)).isoformat(), "FACULTY_AFFILIATION") == "STALE"
    assert freshness(None, "FACULTY_AFFILIATION") == "UNKNOWN"
    with pytest.raises(ValueError):
        canonical_url("http://127.0.0.1:8000/private")


def test_faculty_review_dedup_and_same_name_ambiguity(discovery):
    source_id = discovery.add_source("Example University", "FACULTY", "https://example.edu/ada")
    evidence_id = discovery.snapshot_source(source_id, excerpt="Ada Example, Professor at Example University. Robotics.",
                                            manually_verified=True)
    candidate = discovery.add_faculty_candidate(source_id, "Ada Example", evidence_id,
                                                "https://example.edu/ada")
    faculty_id = discovery.create_faculty("Ada Example", "Example University", profile_url="https://example.edu/ada",
                                          evidence_id=evidence_id, candidate_id=candidate)
    with pytest.raises(ValueError, match="existing"):
        discovery.create_faculty("Ada Example", "Example University")
    same_name = discovery.create_faculty("Ada Example", "Other University")
    with connect(discovery.db_path) as db:
        duplicate = db.execute("SELECT review_state,reason FROM faculty_duplicate_reviews").fetchone()
    assert duplicate["review_state"] == "PENDING"
    assert duplicate["reason"] == "SAME_NAME_OTHER_INSTITUTION"
    pending_review = discovery.list_duplicate_reviews(pending_only=True)[0]
    discovery.resolve_duplicate_review(pending_review["id"], "DISTINCT")
    assert discovery.list_duplicate_reviews(pending_only=True) == []
    with pytest.raises(ValueError, match="identity, affiliation, and topics"):
        discovery.verify_faculty(faculty_id, "VERIFIED", evidence_by_fact={}, reviewer="Operator", reason="Reviewed")
    discovery.verify_faculty(faculty_id, "VERIFIED",
                             evidence_by_fact={"IDENTITY": evidence_id, "AFFILIATION": evidence_id, "TOPICS": evidence_id},
                             reviewer="Operator", reason="Official profile reviewed",
                             updates={"research_topics": "robotics", "affiliation_state": "CURRENT"})
    detail = discovery.faculty_detail(faculty_id)
    assert detail["verification_state"] == "VERIFIED"
    assert detail["email_state"] == "UNKNOWN"
    assert len(detail["history"]) == 1
    assert detail["evidence"][0]["freshness"] == "CURRENT"
    assert discovery.faculty_detail(same_name)["verification_state"] == "NEEDS_REVERIFICATION"


def test_historical_rows_preserved_and_duplicates_only_queued(tmp_path):
    path = tmp_path / "legacy.db"
    import sqlite3
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE professors (id INTEGER PRIMARY KEY, name TEXT, university TEXT,
            department TEXT, profile_url TEXT, email TEXT, research_interests TEXT)""")
        db.executemany("INSERT INTO professors VALUES (?,?,?,?,?,?,?)", [
            (1, "Ada Example", "Example University", "CS", "https://example.edu/ada", None, "Robotics"),
            (2, "Ada Example", "Example University", "CS", "https://example.edu/ada", None, "Robotics"),
        ])
        before = db.execute("SELECT * FROM professors ORDER BY id").fetchall()
    service = Discovery(path)
    assert len(service.list_faculty()) == 2
    assert service.scan_historical_duplicates() == 1
    assert service.scan_historical_duplicates() == 0
    with connect(path) as db:
        assert [tuple(r) for r in db.execute("SELECT * FROM professors ORDER BY id")] == before
        assert db.execute("SELECT review_state FROM faculty_duplicate_reviews").fetchone()[0] == "PENDING"


def test_opportunities_and_application_conflict(discovery):
    source_id = discovery.add_source("Example University", "OPPORTUNITY", "https://example.edu/phd")
    evidence = discovery.snapshot_source(source_id, excerpt="PhD position open; apply online.", manually_verified=True)
    with pytest.raises(ValueError, match="application route"):
        discovery.add_opportunity("ADVERTISED_POSITION", "Funded PhD", "Example University", evidence,
                                  opening_status="OPEN")
    opp = discovery.add_opportunity("ADVERTISED_POSITION", "Funded PhD", "Example University", evidence,
                                    opening_status="OPEN", application_route="https://example.edu/apply")
    app = discovery.create_application_from_opportunity(opp, "2026-27")
    ledger = Ledger(discovery.db_path)
    p1 = ledger.create_programme("Example University", "PhD CS")
    p2 = ledger.create_programme("Example University", "PhD Engineering")
    assert discovery.link_to_application(app, programme_id=p1) == "LINKED"
    assert discovery.link_to_application(app, programme_id=p2, evidence_id=evidence) == "CONFLICT_TASK_CREATED"
    assert ledger.get("applications", app)["programme_id"] == p1
    assert ledger.list_tasks(app)[0]["task_type"] == "DISCOVERY_CONFLICT"
    assert discovery.propose_deadline(app, "2026-12-08", evidence) == "ADDED"
    assert discovery.propose_deadline(app, "2026-12-08", evidence) == "ALREADY_RECORDED"
    assert discovery.propose_deadline(app, "2026-12-09", evidence) == "CONFLICT_TASK_CREATED"
    assert ledger.list_deadlines(app)[0]["due_at"] == "2026-12-08"
    assert discovery.propose_requirement(app, "FORMAL_APPLICATION", "CV", "REQUIRED", evidence) == "ADDED"
    assert discovery.propose_requirement(app, "FORMAL_APPLICATION", "CV", "OPTIONAL", evidence) == "CONFLICT_TASK_CREATED"
    assert ledger.requirement_rows(app)[0]["requirement_state"] == "REQUIRED"
    assessment_faculty = discovery.create_faculty("Ada Example", "Example University")
    assessment_id = discovery.assess(assessment_faculty, research_fit=7.0,
                                     unknowns=["Currently accepting students"], evidence_ids=[evidence])
    with connect(discovery.db_path) as db:
        row = db.execute("SELECT * FROM match_assessments WHERE id=?", (assessment_id,)).fetchone()
    assert row["research_fit"] == 7.0
    assert row["application_readiness"] is None
    assert json.loads(row["unknowns_json"]) == ["Currently accepting students"]


def test_openalex_author_review_ambiguity_and_publications(discovery):
    source_id = discovery.add_source("Example University", "FACULTY", "https://example.edu/ada")
    evidence = discovery.snapshot_source(source_id, excerpt="Ada Example is a robotics professor at Example University.",
                                         manually_verified=True)
    faculty = discovery.create_faculty("Ada Example", "Example University")
    discovery.verify_faculty(faculty, "VERIFIED",
                             evidence_by_fact={"IDENTITY": evidence, "AFFILIATION": evidence, "TOPICS": evidence},
                             reviewer="Operator", reason="Official profile", updates={"research_topics": "robotics"})
    author = {"id": "https://openalex.org/A123", "display_name": "Ada Example",
              "last_known_institutions": [{"display_name": "Example University"}],
              "topics": [{"display_name": "Robotics"}], "works_count": 3}
    other = {**author, "id": "https://openalex.org/A456"}
    work = {"id": "https://openalex.org/W789", "title": "Robot evaluation",
            "publication_date": "2025-06-01", "publication_year": 2025,
            "authorships": [{"author": {"id": "https://openalex.org/A123", "display_name": "Ada Example"}}],
            "topics": [{"display_name": "Robotics"}], "primary_location": {"source": {"display_name": "Journal"}}}
    session = Session({
        "https://api.openalex.org/authors": Response({"results": [author, other]}),
        "https://api.openalex.org/authors/A123": Response(author),
        "https://api.openalex.org/works": Response({"results": [work]}),
        "https://example.edu/publications": Response(text="<main>Ada Example wrote Robot evaluation in Journal.</main>"),
    })
    enrichment = OpenAlexEnrichment(discovery.db_path, session=session)
    candidates = enrichment.search_authors(faculty)
    assert len(candidates) == 2
    assert discovery.faculty_detail(faculty)["openalex_resolution_state"] == "AMBIGUOUS"
    resolution = enrichment.resolve_author(faculty, "A123", reviewer="Operator", reason="Confirmed by topics")
    assert resolution["author_id"] == "A123"
    enrichment.search_authors(faculty)
    assert discovery.faculty_detail(faculty)["openalex_resolution_state"] == "RESOLVED"
    with pytest.raises(ValueError, match="silently replaced"):
        enrichment.resolve_author(faculty, "A456", reviewer="Operator")
    assert enrichment.enrich_works(faculty) == 1
    assert enrichment.enrich_works(faculty) == 0
    publication = discovery.faculty_detail(faculty)["publications"][0]
    assert publication["openalex_id"] == "W789"
    assert publication["abstract_text"] is None
    corroboration = enrichment.corroborate_publication(faculty, publication["id"],
        "https://example.edu/publications", reviewer="Operator")
    assert discovery.get_evidence(corroboration)["verification_state"] == "VERIFIED"
    assert any(e["fact_type"] == "OPENALEX_CORROBORATION" for e in discovery.faculty_detail(faculty)["evidence"])
    assert discovery.faculty_detail(faculty)["supervision_state"] == "UNKNOWN"
    truth = ApplicantTruth(discovery.db_path)
    profile = truth.create_profile("Applicant")
    truth.create_track(profile, "Robot evaluation", research_problem="Evaluate robot behaviour")
    relevance = discovery.publication_relevance(publication["id"], profile)
    assert relevance[0]["topic_overlap"] == ["evaluation", "robot"]
    assert relevance[0]["track_state"] == "DRAFT"
