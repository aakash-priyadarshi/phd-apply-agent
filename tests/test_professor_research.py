"""Professor decisions, official-page extraction, and cooperative enrichment."""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from streamlit.testing.v1 import AppTest

from phd_agent.db import connect, transaction
from phd_agent.discovery import Discovery
from phd_agent.faculty_research import FacultyResearch, extract_official_research
from phd_agent.operations import OperationService
from phd_agent.orchestration import Acquisition, ProgrammeOrchestrator
from phd_agent.official_search import OfficialSearchResult
from phd_agent.profile_workspace import ProfileUpload, ProfileWorkspace


PAGE = """<html><body><h1>Professor Ada</h1><p>Example University, Computer Science.</p>
<h2>Research interests</h2><p>Reliable agents, retrieval evaluation, robustness</p>
<h2>Recent publications</h2><ul><li>2026 - Evaluating Long-Horizon Retrieval Agents</li></ul>
<h2>Current projects</h2><p>Reliable Autonomous Agents project</p>
<h2>Teaching</h2><p>Agent evaluation and AI modules for graduate students.</p>
<p>Example University faculty profile and official research information for applicant review.</p>
</body></html>"""


@pytest.fixture
def case(tmp_path):
    """Create an application with applicant context for professor research tests."""
    path = tmp_path / "phd_outreach.db"
    profile = ProfileWorkspace(path).build("Aakash Example", "Reliable agents", [ProfileUpload(
        "cv.txt", b"Aakash Example\nResearch project: I evaluated long-horizon retrieval agents with soundness checks.\n"
                  b"I aim to research robust evaluation of AI agents.\nMSc Computer Science.")])
    orchestrator = ProgrammeOrchestrator(path)
    programme = orchestrator.ledger.create_programme("Example University", "PhD Reliable AI")
    application = orchestrator.ledger.create_application("2027", programme_id=programme)
    return path, profile.context_id, application


def _discover(case, monkeypatch):
    """Configure deterministic official-page discovery for the synthetic professor."""
    path, context_id, application = case
    orchestrator = ProgrammeOrchestrator(path)

    class Provider:
        provider = "fixture"

        def search(self, *args, **kwargs):
            """Return the synthetic official faculty profile as the sole search hit."""
            return [OfficialSearchResult("Professor Ada profile", "Example University",
                "https://example.edu/faculty/ada", "FACULTY", "Agent research", "Professor Ada", "Computer Science")]

    text = BeautifulSoup(PAGE, "html.parser").get_text("\n", strip=True)
    monkeypatch.setattr(orchestrator, "acquire_page", lambda url: Acquisition(
        "ACQUIRED", "STATIC_HTTP", url, text=text, html=PAGE))
    return orchestrator, Provider(), text


def test_pending_extraction_is_sourced_and_decisions_survive_discovery(case, monkeypatch):
    """Pending extraction stays sourced while reruns preserve applicant decisions."""
    path, context_id, application = case
    orchestrator, provider, _ = _discover(case, monkeypatch)
    first = orchestrator.discover_faculty_official_web(application, context_id, provider=provider)
    candidate = first["queued"][0]
    research = FacultyResearch(path)
    snapshot = research.snapshots(candidate_id=candidate["candidate_id"])[0]
    assert snapshot["extraction_state"] == "NEEDS_REVIEW"
    assert snapshot["metadata"]["research_topics"] == ["Reliable agents", "retrieval evaluation", "robustness"]
    assert snapshot["metadata"]["recent_research_candidates"][0]["year"] == 2026
    assert snapshot["metadata"]["current_projects"] == ["Reliable Autonomous Agents project"]
    assert snapshot["source_url"] == "https://example.edu/faculty/ada"
    with connect(path) as db:
        assert db.execute("SELECT verification_state FROM source_evidence WHERE id=?",
                          (snapshot["source_evidence_id"],)).fetchone()[0] == "UNVERIFIED"
    research.decide(application, "PURSUE", candidate_id=candidate["candidate_id"])
    again = orchestrator.discover_faculty_official_web(application, context_id, provider=provider)
    assert again["queued"][0]["candidate_id"] == candidate["candidate_id"]
    assert research.decision(application, candidate_id=candidate["candidate_id"])["state"] == "PURSUE"
    research.decide(application, "REJECTED", candidate_id=candidate["candidate_id"])
    orchestrator.discover_faculty_official_web(application, context_id, provider=provider)
    assert research.decision(application, candidate_id=candidate["candidate_id"])["state"] == "REJECTED"
    research.decide(application, "UNDECIDED", candidate_id=candidate["candidate_id"])
    assert research.decision(application, candidate_id=candidate["candidate_id"])["state"] == "UNDECIDED"
    with connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM faculty_candidates").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM faculty_decision_events").fetchone()[0] == 3


def test_reviewed_card_keeps_verified_work_and_proposals_separate(case, monkeypatch):
    """Professor cards separate verified work from proposed applicant overlap."""
    path, context_id, application = case
    orchestrator, provider, _ = _discover(case, monkeypatch)
    candidate = orchestrator.discover_faculty_official_web(application, context_id, provider=provider)["queued"][0]
    research = FacultyResearch(path)
    research.decide(application, "PURSUE", candidate_id=candidate["candidate_id"])
    faculty_id = Discovery(path).review_faculty_candidate(candidate["candidate_id"])
    assert research.decision(application, faculty_id=faculty_id)["state"] == "PURSUE"
    with connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM application_faculty WHERE application_id=? AND faculty_profile_id=?",
                          (application, faculty_id)).fetchone()[0] == 1
    evidence = orchestrator.ledger.create_evidence("https://example.edu/faculty/ada", "FACULTY",
        "Professor Ada is current faculty and researches reliable agents", "VERIFIED")
    Discovery(path).verify_faculty(faculty_id, "VERIFIED", reviewer="Applicant", reason="Official page checked",
        evidence_by_fact={"IDENTITY": evidence, "AFFILIATION": evidence, "TOPICS": evidence},
        updates={"affiliation_state": "CURRENT", "research_topics": "Reliable agents, retrieval evaluation"})
    publication_evidence = orchestrator.ledger.create_evidence("https://example.edu/papers/agents", "FACULTY",
        "Professor Ada: Evaluating Long-Horizon Retrieval Agents (2026)", "VERIFIED")
    with transaction(path) as db:
        db.execute("""INSERT INTO publications (faculty_profile_id,title,year,source_evidence_id,retrieved_at)
            VALUES(?,?,?,?,datetime('now'))""", (faculty_id, "Evaluating Long-Horizon Retrieval Agents", 2026,
                                                   publication_evidence))
    cards = orchestrator.professor_cards(application, context_id)
    card = next(item for item in cards if item["faculty_id"] == faculty_id)
    assert card["recent_work"][0]["year"] == 2026
    assert card["recent_work"][0]["source_evidence_id"] == publication_evidence
    assert card["recent_work"][0]["verification_state"] == "VERIFIED"
    assert card["relevant_applicant_experience"]
    assert all("aim to" not in item.lower() for item in card["relevant_applicant_experience"])
    assert card["proposed_overlap"] != card["relevant_applicant_experience"]
    assert card["supervision_state"] == "UNKNOWN"
    assert card["current_projects"] == []  # Extraction alone is never verified.
    snapshot = research.snapshots(faculty_id=faculty_id)[0]
    research.review_snapshot(snapshot["id"], "Applicant")
    card = orchestrator.professor_cards(application, context_id)[0]
    assert card["official_recent_work"] == []  # The same title is already a verified publication.
    assert card["current_projects"][0]["source_evidence_id"] == snapshot["source_evidence_id"]
    assert card["supervision_state"] == "UNKNOWN"
    with transaction(path) as db:
        db.execute("UPDATE faculty_research_snapshots SET checked_at='2020-01-01T00:00:00+00:00' WHERE id=?",
                   (snapshot["id"],))
        db.execute("UPDATE faculty_profiles SET openalex_resolution_state='AMBIGUOUS' WHERE id=?", (faculty_id,))
    card = orchestrator.professor_cards(application, context_id)[0]
    assert card["research_freshness"] == "STALE"
    assert card["openalex_resolution_state"] == "AMBIGUOUS"
    research.decide(application, "REJECTED", faculty_id=faculty_id)
    assert orchestrator.professor_cards(application, context_id)[0]["decision_state"] == "REJECTED"
    research.decide(application, "UNDECIDED", faculty_id=faculty_id)
    assert research.decision(application, faculty_id=faculty_id)["state"] == "UNDECIDED"
    orchestrator.discover_faculty_official_web(application, context_id, provider=provider)
    with connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM faculty_profiles WHERE name='Professor Ada'").fetchone()[0] == 1


def test_deep_research_stops_resumes_and_never_inferrs_supervision(case, monkeypatch):
    """Deep research resumes safely without inferring supervision availability."""
    path, context_id, application = case
    orchestrator, provider, text = _discover(case, monkeypatch)
    candidate = orchestrator.discover_faculty_official_web(application, context_id, provider=provider)["queued"][0]
    faculty_id = Discovery(path).review_faculty_candidate(candidate["candidate_id"])
    with transaction(path) as db:
        db.execute("UPDATE faculty_profiles SET lab_url=? WHERE id=?",
                   ("https://example.edu/lab/agents", faculty_id))
    service = OperationService(path)
    operation = service.queue("FACULTY_DISCOVERY", application_id=application,
                              context_id=context_id, faculty_id=faculty_id)
    calls = []

    def fetch(url):
        """Return fixture content and request a stop after the first fetch."""
        calls.append(url)
        if len(calls) == 1:
            service.stop(operation)
        return Acquisition("ACQUIRED", "STATIC_HTTP", url, text=text, html=PAGE)

    service.run(operation, fetcher=fetch, api_key="")
    assert service.get(operation)["status"] == "CANCELLED"
    assert service.get(operation)["completed_units"] == 0
    service.resume(operation)
    service.run(operation, fetcher=fetch, api_key="")
    assert service.get(operation)["status"] == "COMPLETED"
    assert service.get(operation)["completed_units"] == 2
    assert len(service.events(operation)) >= 3
    assert len(FacultyResearch(path).snapshots(faculty_id=faculty_id)) == 2
    assert orchestrator.professor_cards(application, context_id)[0]["supervision_state"] == "UNKNOWN"
    with pytest.raises(ValueError):
        service.resume(operation)
    retry = service.queue("FACULTY_DISCOVERY", application_id=application,
                          context_id=context_id, faculty_id=faculty_id)
    service.stop(retry)
    service.resume(retry, retry=True)
    service.run(retry, fetcher=fetch, api_key="")
    assert service.get(retry)["status"] == "COMPLETED"
    assert len(FacultyResearch(path).snapshots(faculty_id=faculty_id)) == 2


def test_unlabelled_paper_does_not_become_current_project():
    """An unlabelled publication is never classified as a current project."""
    extracted = extract_official_research("Research interests: agents\nRecent publications:\n2026 - Agent paper")
    assert extracted["current_projects"] == []


def test_unchanged_research_refresh_preserves_review_but_changed_content_requires_review(case):
    """Only changed research content resets a verified snapshot to review."""
    path, _, _ = case
    discovery = Discovery(path)
    faculty_id = discovery.create_faculty("Professor Ada", "Example University",
                                          profile_url="https://example.edu/faculty/ada")
    research = FacultyResearch(path)
    url = "https://example.edu/faculty/ada"
    evidence = discovery.ledger.create_evidence(url, "FACULTY", "Research interests: reliable agents")
    metadata = {"research_interest_summary": "Reliable agents", "current_projects": []}
    snapshot_id = research.save_snapshot(url, evidence, metadata, faculty_id=faculty_id)
    research.review_snapshot(snapshot_id, "Applicant")
    same = discovery.ledger.create_evidence(url, "FACULTY", "Research interests: reliable agents")
    assert research.save_snapshot(url, same, metadata, faculty_id=faculty_id) == snapshot_id
    assert research.snapshots(faculty_id=faculty_id)[0]["extraction_state"] == "VERIFIED"
    changed = discovery.ledger.create_evidence(url, "FACULTY", "Research interests: robust agents")
    research.save_snapshot(url, changed, {**metadata, "research_interest_summary": "Robust agents"},
                           faculty_id=faculty_id)
    assert research.snapshots(faculty_id=faculty_id)[0]["extraction_state"] == "NEEDS_REVIEW"


def test_deeper_lab_page_without_professor_identity_is_not_attributed(case, monkeypatch):
    """A lab page lacking the professor's identity is not attributed to them."""
    path, context_id, application = case
    orchestrator, provider, text = _discover(case, monkeypatch)
    candidate = orchestrator.discover_faculty_official_web(application, context_id, provider=provider)["queued"][0]
    faculty_id = Discovery(path).review_faculty_candidate(candidate["candidate_id"])
    with transaction(path) as db:
        db.execute("UPDATE faculty_profiles SET lab_url=? WHERE id=?",
                   ("https://example.edu/lab/agents", faculty_id))
    service = OperationService(path)
    operation = service.queue("FACULTY_DISCOVERY", application_id=application,
                              context_id=context_id, faculty_id=faculty_id)
    anonymous = "Example University. Current projects: Agent evaluation. " * 10
    service.run(operation, api_key="", fetcher=lambda url: Acquisition(
        "ACQUIRED", "STATIC_HTTP", url,
        text=text if "/faculty/" in url else anonymous,
        html=PAGE if "/faculty/" in url else None))
    assert service.get(operation)["status"] == "COMPLETED"
    assert len(FacultyResearch(path).snapshots(faculty_id=faculty_id)) == 1
    assert any("identity review" in event["event_text"] for event in service.events(operation))


def test_people_renders_source_backed_pending_and_reviewed_cards(case, monkeypatch):
    """The People page renders sourced cards and reversible decisions."""
    path, context_id, application = case
    orchestrator, provider, _ = _discover(case, monkeypatch)
    lead = orchestrator.discover_faculty_official_web(application, context_id, provider=provider)["queued"][0]
    monkeypatch.setenv("PHD_AGENT_DATA_DIR", str(path.parent))
    monkeypatch.setenv("PHD_AGENT_ENV", "development")
    monkeypatch.setenv("PHD_AGENT_AUTH_DISABLED", "true")
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=60).run()
    app = next(r for r in app.radio if r.label == "Navigation").set_value("People").run()
    assert not app.exception
    assert any("Research interests" in item.value for item in app.markdown)
    assert any("Needs review" in item.value or "NEEDS_REVIEW" in item.value for item in app.caption)
    faculty_id = Discovery(path).review_faculty_candidate(lead["candidate_id"])
    app = app.run()
    assert not app.exception
    assert any("Professor Ada" in item.value for item in app.markdown)
    assert any("Identity: NEEDS_REVERIFICATION" in item.value for item in app.caption)
    assert any("Research deeper" == button.label for button in app.button)
    assert FacultyResearch(path).decision(application, faculty_id=faculty_id) is None
    app = next(button for button in app.button if button.label == "Reject").click().run()
    assert not app.exception
    assert FacultyResearch(path).decision(application, faculty_id=faculty_id)["state"] == "REJECTED"
    assert any("Rejected professors (1)" == item.label for item in app.expander)
    app = next(button for button in app.button if button.label == "Restore").click().run()
    assert not app.exception
    assert FacultyResearch(path).decision(application, faculty_id=faculty_id)["state"] == "UNDECIDED"
