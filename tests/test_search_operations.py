"""Applicant-controlled searches and reversible record corrections."""

import json
from datetime import date, timedelta
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from phd_agent.db import connect
from phd_agent import launch
from phd_agent.ledger import Ledger
from phd_agent.official_search import OfficialSearchResult
from phd_agent.operations import OperationService
from phd_agent.orchestration import ProgrammeOrchestrator
from phd_agent.profile_workspace import ProfileUpload, ProfileWorkspace
from phd_agent.record_controls import RecordControls, filter_programmes, group_programmes, programme_type


CV = b"""Aakash Example
MSc Computer Science, University of Liverpool.
Research project: evaluated AI agents and retrieval systems.
Python, PyTorch and agent evaluation.
I aim to research reliable AI systems.
"""


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "phd_outreach.db"
    profile = ProfileWorkspace(path).build("Aakash Example", "Reliable AI agents",
                                           [ProfileUpload("cv.txt", CV)])
    return path, profile.context_id


class FakeProvider:
    provider = "fake"

    def __init__(self, count=3):
        self.calls = 0
        self.count = count
        self.query = ""

    def search(self, *args, **kwargs):
        self.calls += 1
        self.query = args[0]
        return [OfficialSearchResult("PhD in AI", "Example University",
                                    f"https://example.edu/phd/{index}", "PROGRAMME", "Agent evaluation")
                for index in range(self.count)]


HTML = """<html><title>PhD in AI | Example University</title><body>
<h1>PhD in AI</h1><p>Example University Department of Computer Science. September 2027 entry.
Applications close 2027-01-15. Fully funded research studentship in reliable agents.
Required documents: CV, transcript, and research proposal.</p></body></html>"""


def test_search_stop_preserves_partial_results_and_resume_from_checkpoint(workspace, monkeypatch):
    path, context_id = workspace
    service = OperationService(path)
    search = service.create_search("AI 2027", "Find funded reliable AI PhD programmes in 2027",
                                   context_id, {"country_codes": ["GB"], "programme_types": ["DPHIL"],
                                                "max_pages": 3})
    operation_id = service.queue("PROGRAMME_SEARCH", search_id=search["id"])
    provider = FakeProvider()
    original = ProgrammeOrchestrator.analyse_supplied
    calls = []

    def analyse(self, url, context_id, *, intent_id):
        calls.append(url)
        result = original(self, url, HTML, context_id, intent_id=intent_id,
                          method="UPLOADED_HTML", filename="test.html")
        if len(calls) == 1:
            service.stop(operation_id)
        return result

    monkeypatch.setattr(ProgrammeOrchestrator, "analyse_url", analyse)
    service.run(operation_id, provider=provider, api_key="")
    stopped = service.get(operation_id)
    assert stopped["status"] == "CANCELLED"
    assert stopped["results_found"] == 1
    assert len(stopped["checkpoint"]["hits"]) == 3
    assert service.events(operation_id)
    service.resume(operation_id)
    service.run(operation_id, provider=provider, api_key="")
    completed = service.get(operation_id)
    assert completed["status"] == "COMPLETED"
    assert completed["completed_units"] == 3
    assert completed["results_found"] == 3
    assert provider.calls == 1
    assert "Countries: United Kingdom" in provider.query
    assert "Programme types: DPHIL" in provider.query
    assert len(calls) == 3  # Resume skips the existing first result.
    assert len(ProgrammeOrchestrator(path).list_candidates(intent_id=search["intent_id"])) == 3


def test_queued_stop_and_restart_recovery(workspace):
    path, context_id = workspace
    service = OperationService(path)
    search = service.create_search("Saved", "Find reliable AI PhD programmes in 2027", context_id, {})
    operation_id = service.queue("PROGRAMME_SEARCH", search_id=search["id"])
    service.stop(operation_id)
    assert service.get(operation_id)["status"] == "CANCELLED"
    service.resume(operation_id)
    assert service._claim(operation_id)
    assert OperationService(path).recover_interrupted() == 1
    assert service.get(operation_id)["status"] == "INTERRUPTED"
    service.resume(operation_id, retry=True)
    assert service.get(operation_id)["checkpoint"] == {}
    assert service.get(operation_id)["status"] == "QUEUED"


def test_launcher_recovers_only_at_service_start(workspace, monkeypatch):
    path, context_id = workspace
    service = OperationService(path)
    search = service.create_search("Saved", "Find reliable AI PhD programmes in 2027", context_id, {})
    operation_id = service.queue("PROGRAMME_SEARCH", search_id=search["id"])
    monkeypatch.setattr(launch, "prepare_runtime", lambda *args, **kwargs: SimpleNamespace(database_path=path))
    launched = []
    launch.main({"PORT": "8080"}, exec_fn=lambda file, args: launched.append(args))
    assert launched
    assert service.get(operation_id)["status"] == "INTERRUPTED"


def test_detached_worker_finishes_ledger_only_search(workspace, monkeypatch):
    path, context_id = workspace
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = OperationService(path)
    search = service.create_search("Offline", "Find reliable AI PhD programmes in 2027", context_id, {})
    operation_id = service.queue("PROGRAMME_SEARCH", search_id=search["id"])
    service.launch(operation_id)
    for _ in range(100):
        result = service.get(operation_id)
        if result["status"] in {"COMPLETED", "FAILED"}:
            break
        time.sleep(.1)
    assert result["status"] == "COMPLETED", result
    assert result["stage"] == "Completed"


def test_application_edit_and_archive_preserve_history(workspace):
    path, _ = workspace
    ledger = Ledger(path)
    controls = RecordControls(path)
    programme_id = ledger.create_programme("Example University", "PhD in AI", country="United Kingdom")
    app_id = ledger.create_application("2027", programme_id=programme_id)
    with pytest.raises(ValueError, match="official source"):
        controls.edit_application(app_id, cycle="2027", status="PLANNING", portal_url="",
            next_action="", owner_notes="", eligibility_state="ELIGIBLE")
    assert ledger.get("applications", app_id)["eligibility_state"] == "UNKNOWN"
    controls.edit_application(app_id, cycle="2028", status="IN_PROGRESS",
        portal_url="https://example.edu/apply", next_action="Ask referee", owner_notes="Check funding",
        eligibility_state="ELIGIBLE", funding_state="PENDING", supervisor_contact_state="NOT_CONTACTED",
        source_url="https://example.edu/admissions")
    controls.edit_programme(programme_id, university="Example University", programme_name="DPhil in AI",
        country="United Kingdom", department="Computing", degree_type="DPhil",
        programme_url="https://example.edu/phd", source_url="https://example.edu/phd", verified=True)
    deadline_id = controls.set_deadline(app_id, "2028-01-15", source_url="https://example.edu/admissions")
    with connect(path) as db:
        assert db.execute("SELECT verification_state FROM deadlines WHERE id=?", (deadline_id,)).fetchone()[0] == "NEEDS_REVIEW"
        assert db.execute("SELECT COUNT(*) FROM record_field_reviews WHERE entity_type='APPLICATION'").fetchone()[0] == 3
    assert ledger.get("applications", app_id)["cycle"] == "2028"
    assert ledger.get("programmes", programme_id)["programme_name"] == "DPhil in AI"
    controls.archive_application(app_id)
    assert ledger.list_applications() == []
    assert ledger.get("applications", app_id)["status"] == "IN_PROGRESS"
    assert len(ledger.list_deadlines(app_id)) == 1
    controls.archive_application(app_id, False)
    assert ledger.list_applications()[0]["id"] == app_id


def test_bulk_result_cleanup_and_strict_group_filters(workspace):
    path, context_id = workspace
    service = OperationService(path)
    search = service.create_search("AI", "Find funded reliable AI PhD programmes in 2027",
                                   context_id, {"country_codes": ["GB"], "programme_types": ["DPHIL"]})
    orchestrator = ProgrammeOrchestrator(path)
    first = orchestrator.analyse_supplied("https://example.edu/one", HTML, context_id,
                                         intent_id=search["intent_id"], method="UPLOADED_HTML", filename="one.html")["candidate"]
    second = orchestrator.analyse_supplied("https://example.edu/two", HTML, context_id,
                                          intent_id=search["intent_id"], method="UPLOADED_HTML", filename="two.html")["candidate"]
    controls = RecordControls(path)
    with pytest.raises(ValueError):
        controls.archive_candidates([first["id"], 999999])
    assert len(orchestrator.list_candidates(intent_id=search["intent_id"])) == 2
    assert controls.archive_candidates([first["id"], second["id"]]) == 2
    assert orchestrator.list_candidates(intent_id=search["intent_id"]) == []
    controls.archive_candidate(first["id"], False)
    assert len(orchestrator.list_candidates(intent_id=search["intent_id"])) == 1
    items = [{"payload": {"country": "United Kingdom", "country_code": "GB", "university": "Oxford",
                          "programme": "DPhil in AI", "research_fit": 9, "funding": "funded"}},
             {"payload": {"country": None, "country_code": None, "university": "Unknown",
                          "programme": "PhD in AI", "funding": "UNKNOWN"}}]
    assert len(filter_programmes(items, search["criteria"])) == 1
    assert list(group_programmes(items, view="Country")) == ["Country unknown", "United Kingdom"]


def test_workspace_navigation_exposes_controllable_views(workspace, tmp_path, monkeypatch):
    monkeypatch.setenv("PHD_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PHD_AGENT_ENV", "development")
    monkeypatch.setenv("PHD_AGENT_AUTH_DISABLED", "true")
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=60).run()
    assert not app.exception
    navigation = next(radio for radio in app.radio if radio.label == "Navigation")
    for page in ("Find programmes", "Searches", "Operations", "Universities"):
        app = navigation.set_value(page).run()
        assert not app.exception
        navigation = next(radio for radio in app.radio if radio.label == "Navigation")


def test_workspace_search_form_queues_without_blocking_on_provider(workspace, tmp_path, monkeypatch):
    monkeypatch.setenv("PHD_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PHD_AGENT_ENV", "development")
    monkeypatch.setenv("PHD_AGENT_AUTH_DISABLED", "true")
    monkeypatch.setattr(OperationService, "launch", lambda self, operation_id: None)
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=60).run()
    app = next(r for r in app.radio if r.label == "Navigation").set_value("Find programmes").run()
    next(value for value in app.text_input if value.label == "Name this search").set_value("AI 2027")
    next(value for value in app.text_area if value.label == "What are you looking for?").set_value(
        "Find funded reliable AI PhD programmes in the UK in 2027")
    app = next(button for button in app.button if button.label == "Start search").click().run()
    assert not app.exception
    service = OperationService(tmp_path / "phd_outreach.db")
    assert service.list_searches()[0]["title"] == "AI 2027"
    assert service.list()[0]["status"] == "QUEUED"
    assert any(button.label == "Stop" for button in app.button)


def test_search_edit_audit_records_both_values(workspace):
    path, context_id = workspace
    service = OperationService(path)
    search = service.create_search("Original", "Find funded reliable AI PhD programmes in 2027",
                                   context_id, {"country_codes": ["GB"], "max_pages": 4})
    before = search["criteria"]
    after = {"country_codes": ["CA"], "max_pages": 7}
    service.update_search(search["id"], title="Canada 2027", criteria=after)
    with connect(path) as db:
        event = db.execute("""SELECT changes_json FROM record_change_events
            WHERE entity_type='SEARCH' AND entity_id=? AND action='EDIT'""", (search["id"],)).fetchone()
    changes = json.loads(event["changes_json"])
    assert changes["title"] == {"before": "Original", "after": "Canada 2027"}
    assert json.loads(changes["criteria_json"]["before"]) == before
    assert json.loads(changes["criteria_json"]["after"]) == after


def test_archived_result_is_not_rediscovered_in_same_search(workspace, monkeypatch):
    path, context_id = workspace
    service = OperationService(path)
    search = service.create_search("AI", "Find reliable AI PhD programmes in 2027", context_id, {})
    candidate = ProgrammeOrchestrator(path).analyse_supplied(
        "https://example.edu/phd/0", HTML, context_id, intent_id=search["intent_id"],
        method="UPLOADED_HTML", filename="existing.html")["candidate"]
    RecordControls(path).archive_candidate(candidate["id"])
    monkeypatch.setattr(ProgrammeOrchestrator, "analyse_url", lambda *args, **kwargs:
                        pytest.fail("Archived page was acquired again"))
    operation_id = service.queue("PROGRAMME_SEARCH", search_id=search["id"])
    service.run(operation_id, provider=FakeProvider(count=1), api_key="")
    assert service.get(operation_id)["status"] == "COMPLETED"
    assert ProgrammeOrchestrator(path).list_candidates(intent_id=search["intent_id"]) == []


def test_record_edits_and_deadline_roll_back_with_failed_audit(workspace, monkeypatch):
    path, _ = workspace
    ledger = Ledger(path)
    programme_id = ledger.create_programme("Example University", "PhD in AI")
    app_id = ledger.create_application("2027", programme_id=programme_id)
    controls = RecordControls(path)

    def fail_audit(*args):
        raise RuntimeError("Simulated audit failure")

    monkeypatch.setattr(controls, "_audit", fail_audit)
    with pytest.raises(RuntimeError, match="audit failure"):
        controls.edit_application(app_id, cycle="2028", status="IN_PROGRESS", portal_url="",
                                  next_action="Ask referee", owner_notes="")
    with pytest.raises(RuntimeError, match="audit failure"):
        controls.edit_programme(programme_id, university="Example University", programme_name="DPhil in AI",
                                country="United Kingdom", department="Computing", degree_type="DPhil",
                                programme_url="https://example.edu/phd")
    with pytest.raises(RuntimeError, match="audit failure"):
        controls.set_deadline(app_id, "2027-12-01", source_url="https://example.edu/admissions")
    assert ledger.get("applications", app_id)["cycle"] == "2027"
    assert ledger.get("programmes", programme_id)["programme_name"] == "PhD in AI"
    with connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM source_evidence").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM deadlines").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM record_field_reviews").fetchone()[0] == 0


def test_type_and_unknown_funding_normalization():
    assert programme_type("Open Research PhD in AI") == "OPEN_RESEARCH_PHD"
    assert programme_type("Open Research Doctoral Programme") == "OPEN_RESEARCH_PHD"
    assert programme_type("PhD in AI") == "PHD"
    items = [{"payload": {"funding": value, "programme": "PhD in AI"}} for value in
             (None, "", "  unknown  ", "Unknown", "Fully funded")]
    assert len(filter_programmes(items, {"funding": "Unknown"})) == 4


def test_archived_applications_do_not_create_today_alerts(workspace):
    path, _ = workspace
    ledger = Ledger(path)
    programme_id = ledger.create_programme("Example University", "PhD in AI")
    app_id = ledger.create_application("2027", programme_id=programme_id)
    evidence_id = ledger.create_evidence("https://example.edu/admissions", "PROGRAMME")
    ledger.create_deadline(app_id, "APPLICATION", (date.today() + timedelta(days=5)).isoformat(), evidence_id)
    ledger.create_task(app_id, "FOLLOW_UP", "Ask professor", due_at=(date.today() - timedelta(days=2)).isoformat())
    before = ledger.today()
    assert before["upcoming_deadlines"] and before["overdue_tasks"]
    RecordControls(path).archive_application(app_id)
    after = ledger.today()
    assert after["upcoming_deadlines"] == []
    assert after["overdue_tasks"] == []


def test_removed_results_are_scoped_to_selected_search(workspace, tmp_path, monkeypatch):
    path, context_id = workspace
    service = OperationService(path)
    first = service.create_search("AI", "Find reliable AI PhD programmes in 2027", context_id, {})
    second = service.create_search("Robotics", "Find reliable robotics PhD programmes in 2027", context_id, {})
    orchestrator = ProgrammeOrchestrator(path)
    for search, title in ((first, "PhD in AI"), (second, "PhD in Robotics")):
        candidate = orchestrator.analyse_supplied(
            f"https://example.edu/{search['id']}", HTML.replace("PhD in AI", title), context_id,
            intent_id=search["intent_id"], method="UPLOADED_HTML", filename="programme.html")["candidate"]
        RecordControls(path).archive_candidate(candidate["id"])
    monkeypatch.setenv("PHD_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PHD_AGENT_ENV", "development")
    monkeypatch.setenv("PHD_AGENT_AUTH_DISABLED", "true")
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=60).run()
    app = next(r for r in app.radio if r.label == "Navigation").set_value("Find programmes").run()
    app = next(choice for choice in app.selectbox if choice.label == "Search results").set_value(first["id"]).run()
    assert not app.exception
    values = [item.value for item in app.markdown]
    assert "PhD in AI" in values
    assert "PhD in Robotics" not in values
