"""Calendar and daily priorities use existing facts without promoting unknowns."""

from datetime import date, timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phd_agent.db import connect
from phd_agent.ledger import Ledger
from phd_agent.planning import DailyPlanner
from phd_agent.profile_workspace import ProfileUpload, ProfileWorkspace
from phd_agent.record_controls import RecordControls


def _application(tmp_path):
    path = tmp_path / "phd_outreach.db"
    ledger = Ledger(path)
    programme_id = ledger.create_programme("Example University", "PhD in AI")
    return path, ledger, ledger.create_application("2027", programme_id=programme_id)


def test_calendar_combines_sourced_and_manual_dates_and_hides_archived(tmp_path):
    path, ledger, app_id = _application(tmp_path)
    planner = DailyPlanner(path)
    due = date.today() + timedelta(days=7)
    evidence = ledger.create_evidence("https://example.edu/admissions", "PROGRAMME")
    ledger.create_deadline(app_id, "APPLICATION", due.isoformat(), evidence,
                           verification_state="NEEDS_REVIEW")
    ledger.create_task(app_id, "FOLLOW_UP", "Ask professor", due_at=due.isoformat())
    ledger.create_referee(app_id, "Professor Smith", deadline_at=due.isoformat())
    manual_id = planner.create_event("Prepare interview", "INTERVIEW", due.isoformat(),
                                      application_id=app_id)
    personal_id = planner.create_event("Personal reminder", "PERSONAL", due.isoformat())
    items = planner.calendar(due, due)
    assert {item["kind"] for item in items} == {"DEADLINE", "TASK", "REFEREE", "MANUAL"}
    assert next(item for item in items if item["kind"] == "DEADLINE")["verification_state"] == "NEEDS_REVIEW"
    RecordControls(path).archive_application(app_id)
    assert [item["id"] for item in planner.calendar(due, due)] == [personal_id]
    RecordControls(path).archive_application(app_id, False)
    assert any(item["id"] == manual_id and item["kind"] == "MANUAL" for item in planner.calendar(due, due))


def test_manual_event_edit_cancel_restore_and_audit(tmp_path):
    path, _, app_id = _application(tmp_path)
    planner = DailyPlanner(path)
    first = date.today() + timedelta(days=3)
    second = first + timedelta(days=2)
    event_id = planner.create_event("Interview", "INTERVIEW", first.isoformat(), application_id=app_id)
    planner.edit_event(event_id, title="Interview with lab", event_type="INTERVIEW",
                       starts_at=second.isoformat(), notes="Prepare proposal")
    assert planner.calendar(first, first) == []
    assert planner.calendar(second, second)[0]["title"] == "Interview with lab"
    planner.set_event_active(event_id, False)
    assert planner.calendar(second, second) == []
    assert planner.calendar(second, second, include_cancelled=True)[0]["status"] == "CANCELLED"
    planner.set_event_active(event_id, True)
    with connect(path) as db:
        actions = [row[0] for row in db.execute("""SELECT action FROM record_change_events
            WHERE entity_type='CALENDAR_EVENT' AND entity_id=? ORDER BY id""", (event_id,))]
    assert actions == ["CREATE", "EDIT", "CANCEL", "RESTORE"]


def test_manual_event_rejects_archived_application_and_bad_dates(tmp_path):
    path, _, app_id = _application(tmp_path)
    planner = DailyPlanner(path)
    with pytest.raises(ValueError, match="ISO date"):
        planner.create_event("Interview", "INTERVIEW", "tomorrow", application_id=app_id)
    RecordControls(path).archive_application(app_id)
    with pytest.raises(ValueError, match="active application"):
        planner.create_event("Interview", "INTERVIEW", date.today().isoformat(), application_id=app_id)


def test_daily_priority_and_task_completion_are_deterministic(tmp_path):
    path, ledger, app_id = _application(tmp_path)
    planner = DailyPlanner(path)
    now = date.today()
    evidence = ledger.create_evidence("https://example.edu/admissions", "PROGRAMME")
    ledger.create_deadline(app_id, "APPLICATION", (now + timedelta(days=4)).isoformat(), evidence,
                           verification_state="NEEDS_REVIEW")
    task_id = ledger.create_task(app_id, "FOLLOW_UP", "Ask professor",
                                 due_at=(now - timedelta(days=1)).isoformat(), priority="HIGH")
    planner.create_event("Prepare interview", "INTERVIEW", (now + timedelta(days=2)).isoformat(),
                         application_id=app_id)
    dashboard = planner.dashboard(today=now)
    assert dashboard["counts"]["deadlines_14_days"] == 1
    assert dashboard["actions"][0]["title"] == "Ask professor"
    assert dashboard["actions"][0]["priority"] == "URGENT"
    assert any(item["title"].startswith("Check: Example University") for item in dashboard["actions"])
    planner.complete_task(task_id)
    assert all(item.get("task_id") != task_id for item in planner.dashboard(today=now)["actions"])
    with connect(path) as db:
        status = db.execute("SELECT status FROM application_tasks WHERE id=?", (task_id,)).fetchone()[0]
        audit = db.execute("""SELECT COUNT(*) FROM record_change_events WHERE entity_type='APPLICATION_TASK'
            AND entity_id=? AND action='COMPLETE'""", (task_id,)).fetchone()[0]
    assert status == "DONE" and audit == 1


def test_calendar_ui_and_today_priority_cards(tmp_path, monkeypatch):
    monkeypatch.setenv("PHD_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PHD_AGENT_ENV", "development")
    monkeypatch.setenv("PHD_AGENT_AUTH_DISABLED", "true")
    ProfileWorkspace(tmp_path / "phd_outreach.db").build(
        "Aakash Example", "Reliable AI agents", [ProfileUpload("cv.txt", b"""Aakash Example
        MSc Computer Science, University of Liverpool.
        Evaluated reliable AI agents in a research project. I aim to study agent evaluation.
        """)])
    path, ledger, app_id = _application(tmp_path)
    ledger.create_task(app_id, "FOLLOW_UP", "Ask professor",
                       due_at=(date.today() - timedelta(days=1)).isoformat())
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=60).run()
    assert not app.exception
    assert any("Ask professor" in item.value for item in app.markdown)
    app = next(radio for radio in app.radio if radio.label == "Navigation").set_value("Calendar").run()
    assert not app.exception
    assert any(title.value == "Calendar" for title in app.title)
    assert any(button.label == "Add event" for button in app.button)
