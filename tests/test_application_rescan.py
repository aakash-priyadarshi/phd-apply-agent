"""Official-source scans fill unknown application facts without overwriting confirmed ones."""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phd_agent.application_rescan import PAGE_BUDGET, ApplicationRescan, official_links, parse_page
from phd_agent.db import connect, utc_now
from phd_agent.ledger import Ledger
from phd_agent.operations import OperationService
from phd_agent.planning import DailyPlanner
from phd_agent.profile_workspace import ProfileUpload, ProfileWorkspace


PROGRAMME = "https://example.edu/cs/phd"
ADMISSIONS = "https://example.edu/admissions"
RICH = """Applications close on 3 December 2026 at 17:00 GMT.
The application fee is £75.
IELTS 7.0 overall is required.
Three referees are required.
Applicants are encouraged to contact a supervisor before applying.
A Clarendon studentship is available.
A research proposal is required.
September 2027 entry. Applications are open.
"""
RICH_HTML = RICH + '<a href="https://example.edu/apply">Apply online</a><a href="https://evil.test/funding">Funding</a>'


def _application(tmp_path, **details):
    path = tmp_path / "phd_outreach.db"
    ledger = Ledger(path)
    programme_id = ledger.create_programme(
        "Example University", "PhD Computer Science", programme_url=PROGRAMME, **details)
    return path, ledger, ledger.create_application("2027", programme_id=programme_id)


def _pages(mapping):
    calls = []

    def fetch(url):
        calls.append(url)
        return mapping[url]

    fetch.calls = calls
    return fetch


def _acquired(url, text, html=""):
    return {"status": "ACQUIRED", "url": url, "text": text, "html": html or text}


def test_scan_identifies_missing_targets_and_skips_fresh_verified_values(tmp_path):
    path, ledger, app_id = _application(tmp_path)
    evidence = ledger.create_evidence("https://example.edu/old", "PROGRAMME", "Applications close 2026-12-03.", "VERIFIED")
    ledger.create_deadline(app_id, "APPLICATION", "2026-12-03", evidence, verification_state="VERIFIED",
                           last_checked_at=utc_now())
    scanner = ApplicationRescan(path)
    missing = scanner.plan_targets(app_id)
    assert "deadline" not in missing
    assert "application_fee" in missing
    assert "supervisor_contact_policy" in missing
    assert "deadline" in scanner.plan_targets(app_id, mode="FULL")


def test_stale_verified_deadline_is_included_in_a_missing_scan(tmp_path):
    path, ledger, app_id = _application(tmp_path)
    evidence = ledger.create_evidence("https://example.edu/old", "PROGRAMME", "Applications close 2026-12-03.")
    ledger.create_deadline(app_id, "APPLICATION", "2026-12-03", evidence, verification_state="VERIFIED",
                           last_checked_at=(date.today() - timedelta(days=200)).isoformat())
    assert "deadline" in ApplicationRescan(path).plan_targets(app_id)


def test_same_domain_budget_early_stop_and_deterministic_extraction(tmp_path):
    path, _, app_id = _application(tmp_path, admissions_url=ADMISSIONS)
    links = "".join(f'<a href="https://example.edu/funding-{index}">funding</a>' for index in range(15))
    filler = "This official programme page is long enough to read but states no fee or deadline. " * 2
    fetch = _pages({
        PROGRAMME: _acquired(PROGRAMME, filler, filler + links + '<a href="https://evil.test/funding">funding</a>'),
        **{f"https://example.edu/funding-{index}": _acquired(f"https://example.edu/funding-{index}", filler) for index in range(15)},
    })
    scanner = ApplicationRescan(path)
    service = OperationService(path)
    operation_id = scanner.queue(app_id, fields=["application_fee"])
    service.run(operation_id, fetcher=fetch)
    assert len(fetch.calls) <= PAGE_BUDGET
    assert all(url.startswith("https://example.edu/") for url in fetch.calls)
    events = " ".join(event["event_text"] for event in service.events(operation_id))
    assert f"Page budget: {PAGE_BUDGET}" in events
    early = _pages({PROGRAMME: _acquired(PROGRAMME, "The application fee is £75. " * 4,
                                        "The application fee is £75. " * 4 + '<a href="/admissions">Admissions</a>')})
    second = scanner.queue(app_id, fields=["application_fee"])
    service.run(second, fetcher=early)
    assert early.calls == [PROGRAMME]
    assert scanner.facts(app_id)["application_fee"].value == "£75"


def test_cancellation_preserves_partial_results_and_resume_skips_finished_pages(tmp_path):
    path, _, app_id = _application(tmp_path, admissions_url=ADMISSIONS)
    service = OperationService(path)
    scanner = ApplicationRescan(path)
    operation_id = scanner.queue(app_id, fields=["application_fee", "deadline"])

    def fetch(url):
        fetch.calls.append(url)
        if url == PROGRAMME:
            service.stop(operation_id)
            return _acquired(url, "The application fee is £75. " * 4)
        return _acquired(url, "Applications close on 8 January 2027. " * 3)

    fetch.calls = []
    service.run(operation_id, fetcher=fetch)
    assert service.get(operation_id)["status"] == "CANCELLED"
    assert scanner.facts(app_id)["application_fee"].value == "£75"
    assert scanner.facts(app_id)["deadline"].value in {None, "UNKNOWN"} or scanner.facts(app_id)["deadline"].state == "UNKNOWN"
    service.resume(operation_id)
    service.run(operation_id, fetcher=fetch)
    assert service.get(operation_id)["status"] == "COMPLETED"
    assert fetch.calls == [PROGRAMME, ADMISSIONS]
    assert scanner.facts(app_id)["deadline"].value == "2027-01-08"
    report = scanner.latest_report(app_id)["summary"]
    assert any(item["field"] == "application_fee" for item in report["resolved"])


def test_duplicate_running_scan_is_rejected(tmp_path):
    path, _, app_id = _application(tmp_path)
    scanner = ApplicationRescan(path)
    scanner.queue(app_id, fields=["application_fee"])
    with pytest.raises(ValueError, match="Scan already running"):
        scanner.queue(app_id, fields=["application_fee"])
    assert scanner.queue(app_id, mode="FULL", fields=["deadline"])


def test_unknown_fee_becomes_extracted_with_evidence_and_confirmed_deadline_stays(tmp_path):
    path, ledger, app_id = _application(tmp_path)
    with connect(path) as db:
        db.execute("""INSERT INTO record_field_reviews
            (entity_type,entity_id,field_name,field_value,verification_state,source_url,recorded_at)
            VALUES('APPLICATION',?,'deadline','2026-12-03','OPERATOR_CONFIRMED','https://example.edu/old',?)""",
            (app_id, utc_now()))
    evidence = ledger.create_evidence("https://example.edu/old", "PROGRAMME", "Applications close 2026-12-03.", "VERIFIED")
    ledger.create_deadline(app_id, "APPLICATION", "2026-12-03", evidence, verification_state="VERIFIED",
                           last_checked_at=utc_now())
    fetch = _pages({PROGRAMME: _acquired(PROGRAMME, RICH, RICH_HTML)})
    scanner = ApplicationRescan(path)
    service = OperationService(path)
    operation_id = scanner.queue(app_id)
    service.run(operation_id, fetcher=fetch)
    fee = scanner.facts(app_id)["application_fee"]
    assert fee.value == "£75" and fee.state == "EXTRACTED"
    with connect(path) as db:
        stored = db.execute("SELECT evidence_id,source_url,excerpt,operation_id FROM application_field_states WHERE application_id=? AND field_name='application_fee'",
                            (app_id,)).fetchone()
        deadline = db.execute("SELECT due_at,verification_state FROM deadlines WHERE application_id=? AND deadline_type='APPLICATION'",
                              (app_id,)).fetchone()
    assert stored["evidence_id"] and stored["source_url"] == PROGRAMME and "£75" in stored["excerpt"]
    assert stored["operation_id"] == operation_id
    assert deadline["due_at"].startswith("2026-12-03") and deadline["verification_state"] == "VERIFIED"
    assert service.get(operation_id)["model_route"]
    assert "luna" in service.get(operation_id)["model_route"].casefold()
    assert "sol" not in service.get(operation_id)["model_route"].casefold()


def test_verified_change_creates_a_conflict_until_the_applicant_accepts_it(tmp_path):
    path, ledger, app_id = _application(tmp_path)
    evidence = ledger.create_evidence("https://example.edu/old", "PROGRAMME", "Applications close 2026-12-03.", "VERIFIED")
    ledger.create_deadline(app_id, "APPLICATION", "2026-12-03", evidence, verification_state="VERIFIED",
                           last_checked_at=utc_now())
    fetch = _pages({PROGRAMME: _acquired(PROGRAMME, "Applications close on 8 January 2027. " * 3)})
    scanner = ApplicationRescan(path)
    service = OperationService(path)
    operation_id = scanner.queue(app_id, mode="FULL", fields=["deadline"])
    service.run(operation_id, fetcher=fetch)
    assert ledger.list_deadlines(app_id)[0]["due_at"].startswith("2026-12-03")
    conflict = scanner.conflicts(app_id)[0]
    assert conflict["old_value"].startswith("2026-12-03") and conflict["new_value"] == "2027-01-08"
    assert "sol" in service.get(operation_id)["model_route"].casefold()
    scanner.resolve_conflict(conflict["id"], "accept")
    assert ledger.list_deadlines(app_id)[0]["due_at"].startswith("2027-01-08")
    with connect(path) as db:
        actions = [row[0] for row in db.execute(
            "SELECT action FROM record_change_events WHERE entity_type='APPLICATION' AND entity_id=? ORDER BY id",
            (app_id,))]
    assert "CONFLICT_CREATED" in actions and "CONFLICT_ACCEPTED" in actions
    assert scanner.facts(app_id)["deadline"].state == "OPERATOR_CONFIRMED"


def test_frozen_submission_archive_is_not_rewritten(tmp_path):
    path, _, app_id = _application(tmp_path)
    now = utc_now()
    with connect(path) as db:
        profile_id = db.execute("INSERT INTO applicant_profiles(owner_name,created_at,updated_at) VALUES(?,?,?)",
                                ("Applicant", now, now)).lastrowid
        profile_version = db.execute("INSERT INTO profile_versions(profile_id,version_number,created_at) VALUES(?,1,?)",
                                     (profile_id, now)).lastrowid
        track_id = db.execute("INSERT INTO research_tracks(profile_id,title,created_at,updated_at) VALUES(?,?,?,?)",
                              (profile_id, "Agents", now, now)).lastrowid
        track_version = db.execute("INSERT INTO research_track_versions(track_id,version_number,created_at) VALUES(?,1,?)",
                                   (track_id, now)).lastrowid
        package_id = db.execute("""INSERT INTO application_packages
            (application_id,version_number,context,profile_version_id,research_track_version_id,
             requirements_json,evidence_json,decisions_json,manifest_json,package_sha256,export_path,built_at)
            VALUES(?,1,'FORMAL_APPLICATION',?,?, '{}','{}','{}','{}','abc','export',?)""",
            (app_id, profile_version, track_version, now)).lastrowid
        db.execute("""INSERT INTO submission_archives
            (application_id,document_package_id,confirmation_number,submitted_at,submitted_by,payment_state,
             answers_json,referee_json,requirement_snapshot_json,package_manifest_json,package_sha256,archive_sha256,created_at)
            VALUES(?,?,'CONF-1',?,'Applicant','NOT_APPLICABLE','{}','[]','[]','{}','abc','def',?)""",
            (app_id, package_id, now, now))
    raw = sqlite3.connect(path)
    before = raw.execute("SELECT * FROM submission_archives").fetchone()
    fetch = _pages({PROGRAMME: _acquired(PROGRAMME, RICH, RICH_HTML)})
    service = OperationService(path)
    service.run(ApplicationRescan(path).queue(app_id), fetcher=fetch)
    after = raw.execute("SELECT * FROM submission_archives").fetchone()
    raw.close()
    assert before == after


def test_resume_uses_supplied_page_text_without_a_second_fetch(tmp_path):
    path, _, app_id = _application(tmp_path)
    fetch = _pages({PROGRAMME: {"status": "HUMAN_INPUT_REQUIRED", "url": PROGRAMME, "text": "", "html": ""}})
    scanner = ApplicationRescan(path)
    service = OperationService(path)
    operation_id = scanner.queue(app_id, fields=["application_fee"])
    service.run(operation_id, fetcher=fetch)
    assert service.get(operation_id)["status"] == "PAUSED"
    scanner.supply(operation_id, PROGRAMME, "The application fee is £75. " * 4)
    service.resume(operation_id)
    service.run(operation_id, fetcher=fetch)
    assert fetch.calls == [PROGRAMME]
    assert scanner.facts(app_id)["application_fee"].value == "£75"


def test_retry_does_not_duplicate_evidence_for_an_unchanged_fact(tmp_path):
    path, _, app_id = _application(tmp_path)
    fetch = _pages({PROGRAMME: _acquired(PROGRAMME, "The application fee is £75. " * 4)})
    scanner = ApplicationRescan(path)
    service = OperationService(path)
    operation_id = scanner.queue(app_id, fields=["application_fee"])
    service.run(operation_id, fetcher=fetch)
    with connect(path) as db:
        before = db.execute("SELECT COUNT(*) FROM source_evidence").fetchone()[0]
    with pytest.raises(ValueError, match="Nothing missing to scan"):
        scanner.queue(app_id, fields=["application_fee"])
    with connect(path) as db:
        after = db.execute("SELECT COUNT(*) FROM source_evidence").fetchone()[0]
    assert after == before
    assert len(scanner.history(app_id)) == 1


def test_scan_history_and_one_completion_notification(tmp_path):
    path, _, app_id = _application(tmp_path)
    fetch = _pages({PROGRAMME: _acquired(PROGRAMME, "The application fee is £75. " * 4)})
    scanner = ApplicationRescan(path)
    service = OperationService(path)
    service.run(scanner.queue(app_id, fields=["application_fee"]), fetcher=fetch)
    service.run(scanner.queue(app_id, fields=["english_requirements"]), fetcher=_pages(
        {PROGRAMME: _acquired(PROGRAMME, "IELTS 7.0 overall is required. " * 4)}))
    assert len(scanner.history(app_id)) == 2
    notes = scanner.notifications()
    assert len(notes) == 2
    assert all(note["message"].count("\n") == 0 for note in notes)
    assert "Example University scan completed" in notes[-1]["message"]


def test_today_lists_missing_details_for_a_direct_scan(tmp_path):
    path, _, app_id = _application(tmp_path)
    action = next(item for item in DailyPlanner(path).dashboard()["actions"] if item.get("scan_application_id") == app_id)
    assert "application details are still missing" in action["title"]
    assert action["page"] == "Applications"


def test_related_links_stay_on_the_official_host():
    links = official_links(
        '<a href="/admissions">Admissions</a><a href="https://evil.test/funding">Funding</a>',
        PROGRAMME, limit=5)
    assert links == ["https://example.edu/admissions"]
    parsed = parse_page(RICH, RICH_HTML, PROGRAMME)
    assert parsed["deadline"].value == "2026-12-03"
    assert parsed["application_fee"].value == "£75"
    assert parsed["referee_count"].value == "3"
    assert parsed["english_requirements"].value == "IELTS 7.0"
    assert parsed["supervisor_contact_policy"].value == "encouraged"
    assert parsed["portal_url"].value == "https://example.edu/apply"


def test_scan_does_not_require_gmail_or_change_profile_trust(tmp_path):
    source = Path(ApplicationRescan.__module__.replace(".", "/") + ".py")
    module_path = Path(__file__).resolve().parents[1] / "phd_agent" / "application_rescan.py"
    assert "gmail" not in module_path.read_text(encoding="utf-8").casefold()
    path = tmp_path / "phd_outreach.db"
    ProfileWorkspace(path).build("Aakash Example", "Reliable AI agents", [ProfileUpload("cv.txt", b"""Aakash Example
        MSc Computer Science, University of Liverpool.
        Evaluated reliable AI agents in a research project. I aim to study agent evaluation.
        """)])
    with connect(path) as db:
        before = [tuple(row) for row in db.execute("SELECT id,trust_level,status FROM applicant_research_contexts ORDER BY id")]
    ledger = Ledger(path)
    app_id = ledger.create_application("2027", programme_id=ledger.create_programme(
        "Example University", "PhD Computer Science", programme_url=PROGRAMME))
    fetch = _pages({PROGRAMME: _acquired(PROGRAMME, "The application fee is £75. " * 4)})
    OperationService(path).run(ApplicationRescan(path).queue(app_id, fields=["application_fee"]), fetcher=fetch)
    with connect(path) as db:
        after = [tuple(row) for row in db.execute("SELECT id,trust_level,status FROM applicant_research_contexts ORDER BY id")]
    assert before == after
    assert source or module_path


def test_applications_page_offers_scan_missing_details(tmp_path, monkeypatch):
    monkeypatch.setenv("PHD_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PHD_AGENT_ENV", "development")
    monkeypatch.setenv("PHD_AGENT_AUTH_DISABLED", "true")
    _application(tmp_path)
    ProfileWorkspace(tmp_path / "phd_outreach.db").build("Aakash Example", "Reliable AI agents", [ProfileUpload("cv.txt", b"""Aakash Example
        MSc Computer Science, University of Liverpool.
        Evaluated reliable AI agents in a research project. I aim to study agent evaluation.
        """)])
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=60).run()
    app = next(radio for radio in app.radio if radio.label == "Navigation").set_value("Applications").run()
    assert not app.exception
    assert any(button.label == "Scan missing details" for button in app.button)
    assert any(button.label == "Edit manually" for button in app.button)
