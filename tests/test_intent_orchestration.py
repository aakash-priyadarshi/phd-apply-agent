"""Applicant-context, intent orchestration, and supervised browser tests."""

import json
import shutil
import sqlite3
import socket
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phd_agent.applicant_context import ApplicantResearchContextService
from phd_agent.browser_worker import FormPlanService, fields_from_html, labels_match, unique_matching_index
from phd_agent.db import connect, migrate
from phd_agent.documents import DocumentVault
from phd_agent.materials import MaterialStudio
from phd_agent.model_router import ModelConfig, ModelRouter
from phd_agent.orchestration import (
    Acquisition, ProgrammeOrchestrator, _intent_filters_keeping_unknowns, _iso_deadline,
    _parse_human_date,
)
from phd_agent.official_search import OfficialSearchResult, OpenAIOfficialSearchProvider, SearchBatch, SearchHit
from phd_agent.portal import PortalAssistance
from phd_agent.profile import ApplicantTruth
from phd_agent.ui_agent import _companion_command


@pytest.fixture
def applicant(tmp_path):
    path = tmp_path / "agent.db"
    migrate(path)
    vault = DocumentVault(path)
    truth = ApplicantTruth(path, vault)
    studio = MaterialStudio(path, vault)
    profile_id = truth.create_profile("Aakash Example")
    source = vault.upload(b"Approved CV evidence", "master-source.txt", "SOURCE", "CV", "Master CV source")
    vault.set_approval(source["version_id"], True, "Reviewer")
    reliable = truth.create_claim(
        profile_id, "RESEARCH_PROJECT",
        "I evaluated long-horizon retrieval agents with soundness and reward-hacking checks.",
        "research_project", "FACT", source_document_version_id=source["version_id"],
    )
    education = truth.create_claim(
        profile_id, "EDUCATION", "I completed an MSc in Computer Science.",
        "degree", "FACT", source_document_version_id=source["version_id"],
    )
    aspiration = truth.create_claim(
        profile_id, "OTHER", "I aim to improve the reliability of agentic AI systems.",
        "research_goal", "ASPIRATION",
    )
    for claim in (reliable, education):
        truth.review_claim(claim, True, "Reviewer", verification_state="VERIFIED",
                           for_application=True, for_outreach=True)
    truth.review_claim(aspiration, True, "Reviewer", for_application=True, for_outreach=True)
    profile_version = truth.create_profile_version(
        profile_id, [reliable, education, aspiration], approve=True, reviewer="Reviewer")
    _, track_version = truth.create_track(
        profile_id, "Reliable agentic AI", research_problem="How can autonomous agents be evaluated reliably?",
        research_questions="Which evaluations reveal reward hacking and unsound behaviour?",
        proposed_methodology="Long-horizon controlled benchmarks and retrieval audits.",
        evaluation_strategy="Soundness, leakage, seed-control, and failure-mode checks.",
        expected_contribution="Reliable evaluation methods for agentic AI.",
        supporting_claim_revision_ids=[reliable],
    )
    truth.approve_track(track_version, "Reviewer")
    master = studio.create_master_cv(profile_version, [
        {"name": "Research Experience", "bullets": [{
            "text": "I evaluated long-horizon retrieval agents with soundness and reward-hacking checks.",
            "claim_revision_ids": [reliable],
        }]},
        {"name": "Education", "bullets": [{
            "text": "I completed an MSc in Computer Science.",
            "claim_revision_ids": [education],
        }]},
    ])
    studio.review_master_cv(master, "Reviewer", True)
    contexts = ApplicantResearchContextService(path)
    context = contexts.build(profile_version, master, track_version)
    return locals()


def test_context_is_cv_grounded_retrievable_and_immutable(applicant):
    context = applicant["context"]
    assert context["master_cv_version_id"] == applicant["master"]
    items = context["context"]["items"]
    assert any(item["cv_section"] == "Research Experience" for item in items)
    retrieval = applicant["contexts"].retrieve(
        context["id"], "PROFESSOR_EMAIL", "robust evaluation of autonomous retrieval agents",
        use="outreach", include_proposed=False,
    )
    assert retrieval.items[0].classification == "DEMONSTRATED"
    assert applicant["reliable"] in retrieval.claim_revision_ids
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with connect(applicant["path"]) as db:
            db.execute("UPDATE applicant_research_contexts SET model='changed' WHERE id=?", (context["id"],))
            db.commit()


def test_context_change_marks_linked_outputs_for_review_without_rewriting(applicant):
    contexts = applicant["contexts"]
    answer_ids = contexts.bootstrap_profile_answers(applicant["context"]["id"])
    assert answer_ids
    new_claim = applicant["truth"].create_claim(
        applicant["profile_id"], "SKILL_METHOD", "I use Python for AI evaluation tooling.",
        "skill", "FACT", source_document_version_id=applicant["source"]["version_id"],
    )
    applicant["truth"].review_claim(new_claim, True, "Reviewer", verification_state="VERIFIED",
                                    for_application=True, for_outreach=True)
    profile_v2 = applicant["truth"].create_profile_version(
        applicant["profile_id"], [applicant["reliable"], applicant["education"], applicant["aspiration"], new_claim],
        approve=True, reviewer="Reviewer",
    )
    master_v2 = applicant["studio"].create_master_cv(profile_v2, [{
        "name": "Research Experience", "bullets": [
            {"text": "I evaluated long-horizon retrieval agents with soundness and reward-hacking checks.",
             "claim_revision_ids": [applicant["reliable"]]},
            {"text": "I use Python for AI evaluation tooling.", "claim_revision_ids": [new_claim]},
        ],
    }])
    applicant["studio"].review_master_cv(master_v2, "Reviewer", True)
    new_context = contexts.build(profile_v2, master_v2, applicant["track_version"])
    pinned_context = contexts.build_current(
        profile_id=applicant["profile_id"], profile_version_id=applicant["profile_version"],
        track_version_id=applicant["track_version"])
    stale = contexts.stale_outputs()
    assert new_context["id"] != applicant["context"]["id"]
    assert pinned_context["profile_version_id"] == applicant["profile_version"]
    assert any(row["output_type"] == "PORTAL_ANSWER" and row["output_id"] in answer_ids for row in stale)
    with connect(applicant["path"]) as db:
        assert db.execute("SELECT COUNT(*) FROM answer_library WHERE id IN (%s)" %
                          ",".join("?" for _ in answer_ids), answer_ids).fetchone()[0] == len(answer_ids)


def test_intent_url_to_reviewed_application_and_workload_metrics(applicant):
    orchestrator = ProgrammeOrchestrator(applicant["path"])
    intent = orchestrator.create_intent(
        "Find funded 2027 PhD programmes in reliable AI, agent evaluation and RAG reliability in the UK.",
        applicant["context"]["id"],
    )
    html = """
      <html><head><title>PhD in Reliable AI | Example University</title></head>
      <body><h1>PhD in Reliable AI</h1>
      <p>Example University Department of Computer Science. September 2027 entry.</p>
      <p>Applications close 2027-01-15. This is a fully funded studentship.</p>
      <p>Entry requirements include a relevant masters degree and English language evidence.</p>
      <p>Required documents: CV, research proposal, transcript, and two academic references.</p>
      <p>Applicants are encouraged to contact a potential supervisor before applying.</p>
      </body></html>
    """
    result = orchestrator.analyse_supplied(
        "https://example.edu/phd/reliable-ai", html, applicant["context"]["id"],
        intent_id=intent["id"], method="UPLOADED_HTML", filename="programme.html",
    )
    candidate = result["candidate"]
    assert candidate["payload"]["programme"] == "PhD in Reliable AI"
    assert candidate["payload"]["deadline"] == "2027-01-15"
    assert candidate["payload"]["relevant_applicant_experience"]
    assert candidate["payload"]["research_fit"] >= 0
    assert candidate["payload"]["qs_match_state"] == "UNKNOWN"
    accepted = orchestrator.accept_candidate(candidate["id"], "Reviewer", cycle="2027")
    with connect(applicant["path"]) as db:
        app = db.execute("SELECT * FROM applications WHERE id=?", (accepted["application_id"],)).fetchone()
        opportunity = db.execute("SELECT * FROM opportunities WHERE id=?", (app["opportunity_id"],)).fetchone()
        deadline = db.execute("SELECT * FROM deadlines WHERE application_id=?", (app["id"],)).fetchone()
        evidence = db.execute("SELECT * FROM source_evidence WHERE id=?", (accepted["source_evidence_id"],)).fetchone()
        requirements = db.execute("SELECT normalized_document_type,requirement_state FROM requirements WHERE application_id=?",
                                  (accepted["application_id"],)).fetchall()
    assert app and evidence["verification_state"] == "VERIFIED"
    assert opportunity["contact_policy"] == "UNKNOWN"
    assert "Unverified supervisor-contact excerpt" in opportunity["notes"]
    assert deadline["verification_state"] == "NEEDS_REVIEW"
    assert orchestrator.accept_candidate(candidate["id"], "Reviewer") == accepted
    assert "fully funded studentship" in evidence["relevant_excerpt"]
    assert {row["normalized_document_type"] for row in requirements} >= {"CV", "RESEARCH_PROPOSAL", "TRANSCRIPT"}
    assert any(row["requirement_state"] == "UNKNOWN" for row in requirements)
    metrics = orchestrator.workload_summary()
    assert metrics["PAGES_INGESTED"] == 1
    assert metrics["FIELDS_EXTRACTED_AUTOMATICALLY"] > 0


def test_programme_ingestion_records_qs_rank_separately_from_research_fit(applicant):
    orchestrator = ProgrammeOrchestrator(applicant["path"])
    intent = orchestrator.create_intent(
        "Find funded 2027 PhD programmes in reliable AI in the US, preferably QS top 50",
        applicant["context"]["id"],
    )
    assert intent["filters"]["qs_max"] == 50
    assert "US" in intent["filters"]["country_codes"]
    html = """
      <html><head><title>PhD in Computer Science | Stanford University</title></head>
      <body><h1>PhD in Computer Science</h1>
      <p>Stanford University, United States. September 2027 entry.</p>
      <p>Applications close 2026-12-08. This is a fully funded studentship.</p>
      <p>Required documents: CV, statement of purpose, and three academic references.</p>
      </body></html>
    """
    result = orchestrator.analyse_supplied(
        "https://cs.stanford.edu/phd", html, applicant["context"]["id"], intent_id=intent["id"],
        method="UPLOADED_HTML", filename="stanford.html",
    )
    payload = result["candidate"]["payload"]
    assert payload["university"] == "Stanford University"
    assert payload["qs_rank_display"] == "=2"
    assert payload["qs_rank_numeric"] == 2
    assert payload["qs_match_state"] == "EXACT"
    assert payload["country"] == "United States"
    assert payload["country_code"] == "US"
    assert payload["country_match_state"] == "CONFIRMED"
    listed = orchestrator.list_candidates(
        intent_id=intent["id"], extra_filters={"qs_max": 25, "country_codes": ["US"]})
    assert listed and listed[0]["id"] == result["candidate"]["id"]
    assert orchestrator.list_candidates(intent_id=intent["id"], extra_filters={"qs_max": 1}) == []
    accepted = orchestrator.accept_candidate(result["candidate"]["id"], "Reviewer", cycle="2027")
    with connect(applicant["path"]) as db:
        programme = db.execute("SELECT * FROM programmes WHERE id=?",
                               (accepted["programme_id"],)).fetchone()
    assert programme["qs_rank_display"] == "=2"
    assert programme["qs_rank_numeric"] == 2
    assert programme["qs_source_evidence_id"]
    assert programme["country"] == "United States"


def test_intent_filters_keep_unknown_enrichment_and_extra_filters_stay_hard(applicant):
    orchestrator = ProgrammeOrchestrator(applicant["path"])
    intent = orchestrator.create_intent(
        "Find funded 2027 PhD programmes in the US, preferably QS top 25",
        applicant["context"]["id"],
    )
    html = """
      <html><head><title>PhD in Computing | Unknown College</title></head>
      <body><h1>PhD in Computing</h1>
      <p>Unknown College Department of Computing. September 2027 entry.</p>
      <p>Applicants should hold a masters degree.</p>
      </body></html>
    """
    result = orchestrator.analyse_supplied(
        "https://unknown.example/phd", html, applicant["context"]["id"], intent_id=intent["id"],
        method="UPLOADED_HTML", filename="unknown.html",
    )
    payload = result["candidate"]["payload"]
    assert not payload.get("country_code")
    assert payload["qs_match_state"] == "UNKNOWN"
    relaxed = _intent_filters_keeping_unknowns(payload, intent["filters"])
    assert "country_codes" not in relaxed
    assert "qs_max" not in relaxed
    assert "funded_only" not in relaxed
    listed = orchestrator.list_candidates(intent_id=intent["id"])
    assert any(item["id"] == result["candidate"]["id"] for item in listed)
    assert orchestrator.list_candidates(
        intent_id=intent["id"], extra_filters={"country_codes": ["US"]}) == []


def test_deadline_parsing_keeps_august_and_ignores_unrelated_dates():
    assert _parse_human_date("8th August 2026") == "2026-08-08"
    assert _parse_human_date("August 8th, 2026") == "2026-08-08"
    assert _iso_deadline("Deadline: 8th August 2026")[0] == "2026-08-08"
    assert _iso_deadline("The deadline is 2026-12-08.")[0] == "2026-12-08"
    assert _iso_deadline("Last updated 2026-01-02. Contact the department.")[0] is None


def test_failed_automation_requests_human_content_without_creating_candidate(applicant, monkeypatch):
    orchestrator = ProgrammeOrchestrator(applicant["path"])
    monkeypatch.setattr(orchestrator, "acquire_static", lambda url: Acquisition(
        "HUMAN_INPUT_REQUIRED", "STATIC_HTTP", url, reason="Access denied"))
    result = orchestrator.analyse_url("https://example.edu/blocked", applicant["context"]["id"])
    assert result["status"] == "HUMAN_INPUT_REQUIRED"
    assert result["actions"] == ["Paste page text", "Upload saved HTML/PDF"]
    with connect(applicant["path"]) as db:
        assert db.execute("SELECT COUNT(*) FROM programme_candidates").fetchone()[0] == 0


def test_static_acquisition_validates_redirect_before_second_request(applicant, monkeypatch):
    calls = []

    def addresses(host, port, type=None):
        address = "127.0.0.1" if host == "127.0.0.1" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    class Redirect:
        status_code = 302
        headers = {"location": "http://127.0.0.1/private"}
        url = "https://example.edu/start"

        def close(self):
            pass

    def fake_get(url, **kwargs):
        calls.append(url)
        return Redirect()

    monkeypatch.setattr("phd_agent.orchestration.socket.getaddrinfo", addresses)
    monkeypatch.setattr("phd_agent.orchestration.requests.get", fake_get)
    result = ProgrammeOrchestrator(applicant["path"]).acquire_static("https://example.edu/start")
    assert result.status == "HUMAN_INPUT_REQUIRED"
    assert result.browser_fallback_allowed is False
    assert calls == ["https://example.edu/start"]

    class BrowserMustNotRun:
        def acquire(self, url):
            raise AssertionError("Private redirect rejection must not fall back to Playwright")

    orchestrator = ProgrammeOrchestrator(applicant["path"])
    monkeypatch.setattr(orchestrator, "acquire_static", lambda url: result)
    fallback = orchestrator.analyse_url(
        "https://example.edu/start", applicant["context"]["id"], browser_worker=BrowserMustNotRun())
    assert fallback["status"] == "HUMAN_INPUT_REQUIRED"


def test_static_acquisition_returns_safe_fallback_for_invalid_url(applicant):
    result = ProgrammeOrchestrator(applicant["path"]).acquire_static("http://127.0.0.1/private")
    assert result.status == "HUMAN_INPUT_REQUIRED"
    assert result.url == "invalid://public-url"
    assert result.browser_fallback_allowed is False


def test_static_acquisition_classifies_access_status_before_raise(applicant, monkeypatch):
    class Blocked:
        status_code = 403
        headers = {}

        def close(self):
            self.closed = True

        def raise_for_status(self):
            raise AssertionError("Blocked statuses must be classified before raise_for_status")

    response = Blocked()
    monkeypatch.setattr("phd_agent.orchestration.socket.getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr("phd_agent.orchestration.requests.get", lambda *args, **kwargs: response)
    result = ProgrammeOrchestrator(applicant["path"]).acquire_static("https://example.edu/private")
    assert result.status == "HUMAN_INPUT_REQUIRED"
    assert result.browser_fallback_allowed is True
    assert response.closed is True


def test_browser_acquisition_rejects_non_public_final_destination(applicant, monkeypatch):
    orchestrator = ProgrammeOrchestrator(applicant["path"])
    monkeypatch.setattr(orchestrator, "acquire_static", lambda url: Acquisition(
        "HUMAN_INPUT_REQUIRED", "STATIC_HTTP", url, reason="JavaScript required"))

    class PrivateRedirectBrowser:
        def acquire(self, url):
            return type("Rendered", (), {
                "status": "ACQUIRED", "method": "PLAYWRIGHT", "url": "http://127.0.0.1/private",
                "text": "private response " * 100, "html": "", "human_action": None,
            })()

    result = orchestrator.analyse_url(
        "https://example.edu/start", applicant["context"]["id"], browser_worker=PrivateRedirectBrowser())
    assert result["status"] == "HUMAN_INPUT_REQUIRED"
    assert "public" in result["reason"].casefold()


def test_browser_fill_plan_reuses_only_approved_safe_values(applicant):
    orchestrator = ProgrammeOrchestrator(applicant["path"])
    programme = orchestrator.ledger.create_programme("Example University", "PhD AI", portal_url="https://apply.example.edu")
    application = orchestrator.ledger.create_application("2027", programme_id=programme,
                                                         portal_url="https://apply.example.edu/form")
    answer = PortalAssistance(applicant["path"]).save_answer("FULL_NAME", "Full legal name", "Aakash Example")
    PortalAssistance(applicant["path"]).approve_answer(answer, "Reviewer")
    fields = fields_from_html("""
      <form><label for='name'>Full legal name</label><input id='name' required>
      <label for='research'>Research experience</label><textarea id='research'></textarea>
      <label for='password'>Password</label><input id='password' type='password'></form>
    """)
    plan = FormPlanService(applicant["path"]).create_plan(
        application, "https://apply.example.edu/form", fields, context_id=applicant["context"]["id"])
    actions = {item["label"]: item["action"] for item in plan["items"]}
    assert actions == {"Full legal name": "SAFE_AUTOFILL", "Research experience": "GENERATE_AND_REVIEW",
                       "Password": "MANUAL_SENSITIVE"}
    assert plan["safe_count"] == plan["review_count"] == plan["manual_count"] == 1
    narrative = next(item for item in plan["items"] if item["label"] == "Research experience")
    assert "retrieval agents" in narrative["value"]
    assert narrative["source"]["answer_id"]
    with connect(applicant["path"]) as db:
        link = db.execute("SELECT * FROM output_context_links WHERE output_type='BROWSER_FILL_PLAN' AND output_id=?",
                          (plan["id"],)).fetchone()
    assert link and link["applicant_context_id"] == applicant["context"]["id"]


def test_model_router_uses_configured_family_and_only_escalates():
    router = ModelRouter(ModelConfig(provider="test", luna="luna", terra="terra", sol="sol"))
    assert router.route("PAGE_EXTRACTION").model == "luna"
    assert router.route("PAGE_EXTRACTION", confidence=0.5).model == "terra"
    assert router.route("PROGRAMME_TRIAGE", conflicting_evidence=True).model == "sol"
    assert router.route("TAILORED_SOP").model == "sol"
    with pytest.raises(ValueError, match="cannot reduce"):
        router.route("TAILORED_SOP", operator_tier="LUNA")


def test_browser_locators_prefer_label_name_and_tag_specific_nth():
    fields = fields_from_html("""
      <form>
        <label>Research experience<textarea name="research"></textarea></label>
        <select name="country" aria-label="Country"></select>
        <input aria-label="Full legal name">
      </form>
    """)
    locators = {field.label: field.locator for field in fields}
    assert locators["Research experience"] == "textarea[name='research']"
    assert locators["Country"] == "select[name='country']"
    assert locators["Full legal name"] == "get-by-label:Full legal name"
    assert not any("nth-of-type" in field.locator for field in fields)
    assert not any(field.locator.startswith("input:") for field in fields)
    placeholder = fields_from_html("<form><input placeholder='Email address'></form>")
    assert placeholder[0].label == "Email address"
    assert placeholder[0].locator == "input[placeholder='Email address']"
    assert labels_match("Full legal name", "Full legal name")
    assert not labels_match("Name", "Full legal name")
    assert unique_matching_index("Email address", [["Email address"], ["Email address"]]) is None
    assert unique_matching_index("Email address", [["Name"], ["Email address"]]) == 1


def test_browser_locators_use_tag_specific_nth_without_identity():
    fields = fields_from_html("<form><input><textarea></textarea><select></select></form>")
    assert [field.locator for field in fields] == [
        "xpath=(//input)[1]", "xpath=(//textarea)[1]", "xpath=(//select)[1]",
    ]


def test_browser_companion_command_matches_platform():
    assert _companion_command(12, "nt") == (
        ".\\.venv\\Scripts\\python.exe -m scripts.browser_companion 12", "powershell")
    assert _companion_command(12, "posix") == (
        "./.venv/bin/python -m scripts.browser_companion 12", "bash")


def test_official_search_filters_unofficial_and_non_https_results():
    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"output_parsed": SearchBatch(results=[
                SearchHit(title="Official PhD", institution="Example University",
                          official_url="https://example.edu/phd", source_kind="PROGRAMME",
                          relevance_reason="Relevant official programme", official_source=True),
                SearchHit(title="Directory", institution="Example University",
                          official_url="https://findaphd.com/example", source_kind="PROGRAMME",
                          relevance_reason="Directory", official_source=True),
                SearchHit(title="Insecure", institution="Example University",
                          official_url="http://example.edu/other", source_kind="PROGRAMME",
                          relevance_reason="No TLS", official_source=True),
            ])})()

    client = type("Client", (), {"responses": Responses()})()
    provider = OpenAIOfficialSearchProvider("", client=client)
    route = ModelRouter(ModelConfig(provider="test", luna="luna", terra="terra", sol="sol")).route("DISCOVERY_PLANNING")
    results = provider.search("reliable AI PhD", ("evaluated retrieval agents",), route)
    assert [result.official_url for result in results] == ["https://example.edu/phd"]
    assert captured["model"] == "terra"
    assert captured["tools"] == [{"type": "web_search"}]
    assert "evaluated retrieval agents" in captured["input"]


def test_faculty_web_discovery_queues_unverified_official_candidate(applicant, monkeypatch):
    orchestrator = ProgrammeOrchestrator(applicant["path"])
    programme = orchestrator.ledger.create_programme("Example University", "PhD Reliable AI")
    application = orchestrator.ledger.create_application("2027", programme_id=programme)

    class Provider:
        provider = "test-search"

        def search(self, intent, relevant_experience, route, *, purpose="PROGRAMME"):
            assert purpose == "FACULTY"
            assert any("retrieval agents" in item for item in relevant_experience)
            return [OfficialSearchResult(
                "Professor Ada profile", "Example University", "https://example.edu/faculty/ada",
                "FACULTY", "Works on reliable agent evaluation", "Professor Ada", "Computer Science")]

    monkeypatch.setattr(orchestrator, "acquire_static", lambda url: Acquisition(
        "ACQUIRED", "STATIC_HTTP", url,
        text=("Professor Ada is a current faculty member in Computer Science. "
              "Her research covers reliable evaluation of autonomous agents. ") * 6))
    result = orchestrator.discover_faculty_official_web(
        application, applicant["context"]["id"], provider=Provider())
    assert len(result["queued"]) == 1
    assert result["queued"][0]["name"] == "Professor Ada"
    with connect(applicant["path"]) as db:
        candidate = db.execute("SELECT * FROM faculty_candidates WHERE id=?",
                               (result["queued"][0]["candidate_id"],)).fetchone()
        evidence = db.execute("SELECT * FROM source_evidence WHERE id=?",
                              (candidate["source_evidence_id"],)).fetchone()
    assert candidate["review_state"] == "NEW"
    assert evidence["verification_state"] == "UNVERIFIED"


def test_migration_contains_intent_context_and_browser_tables(tmp_path):
    path = tmp_path / "schema.db"
    migrate(path)
    with connect(path) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        version = db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    assert version == 10
    assert {"applicant_research_contexts", "context_retrievals", "programme_candidates",
            "browser_fill_plans", "workload_events", "profile_build_operations",
            "university_rankings", "university_aliases"} <= tables


def test_intent_first_streamlit_home_renders_with_approved_context(applicant, monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    shutil.copy2(applicant["path"], runtime / "phd_outreach.db")
    monkeypatch.setenv("PHD_AGENT_DATA_DIR", str(runtime))
    monkeypatch.setenv("PHD_AGENT_ENV", "development")
    monkeypatch.setenv("PHD_AGENT_AUTH_DISABLED", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py", default_timeout=60).run()
    assert not app.exception
    assert app.toggle[0].label == "Advanced tools"
    assert any(area.label == "Navigation" and area.value == "Home" for area in app.radio)
    assert any("Welcome back" in markdown.value for markdown in app.markdown)
