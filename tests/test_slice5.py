"""Slice 5 reply follow-through, portal assistance, archive, and backup tests."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from phd_agent.backup import BackupService
from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.discovery import Discovery
from phd_agent.documents import DocumentVault
from phd_agent.ledger import Ledger
from phd_agent.materials import MaterialStudio, render_pdf
from phd_agent.outreach import OutreachBlocked, OutreachService
from phd_agent.packages import PackageBuilder
from phd_agent.portal import PortalAssistance
from phd_agent.profile import ApplicantTruth
from phd_agent.replies import ReplyIntelligence


@pytest.fixture
def case(tmp_path):
    path = tmp_path / "case.db"
    ledger, vault = Ledger(path), DocumentVault(path)
    truth, discovery = ApplicantTruth(path, vault), Discovery(path)
    studio, builder, outreach = MaterialStudio(path, vault), PackageBuilder(path, vault), OutreachService(path, vault)
    profile = truth.create_profile("Demo Applicant")
    source = vault.upload(b"Reviewed synthetic applicant source", "source.txt", "SOURCE", "CV", "Source CV")
    vault.set_approval(source["version_id"], True, "Test reviewer")
    fact = truth.create_claim(profile, "RESEARCH_PROJECT", "I evaluated retrieval agents with controlled benchmarks.",
        "project", "FACT", source_document_version_id=source["version_id"])
    truth.review_claim(fact, True, "Test reviewer", verification_state="VERIFIED", for_application=True, for_outreach=True)
    aim = truth.create_claim(profile, "OTHER", "I aim to improve agent reliability.", "goal", "ASPIRATION")
    truth.review_claim(aim, True, "Test reviewer", for_application=True, for_outreach=True)
    profile_version = truth.create_profile_version(profile, [fact, aim], approve=True, reviewer="Test reviewer")
    track, track_version = truth.create_track(profile, "Reliable retrieval agents",
        research_problem="How can retrieval agents be evaluated reliably?",
        motivation="I aim to improve agent reliability.",
        proposed_methodology="Controlled retrieval benchmarks.",
        evaluation_strategy="Compare reliability across benchmarks.",
        expected_contribution="Better reliability evaluation.", supporting_claim_revision_ids=[fact])
    truth.approve_track(track_version, "Test reviewer")
    faculty_evidence = ledger.create_evidence("https://example.edu/faculty/one", "FACULTY",
        "Dr Ada One studies retrieval agent evaluation and allows contact.", "VERIFIED")
    paper_evidence = ledger.create_evidence("https://example.edu/papers/retrieval", "PUBLICATION",
        "Evaluation of Retrieval Agents", "VERIFIED")
    requirement_evidence = ledger.create_evidence("https://example.edu/apply/contact", "PROGRAMME",
        "Faculty contact requires a CV and proposal.", "VERIFIED")
    faculty = discovery.create_faculty("Dr Ada One", "Example University", evidence_id=faculty_evidence)
    discovery.verify_faculty(faculty, "VERIFIED", evidence_by_fact={"IDENTITY":faculty_evidence,
        "AFFILIATION":faculty_evidence,"EMAIL":faculty_evidence,"TOPICS":faculty_evidence},
        reviewer="Test reviewer", reason="Official faculty page reviewed",
        updates={"email":"ada.one@example.edu", "email_state":"VERIFIED",
                 "research_topics":"retrieval agent evaluation", "affiliation_state":"CURRENT"})
    with transaction(path) as db:
        publication = db.execute("""INSERT INTO publications
            (faculty_profile_id,title,year,authors_json,topics_json,source_evidence_id,retrieved_at)
            VALUES(?,?,?,?,?,?,?)""", (faculty,"Evaluation of Retrieval Agents",datetime.now().year,
            '["Dr Ada One"]','["retrieval", "agents"]',paper_evidence,utc_now())).lastrowid
    programme = ledger.create_programme("Example University", "PhD Computer Science",
                                        portal_url="https://example.edu/apply")
    opportunity = ledger.create_opportunity("FACULTY_ENQUIRY", "Faculty contact", "Example University",
        programme_id=programme, contact_policy="ALLOWED", verification_state="VERIFIED",
        source_evidence_id=faculty_evidence, opening_status="OPEN")
    app = ledger.create_application("2027", programme_id=programme, opportunity_id=opportunity,
                                    eligibility_state="ELIGIBLE")
    discovery.link_to_application(app, faculty_id=faculty, evidence_id=faculty_evidence)
    cv_req = ledger.create_requirement(app,"FACULTY_OUTREACH","REQUIRED","CV",requirement_evidence,
                                       normalized_document_type="CV",file_format="pdf")
    proposal_req = ledger.create_requirement(app,"FACULTY_OUTREACH","REQUIRED","Research proposal",requirement_evidence,
                                             normalized_document_type="RESEARCH_PROPOSAL",file_format="pdf")
    master = studio.create_master_cv(profile_version,[{"name":"Research Experience","bullets":[
        {"text":"I evaluated retrieval agents with controlled benchmarks.","claim_revision_ids":[fact]}]}])
    studio.review_master_cv(master,"Test reviewer",True)
    cv = studio.tailor_cv(master,track_version,application_id=app,faculty_id=faculty)
    studio.review_artifact(cv,"Test reviewer",True)
    proposal = studio.create_proposal(profile_version,track_version,application_id=app,faculty_id=faculty,
        requirement_id=proposal_req,publication_ids=[publication])
    studio.review_artifact(proposal,"Test reviewer",True)
    with connect(path) as db:
        cv_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?",(cv,)).fetchone()[0]
        proposal_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?",(proposal,)).fetchone()[0]
    vault.link_to_application(app,cv_req,cv_version)
    vault.link_to_application(app,proposal_req,proposal_version)
    document_package = builder.build(app,profile_version,track_version,context="FACULTY_OUTREACH")
    builder.mark_ready(document_package,"Test reviewer")
    return locals()


def _send(case):
    outreach = case["outreach"]
    package = outreach.prepare(case["faculty"], case["app"], case["document_package"],
                               case["profile_version"], case["track_version"])
    outreach.approve(package, "Test reviewer")
    outreach.reconcile_sent([], source="TEST")
    class Transport:
        def send(self, *args):
            return {"message_id": "gmail-sent-1", "thread_id": "thread-sent-1"}
    result = outreach.send(package, Transport(), "Test reviewer")
    return package, result


def test_heuristics_prefer_explicit_signals():
    intel = ReplyIntelligence
    assert intel.suggest_category("Delivery Status Notification (Bounce)") == "BOUNCE"
    assert intel.suggest_category("Out of office") == "OUT_OF_OFFICE"
    assert intel.suggest_category("Please send a research proposal") == "PROPOSAL_REQUESTED"
    assert intel.suggest_category("Thanks") == "OTHER"


def test_proposal_request_creates_task_and_draft_document_without_sending(case):
    _send(case)
    outreach = case["outreach"]
    reply = outreach.record_reply({"message_id": "reply-1", "thread_id": "thread-sent-1",
                                   "subject": "Re: please send a research proposal", "message_at": utc_now()})
    intel = ReplyIntelligence(case["path"])
    result = intel.classify(reply, "PROPOSAL_REQUESTED", "Test reviewer", requested_length="one page")
    assert result["task_id"]
    artifact = intel.generate_requested_document(result["id"], profile_version_id=case["profile_version"],
                                                 track_version_id=case["track_version"])
    with connect(case["path"]) as db:
        assert db.execute("SELECT COUNT(*) FROM outreach_messages").fetchone()[0] == 1
        task = db.execute("SELECT * FROM application_tasks WHERE id=?", (result["task_id"],)).fetchone()
        generated = db.execute("SELECT * FROM generated_artifacts WHERE id=?", (artifact,)).fetchone()
    assert task["task_type"] == "DOCUMENT" and task["status"] == "TODO"
    assert generated["approval_state"] == "DRAFT" and generated["kind"] == "RESEARCH_PROPOSAL"


def test_bounce_marks_sent_package_without_contacting_gmail(case):
    package, _ = _send(case)
    intel = ReplyIntelligence(case["path"])
    reply = case["outreach"].record_reply({"message_id": "bounce-1", "thread_id": "thread-sent-1",
                                           "subject": "Mail Delivery Subsystem: bounce", "message_at": utc_now()})
    intel.classify(reply, "BOUNCE", "Test reviewer")
    assert case["outreach"].get_package(package)["status"] == "BOUNCED"


def test_follow_up_draft_after_delay_stays_unsent(case):
    package, _ = _send(case)
    intel = ReplyIntelligence(case["path"])
    too_soon = intel.mark_follow_ups_due(now=datetime.now(timezone.utc) + timedelta(days=1))
    assert too_soon == []
    due = intel.mark_follow_ups_due(now=datetime.now(timezone.utc) + timedelta(days=20))
    assert package in due
    assert case["outreach"].get_package(package)["status"] == "FOLLOW_UP_DUE"
    follow = intel.draft_follow_up(package)
    row = case["outreach"].get_package(follow)
    assert row["stage"] == "FOLLOW_UP"
    assert row["status"] in {"DRAFT", "NEEDS_REVIEW"}
    with connect(case["path"]) as db:
        assert db.execute("SELECT COUNT(*) FROM outreach_messages").fetchone()[0] == 1


def test_previous_contact_still_blocks_a_new_initial_email(case):
    _send(case)
    with pytest.raises(OutreachBlocked, match="PREVIOUS_GMAIL_CONTACT"):
        case["outreach"].prepare(case["faculty"], case["app"], case["document_package"],
                                 case["profile_version"], case["track_version"])


def _ready_formal(case):
    s = case
    evidence = s["requirement_evidence"]
    cv_req = s["ledger"].create_requirement(s["app"], "FORMAL_APPLICATION", "REQUIRED", "CV", evidence,
                                            normalized_document_type="CV", file_format="pdf")
    sop_req = s["ledger"].create_requirement(s["app"], "FORMAL_APPLICATION", "REQUIRED", "SOP", evidence,
                                             normalized_document_type="SOP", word_limit=250, file_format="pdf")
    transcript_req = s["ledger"].create_requirement(s["app"], "FORMAL_APPLICATION", "REQUIRED", "Transcript",
                                                    evidence, normalized_document_type="TRANSCRIPT", file_format="pdf")
    s["ledger"].create_deadline(s["app"], "APPLICATION",
                                (datetime.now(timezone.utc) + timedelta(days=90)).date().isoformat(),
                                evidence, verification_state="VERIFIED")
    module = s["studio"].create_module(s["profile_version"], "WHY_PHD",
        "I evaluated retrieval agents with controlled benchmarks.", [s["fact"]])
    s["studio"].review_module(module, "Test reviewer", True)
    sop = s["studio"].create_statement("SOP", s["profile_version"], s["track_version"], s["app"], sop_req, [module])
    s["studio"].review_artifact(sop, "Test reviewer", True)
    transcript = s["vault"].upload(render_pdf("Transcript", "Completed degree"), "transcript.pdf",
                                   "SOURCE", "TRANSCRIPT", "Transcript")
    s["vault"].set_approval(transcript["version_id"], True, "Test reviewer")
    with connect(s["path"]) as db:
        sop_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?", (sop,)).fetchone()[0]
    s["vault"].link_to_application(s["app"], cv_req, s["cv_version"])
    s["vault"].link_to_application(s["app"], sop_req, sop_version)
    s["vault"].link_to_application(s["app"], transcript_req, transcript["version_id"])
    package_id = s["builder"].build(s["app"], s["profile_version"], s["track_version"])
    s["builder"].mark_ready(package_id, "Test reviewer")
    return package_id


def test_portal_answers_checklist_and_frozen_submission_archive(case):
    portal = PortalAssistance(case["path"], case["vault"])
    answer = portal.save_answer("EDUCATION", "Highest degree", "MSc Computer Science, University of Liverpool",
                                claim_revision_id=case["fact"])
    portal.approve_answer(answer, "Test reviewer")
    field = portal.add_checklist_field(case["app"], "EDUCATION", "Highest qualification")
    portal.fill_field(field, answer)
    portal.review_field(field, "Test reviewer")
    ready = portal.copy_ready(case["app"])
    assert ready[0]["value"].startswith("MSc")
    assert ready[0]["sources"][0]["id"] == case["fact"]
    package_id = _ready_formal(case)
    with pytest.raises(ValueError, match="confirmation"):
        portal.record_submission(case["app"], package_id, " ", utc_now(), "Test reviewer")
    archive_id = portal.record_submission(
        case["app"], package_id, "CONF-2027-1", utc_now(), "Test reviewer",
        payment_state="PENDING_USER", notes="Operator submitted the portal form manually")
    archive = portal.get_archive(archive_id)
    assert archive["confirmation_number"] == "CONF-2027-1"
    assert archive["payment_state"] == "PENDING_USER"
    assert json.loads(archive["answers_json"])[0]["value"].startswith("MSc")
    with connect(case["path"]) as db:
        assert db.execute("SELECT status FROM applications WHERE id=?", (case["app"],)).fetchone()[0] == "SUBMITTED"
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE submission_archives SET confirmation_number='tampered' WHERE id=?", (archive_id,))


def test_backup_excludes_credentials_and_restores_vault_bytes(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    db = data / "phd_outreach.db"
    vault = DocumentVault(db)
    uploaded = vault.upload(b"vault-bytes", "note.txt", "SOURCE", "OTHER", "Note")
    (data / "credentials.json").write_text('{"client":"secret"}', encoding="utf-8")
    (data / "gmail_token.json").write_text('{"token":"secret"}', encoding="utf-8")
    service = BackupService(db, data)
    archive = tmp_path / "backup-1"
    manifest = service.create_backup(archive, "Tester")
    names = {item["path"] for item in manifest["files"]}
    assert "credentials.json" not in names and "gmail_token.json" not in names
    assert not (archive / "credentials.json").exists()
    dest = tmp_path / "restored"
    restored = service.restore(archive, dest, "Tester")
    assert "credentials.json" not in restored["files"]
    restored_vault = DocumentVault(dest / "phd_outreach.db")
    assert restored_vault.storage.get(uploaded["storage_key"]) == b"vault-bytes"
    with pytest.raises(FileExistsError):
        service.restore(archive, dest, "Tester")
    service.restore(archive, dest, "Tester", replace_existing=True)


def test_slice5_migration_is_applied(tmp_path):
    path = tmp_path / "empty.db"
    applied = migrate(path)
    assert 7 in applied
    with connect(path) as db:
        names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"reply_classifications", "answer_library", "portal_checklist_fields",
            "submission_archives", "backup_runs"} <= names
