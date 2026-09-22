"""Isolated Slice 4 review, Gmail-memory, and send-state tests."""

import hashlib
import json
import sqlite3
import base64
from email import message_from_bytes
from email.policy import default
from datetime import datetime, timezone

import pytest

from phd_agent.db import connect, transaction, utc_now
from phd_agent.discovery import Discovery
from phd_agent.faculty_research import FacultyResearch
from phd_agent.documents import DocumentVault
from phd_agent.gmail_gateway import GmailGateway
from phd_agent.ledger import Ledger
from phd_agent.materials import MaterialStudio
from phd_agent.outreach import CampaignPolicy, OutreachBlocked, OutreachService
from phd_agent.packages import PackageBuilder
from phd_agent.profile import ApplicantTruth


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
    programme = ledger.create_programme("Example University", "PhD Computer Science")
    opportunity = ledger.create_opportunity("FACULTY_ENQUIRY", "Faculty contact", "Example University",
        programme_id=programme, contact_policy="ALLOWED", verification_state="VERIFIED",
        source_evidence_id=faculty_evidence)
    app = ledger.create_application("2027", programme_id=programme, opportunity_id=opportunity)
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


def test_context_grounded_draft_quality_and_immutable_package(case):
    c = case
    outreach = c["outreach"]
    package = outreach.prepare(c["faculty"],c["app"],c["document_package"],c["profile_version"],c["track_version"])
    row = outreach.get_package(package)
    context, draft = outreach._unpack(row)
    assert context.claim_ids == (c["fact"],)
    assert set(draft.attachment_document_version_ids) == {c["cv_version"],c["proposal_version"]}
    assert draft.publication_ids == (c["publication"],)
    assert 120 <= len(draft.body.split()) <= 170
    assert json.loads(row["quality_json"])["status"] in {"PASS","WARNING"}
    with connect(c["path"]) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE outreach_packages SET snapshot_json='{}' WHERE id=?", (package,))
    outreach.approve(package,"Test reviewer")
    assert outreach.get_package(package)["status"] == "APPROVED"
    with pytest.raises(OutreachBlocked, match="DUPLICATE_ACTIVE_OUTREACH"):
        outreach.prepare(c["faculty"],c["app"],c["document_package"],c["profile_version"],c["track_version"])


def test_rejected_professor_cannot_be_prepared_for_outreach(case):
    c = case
    FacultyResearch(c["path"]).decide(c["app"], "REJECTED", faculty_id=c["faculty"])
    with pytest.raises(OutreachBlocked, match="PROFESSOR_REJECTED_FOR_APPLICATION"):
        c["outreach"].prepare(c["faculty"], c["app"], c["document_package"],
                              c["profile_version"], c["track_version"])


def test_quality_blocks_wrong_professor_institution_claims_and_attachment_mentions(case):
    c = case
    context = c["outreach"].build_context(c["faculty"],c["app"],c["document_package"],c["profile_version"],c["track_version"])
    draft = c["outreach"].draft_text(context)
    from dataclasses import replace
    wrong = replace(draft, body=draft.body.replace("Dr Ada One", "Dr Wrong")
        .replace("Example University", "Wrong University") + "\n\nPhD AI. Your group is accepting new students. Funding is available. I attached a transcript.")
    c["ledger"].create_programme("Wrong University", "PhD AI")
    with transaction(c["path"]) as db:
        db.execute("INSERT INTO faculty_profiles(name,institution,created_at,updated_at) VALUES('Dr Wrong','Wrong University',?,?)",(utc_now(),utc_now()))
    gate = c["outreach"].quality_gate(context,wrong)
    blocks = {r["rule_id"] for r in gate["rules"] if r["status"] == "BLOCK"}
    assert {"WRONG_PROFESSOR","WRONG_INSTITUTION","WRONG_PROGRAMME","SUPERVISION_CLAIM","FUNDING_CLAIM","ATTACHMENT_MENTIONS"} <= blocks
    wrong_ids = replace(draft,professor_evidence_ids=(999,),publication_ids=(999,))
    blocked = {r["rule_id"] for r in c["outreach"].quality_gate(context,wrong_ids)["rules"] if r["status"]=="BLOCK"}
    assert {"PROFESSOR_EVIDENCE","PUBLICATIONS"} <= blocked


def test_sent_reconciliation_idempotency_ambiguous_and_reply(case):
    c = case
    outreach = c["outreach"]
    package = outreach.prepare(c["faculty"],c["app"],c["document_package"],c["profile_version"],c["track_version"])
    outreach.approve(package,"Test reviewer")
    outreach.reconcile_sent([],source="TEST")
    class Uncertain:
        def send(self,*args):
            raise TimeoutError("network outcome unknown")
    first = outreach.send(package,Uncertain(),"Test reviewer")
    assert first["status"] == "AMBIGUOUS_SEND" and not first["retry_allowed"]
    with pytest.raises(OutreachBlocked):
        outreach.send(package,Uncertain(),"Test reviewer")
    assert outreach.recover_ambiguous(first["message_id"],"Test reviewer")["status"] == "AMBIGUOUS_SEND"
    context,draft = outreach._unpack(outreach.get_package(package))
    outreach.reconcile_sent([{"recipient":context.recipient,"subject":draft.subject,
        "message_at":utc_now(),"message_id":"gmail-1","thread_id":"thread-1",
        "package_identity":outreach.get_package(package)["snapshot_sha256"]}],source="TEST")
    assert outreach.recover_ambiguous(first["message_id"],"Test reviewer")["status"] == "SENT"
    reply = outreach.record_reply({"message_id":"reply-1","thread_id":"thread-1",
        "subject":"Re: " + draft.subject,"message_at":utc_now()})
    assert reply > 0 and outreach.get_package(package)["status"] == "REPLIED"


def test_campaign_dry_run_emergency_stop_and_contact_memory(case):
    c = case
    outreach = c["outreach"]
    policy = CampaignPolicy(included_application_ids=(c["app"],),timezone_name="UTC",
                            allowed_weekdays=(0,1,2,3,4,5,6),local_start_hour=0,local_end_hour=24)
    campaign = outreach.create_campaign("Demo", "2027", policy)
    with connect(c["path"]) as db:
        assert db.execute("SELECT auto_send_enabled FROM campaigns WHERE id=?",(campaign,)).fetchone()[0] == 0
    generated = outreach.dry_run(campaign,c["profile_version"],c["track_version"],
        now=datetime(2026,9,21,12,tzinfo=timezone.utc))
    assert generated["candidates"][0]["outcome"] == "WOULD_GENERATE"
    package = outreach.prepare(c["faculty"],c["app"],c["document_package"],c["profile_version"],c["track_version"],campaign_id=campaign)
    outreach.approve(package,"Test reviewer")
    ready = outreach.dry_run(campaign,c["profile_version"],c["track_version"],
        now=datetime(2026,9,21,12,tzinfo=timezone.utc))
    assert ready["candidates"][0]["outcome"] == "WOULD_SEND"
    outreach.set_campaign_controls(campaign,"Test reviewer",emergency_stop=True)
    stopped = outreach.dry_run(campaign,c["profile_version"],c["track_version"],
        now=datetime(2026,9,21,12,tzinfo=timezone.utc))
    assert stopped["candidates"][0]["outcome"] == "BLOCKED"
    assert "EMERGENCY_STOP" in stopped["candidates"][0]["reasons"]
    outreach.record_manual_contact(c["faculty"],"ada.one@example.edu","Historical enquiry",utc_now(),"Test reviewer")
    with pytest.raises(OutreachBlocked, match="PREVIOUS_GMAIL_CONTACT"):
        outreach.build_context(c["faculty"],c["app"],c["document_package"],c["profile_version"],c["track_version"],
            campaign_id=campaign,allow_package_id=package)


def test_successful_mock_send_uses_exact_versions_and_blocks_reuse(case):
    c = case
    service = c["outreach"]
    package = service.prepare(c["faculty"],c["app"],c["document_package"],c["profile_version"],c["track_version"])
    service.approve(package,"Test reviewer")
    service.reconcile_sent([],source="TEST")
    class Capture:
        def __init__(self): self.calls=[]
        def send(self,*args):
            self.calls.append(args)
            return {"message_id":"gmail-sent-1","thread_id":"thread-sent-1"}
    transport=Capture()
    result=service.send(package,transport,"Test reviewer")
    assert result["status"]=="SENT" and len(transport.calls)==1
    context,draft=service._unpack(service.get_package(package))
    recipient,subject,body,attachments,identity=transport.calls[0]
    assert recipient==context.recipient and subject==draft.subject and body==draft.body
    assert identity==service.get_package(package)["snapshot_sha256"]
    assert [filename for filename,_ in attachments]==[a.filename for a in context.attachments]
    assert [hashlib.sha256(data).hexdigest() for _,data in attachments]==[a.sha256 for a in context.attachments]
    with pytest.raises(OutreachBlocked):
        service.send(package,transport,"Test reviewer")
    with connect(c["path"]) as db:
        assert db.execute("SELECT COUNT(*) FROM outreach_messages").fetchone()[0]==1
        assert db.execute("SELECT COUNT(*) FROM outreach_attempts").fetchone()[0]==1


def test_hash_mismatch_stale_evidence_and_missing_applicant_review(case):
    c=case
    service=c["outreach"]
    with pytest.raises(OutreachBlocked,match="APPLICANT_REVIEW"):
        service.build_context(c["faculty"],c["app"],c["document_package"],999,c["track_version"])
    with pytest.raises(OutreachBlocked,match="APPLICANT_REVIEW"):
        service.build_context(c["faculty"],c["app"],c["document_package"],c["profile_version"],999)
    package=service.prepare(c["faculty"],c["app"],c["document_package"],c["profile_version"],c["track_version"])
    service.approve(package,"Test reviewer")
    service.reconcile_sent([],source="TEST")
    version=c["vault"].get_version(c["cv_version"])
    storage_path=c["vault"].storage._path(version["storage_key"])
    storage_path.write_bytes(b"tampered")
    with pytest.raises(OutreachBlocked):
        service.send(package,object(),"Test reviewer")
    assert service.refresh_staleness(package)
    assert service.get_package(package)["stale_at"]


def test_stale_faculty_evidence_blocks_generation(case,monkeypatch):
    c=case
    monkeypatch.setattr("phd_agent.outreach.freshness",lambda *_:"STALE")
    with pytest.raises(OutreachBlocked,match="STALE_AFFILIATION_EVIDENCE"):
        c["outreach"].build_context(c["faculty"],c["app"],c["document_package"],
                                    c["profile_version"],c["track_version"])


def test_campaign_window_cap_and_academic_verification(tmp_path,case):
    c=case
    service=c["outreach"]
    policy=CampaignPolicy(timezone_name="UTC",allowed_weekdays=(0,),local_start_hour=9,
        local_end_hour=17,daily_cap=1,minimum_spacing_minutes=90)
    campaign=service.create_campaign("Window","2027",policy)
    assert "OUTSIDE_SEND_WINDOW" in service.campaign_send_reasons(campaign,now=datetime(2026,9,21,20,tzinfo=timezone.utc))
    package=service.prepare(c["faculty"],c["app"],c["document_package"],c["profile_version"],c["track_version"],campaign_id=campaign)
    with transaction(c["path"]) as db:
        db.execute("""INSERT INTO outreach_messages(outreach_key,package_id,recipient,subject,status,sent_at,created_at,updated_at)
            VALUES(?,?,?,'Other','SENT',?,?,?)""",
            (service.key(c["faculty"],campaign_id=campaign),package,"ada.one@example.edu",
             "2026-09-21T10:00:00+00:00",utc_now(),utc_now()))
    reasons=service.campaign_send_reasons(campaign,now=datetime(2026,9,21,10,30,tzinfo=timezone.utc))
    assert {"DAILY_CAP_REACHED","MINIMUM_SPACING"} <= set(reasons)
    # Source academic versions require a separate authenticity/type review before approval.
    vault=DocumentVault(tmp_path/"academic.db")
    source=vault.upload(b"%PDF-1.4 synthetic academic test bytes","transcript.pdf","SOURCE","TRANSCRIPT",
                        "Academic test",sensitivity="CONFIDENTIAL")
    with transaction(vault.db_path) as db:
        db.execute("UPDATE document_versions SET verification_state='NEEDS_REVIEW' WHERE id=?",(source["version_id"],))
    with pytest.raises(ValueError,match="authenticity"):
        vault.set_approval(source["version_id"],True,"Test reviewer")
    vault.set_verification(source["version_id"],"VERIFIED","Test reviewer")
    vault.set_approval(source["version_id"],True,"Test reviewer")
    assert vault.get_version(source["version_id"])["approval_state"]=="APPROVED"


def test_gmail_gateway_stays_unavailable_without_rotated_credentials(tmp_path):
    gateway=GmailGateway(data_dir=tmp_path)
    assert not gateway.credentials_ready()
    with pytest.raises(Exception,match="Rotated OAuth"):
        gateway.list_sent()


def test_mock_gmail_gateway_metadata_and_exact_mime_attachment():
    class Request:
        def __init__(self,value): self.value=value
        def execute(self): return self.value
    class FakeService:
        def __init__(self): self.raw=None
        def users(self): return self
        def messages(self): return self
        def list(self,**kwargs):
            self.last_q=kwargs.get("q")
            return Request({"messages":[{"id":"m1"}]})
        def get(self,**_): return Request({"id":"m1","threadId":"t1","internalDate":"1789980000000",
            "payload":{"headers":[{"name":"To","value":"Dr One <one@example.edu>"},
                                  {"name":"Subject","value":"Research enquiry"}]}})
        def send(self,*,body,**_):
            self.raw=base64.urlsafe_b64decode(body["raw"])
            return Request({"id":"new1","threadId":"new-thread"})
    fake=FakeService()
    gateway=GmailGateway(service=fake)
    records=gateway.list_sent()
    assert fake.last_q=="in:sent"
    later=gateway.list_sent(after=datetime(2026,9,21,tzinfo=timezone.utc))
    assert "after:" in fake.last_q
    assert records[0]["recipient"]=="Dr One <one@example.edu>"
    assert records[0]["message_id"]=="m1" and records[0]["thread_id"]=="t1"
    result=gateway.send("one@example.edu","Research enquiry","Reviewed body",[("cv.pdf",b"exact bytes")],"snapshot-hash")
    assert result=={"message_id":"new1","thread_id":"new-thread"}
    message=message_from_bytes(fake.raw,policy=default)
    assert message["X-PhDAgent-Package"]=="snapshot-hash"
    assert [part.get_payload(decode=True) for part in message.iter_attachments()]==[b"exact bytes"]


def test_live_gateway_rechecks_sent_history_immediately_before_send(case):
    c=case
    outreach=c["outreach"]
    package=outreach.prepare(c["faculty"],c["app"],c["document_package"],
                             c["profile_version"],c["track_version"])
    outreach.approve(package,"Test reviewer")
    outreach.reconcile_sent([],source="TEST")
    class Request:
        def __init__(self,value): self.value=value
        def execute(self): return self.value
    class ExternalContact:
        def users(self): return self
        def messages(self): return self
        def list(self,**_): return Request({"messages":[{"id":"external-1"}]})
        def get(self,**_): return Request({"id":"external-1","threadId":"external-thread",
            "internalDate":"1789980000000","payload":{"headers":[
                {"name":"To","value":"ada.one@example.edu"},
                {"name":"Subject","value":"Earlier enquiry"}]}})
        def send(self,**_): raise AssertionError("Gmail send must not be called")
    with pytest.raises(OutreachBlocked,match="PREVIOUS_GMAIL_CONTACT"):
        outreach.send(package,GmailGateway(service=ExternalContact()),"Test reviewer")
    with connect(c["path"]) as db:
        assert db.execute("SELECT COUNT(*) FROM outreach_messages").fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM gmail_threads WHERE gmail_message_id='external-1'").fetchone()[0]==1


def test_unresolved_sent_recipient_is_stored_without_aborting(case):
    outreach = case["outreach"]
    result = outreach.reconcile_sent([
        {"recipient": "undisclosed-recipients:;", "subject": "Hidden", "message_at": utc_now(),
         "message_id": "gid-unresolved", "thread_id": "tid-unresolved"},
        {"recipient": "ada.one@example.edu", "subject": "Known", "message_at": utc_now(),
         "message_id": "gid-known", "thread_id": "tid-known"},
    ], source="TEST")
    assert result["scanned"] == 2 and result["exact_email_candidates"] == 1
    with connect(case["path"]) as db:
        unresolved = db.execute("SELECT * FROM gmail_threads WHERE gmail_message_id='gid-unresolved'").fetchone()
        known = db.execute("SELECT * FROM gmail_threads WHERE gmail_message_id='gid-known'").fetchone()
    assert unresolved["faculty_profile_id"] is None
    assert unresolved["match_state"] == "PENDING" and unresolved["match_confidence"] == "UNKNOWN"
    assert known["faculty_profile_id"] == case["faculty"]
