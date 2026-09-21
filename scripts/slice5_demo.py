"""Reproducible Slice 5 demonstration; no Gmail, portal submit, or production writes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from phd_agent.backup import BackupService
from phd_agent.db import utc_now
from phd_agent.discovery import Discovery
from phd_agent.documents import DocumentVault
from phd_agent.ledger import Ledger
from phd_agent.materials import MaterialStudio
from phd_agent.outreach import OutreachService
from phd_agent.packages import PackageBuilder
from phd_agent.portal import PortalAssistance
from phd_agent.profile import ApplicantTruth
from phd_agent.replies import ReplyIntelligence


def run(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Choose a new isolated demo directory")
    output_dir.mkdir(parents=True)
    db_path = output_dir / "demo.db"
    ledger, vault = Ledger(db_path), DocumentVault(db_path)
    truth, discovery = ApplicantTruth(db_path, vault), Discovery(db_path)
    studio, builder = MaterialStudio(db_path, vault), PackageBuilder(db_path, vault)
    outreach, intel = OutreachService(db_path, vault), ReplyIntelligence(db_path)
    portal = PortalAssistance(db_path, vault)
    profile = truth.create_profile("Demo Applicant")
    source = vault.upload(b"Synthetic reviewed source", "source.txt", "SOURCE", "CV", "Demo source")
    vault.set_approval(source["version_id"], True, "Demo reviewer")
    claim = truth.create_claim(profile, "RESEARCH_PROJECT",
        "I evaluated retrieval agents with controlled benchmarks.", "project", "FACT",
        source_document_version_id=source["version_id"])
    truth.review_claim(claim, True, "Demo reviewer", verification_state="VERIFIED",
                       for_application=True, for_outreach=True)
    profile_version = truth.create_profile_version(profile, [claim], approve=True, reviewer="Demo reviewer")
    track, track_version = truth.create_track(profile, "Reliable retrieval agents",
        research_problem="How can retrieval agents be evaluated reliably?",
        motivation="Agent evaluation needs controlled evidence.",
        proposed_methodology="Compare agents with reproducible retrieval benchmarks.",
        evaluation_strategy="Measure failure modes and reliability.",
        expected_contribution="Clearer evaluation of retrieval agents.",
        supporting_claim_revision_ids=[claim])
    truth.approve_track(track_version, "Demo reviewer")
    faculty_source = ledger.create_evidence("https://example.edu/faculty/demo", "FACULTY",
        "Synthetic official-style record: retrieval agent evaluation, contact permitted.", "VERIFIED")
    paper_source = ledger.create_evidence("https://example.edu/paper/demo", "PUBLICATION",
        "Evaluation of Retrieval Agents", "VERIFIED")
    requirement_source = ledger.create_evidence("https://example.edu/contact/demo", "PROGRAMME",
        "Faculty contact requires a CV and proposal.", "VERIFIED")
    professor = discovery.create_faculty("Dr Ada One", "Example University", evidence_id=faculty_source)
    discovery.verify_faculty(professor, "VERIFIED", evidence_by_fact={"IDENTITY": faculty_source,
        "AFFILIATION": faculty_source, "EMAIL": faculty_source, "TOPICS": faculty_source},
        reviewer="Demo reviewer", reason="Synthetic verified demo source",
        updates={"email": "ada.one@example.edu", "email_state": "VERIFIED",
                 "research_topics": "retrieval agent evaluation", "affiliation_state": "CURRENT"})
    from phd_agent.db import transaction
    with transaction(db_path) as db:
        publication = db.execute("""INSERT INTO publications
            (faculty_profile_id,title,year,authors_json,topics_json,source_evidence_id,retrieved_at)
            VALUES(?,?,?,?,?,?,?)""", (professor, "Evaluation of Retrieval Agents", 2026,
            '["Dr Ada One"]', '["retrieval"]', paper_source, utc_now())).lastrowid
    programme = ledger.create_programme("Example University", "PhD Computer Science",
                                        portal_url="https://example.edu/apply")
    opportunity = ledger.create_opportunity("FACULTY_ENQUIRY", "Faculty enquiry demo", "Example University",
        programme_id=programme, contact_policy="ALLOWED", verification_state="VERIFIED",
        source_evidence_id=faculty_source, opening_status="OPEN")
    app = ledger.create_application("2026-27", programme_id=programme, opportunity_id=opportunity,
                                    eligibility_state="ELIGIBLE")
    discovery.link_to_application(app, faculty_id=professor, evidence_id=faculty_source)
    cv_req = ledger.create_requirement(app, "FACULTY_OUTREACH", "REQUIRED", "CV", requirement_source,
                                       normalized_document_type="CV", file_format="pdf")
    proposal_req = ledger.create_requirement(app, "FACULTY_OUTREACH", "REQUIRED", "Research proposal",
                                             requirement_source, normalized_document_type="RESEARCH_PROPOSAL",
                                             file_format="pdf")
    master = studio.create_master_cv(profile_version, [{"name": "Research Experience", "bullets": [
        {"text": "I evaluated retrieval agents with controlled benchmarks.", "claim_revision_ids": [claim]}]}])
    studio.review_master_cv(master, "Demo reviewer", True)
    cv = studio.tailor_cv(master, track_version, application_id=app, faculty_id=professor)
    studio.review_artifact(cv, "Demo reviewer", True)
    proposal = studio.create_proposal(profile_version, track_version, application_id=app, faculty_id=professor,
                                      requirement_id=proposal_req, publication_ids=[publication])
    studio.review_artifact(proposal, "Demo reviewer", True)
    from phd_agent.db import connect
    with connect(db_path) as db:
        cv_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?", (cv,)).fetchone()[0]
        proposal_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?", (proposal,)).fetchone()[0]
    vault.link_to_application(app, cv_req, cv_version)
    vault.link_to_application(app, proposal_req, proposal_version)
    document_package = builder.build(app, profile_version, track_version, context="FACULTY_OUTREACH")
    builder.mark_ready(document_package, "Demo reviewer")
    package = outreach.prepare(professor, app, document_package, profile_version, track_version)
    outreach.approve(package, "Demo reviewer")
    outreach.reconcile_sent([], source="TEST")
    class Transport:
        def send(self, *args):
            return {"message_id": "demo-sent", "thread_id": "demo-thread"}
    sent = outreach.send(package, Transport(), "Demo reviewer")
    reply = outreach.record_reply({"message_id": "demo-reply", "thread_id": "demo-thread",
                                   "subject": "Please send a research proposal", "message_at": utc_now()})
    classified = intel.classify(reply, "PROPOSAL_REQUESTED", "Demo reviewer", requested_length="one page")
    generated = intel.generate_requested_document(classified["id"], profile_version_id=profile_version,
                                                  track_version_id=track_version)
    due = intel.mark_follow_ups_due(now=datetime.now(timezone.utc) + timedelta(days=20))
    follow = intel.draft_follow_up(package)
    answer = portal.save_answer("EDUCATION", "Highest degree", "MSc Computer Science",
                                claim_revision_id=claim)
    portal.approve_answer(answer, "Demo reviewer")
    field = portal.add_checklist_field(app, "EDUCATION", "Highest qualification")
    portal.fill_field(field, answer)
    portal.review_field(field, "Demo reviewer")
    (output_dir / "credentials.json").write_text("{}", encoding="utf-8")
    backup = BackupService(db_path, output_dir).create_backup(output_dir / "backup", "Demo reviewer")
    report = {
        "demo_db": str(db_path),
        "gmail_calls": 0,
        "portal_submits": 0,
        "send": sent,
        "reply_classification": classified,
        "generated_proposal_draft": generated,
        "follow_ups_due": due,
        "follow_up_package": follow,
        "copy_ready": portal.copy_ready(app),
        "backup_excluded_credentials": backup["excluded_credentials"],
        "auto_send_enabled": False,
    }
    (output_dir / "demo-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/slice5-demo"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
