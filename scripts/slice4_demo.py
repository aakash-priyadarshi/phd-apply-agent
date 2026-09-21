"""Reproducible synthetic Slice 4 demonstration; no Gmail or production writes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from phd_agent.db import connect, transaction, utc_now
from phd_agent.discovery import Discovery
from phd_agent.documents import DocumentVault
from phd_agent.ledger import Ledger
from phd_agent.materials import MaterialStudio, render_pdf
from phd_agent.outreach import CampaignPolicy, OutreachService
from phd_agent.packages import PackageBuilder
from phd_agent.profile import ApplicantTruth


def run(output_dir: Path, source_db: Path | None = None) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Choose a new isolated demo directory")
    active_hash = hashlib.sha256(source_db.read_bytes()).hexdigest() if source_db else None
    output_dir.mkdir(parents=True)
    db_path = output_dir / "demo.db"
    ledger, vault, truth, discovery = Ledger(db_path), DocumentVault(db_path), ApplicantTruth(db_path), Discovery(db_path)
    studio, builder, outreach = MaterialStudio(db_path, vault), PackageBuilder(db_path, vault), OutreachService(db_path, vault)
    profile = truth.create_profile("Demo Applicant")
    source = vault.upload(b"Synthetic reviewed source for an isolated test applicant", "demo-source.txt",
                          "SOURCE", "CV", "Demo applicant evidence")
    vault.set_approval(source["version_id"], True, "Demo reviewer")
    claim = truth.create_claim(profile,"RESEARCH_PROJECT",
        "I evaluated retrieval agents using controlled benchmark tasks.","project","FACT",
        source_document_version_id=source["version_id"])
    truth.review_claim(claim,True,"Demo reviewer",verification_state="VERIFIED",for_application=True,for_outreach=True)
    profile_version = truth.create_profile_version(profile,[claim],approve=True,reviewer="Demo reviewer")
    track,track_version = truth.create_track(profile,"Reliable retrieval agents",
        research_problem="How can retrieval agents be evaluated reliably?",
        motivation="Agent evaluation needs controlled evidence.",
        proposed_methodology="Compare agents with reproducible retrieval benchmarks.",
        evaluation_strategy="Measure failure modes and reliability.",
        expected_contribution="Clearer evaluation of retrieval agents.",
        supporting_claim_revision_ids=[claim])
    truth.approve_track(track_version,"Demo reviewer")
    master = studio.create_master_cv(profile_version,[{"name":"Research Experience","bullets":[
        {"text":"I evaluated retrieval agents using controlled benchmark tasks.","claim_revision_ids":[claim]}]}])
    studio.review_master_cv(master,"Demo reviewer",True)
    reusable_cv = vault.upload(render_pdf("Demo CV","# Research Experience\nI evaluated retrieval agents using controlled benchmark tasks."),
        "demo-cv.pdf","SOURCE","CV","Demo reusable CV")
    vault.set_approval(reusable_cv["version_id"],True,"Demo reviewer")
    campaign = outreach.create_campaign("Synthetic research outreach","2026-27",
        CampaignPolicy(timezone_name="UTC",allowed_weekdays=(0,1,2,3,4,5,6),
                       local_start_hour=0,local_end_hour=24,daily_cap=3,minimum_spacing_minutes=0))

    def candidate(name: str, email: str, number: int, *, proposal: bool = False):
        faculty_source = ledger.create_evidence(f"https://example.edu/faculty/{number}","FACULTY",
            f"Synthetic official-style record for {name}: retrieval agent evaluation, email, current affiliation, contact permitted.","VERIFIED")
        requirement_source = ledger.create_evidence(f"https://example.edu/contact/{number}","PROGRAMME",
            "Synthetic faculty contact instructions and document requirements.","VERIFIED")
        professor = discovery.create_faculty(name,"Example University",evidence_id=faculty_source)
        discovery.verify_faculty(professor,"VERIFIED",evidence_by_fact={"IDENTITY":faculty_source,
            "AFFILIATION":faculty_source,"EMAIL":faculty_source,"TOPICS":faculty_source},
            reviewer="Demo reviewer",reason="Synthetic verified demo source",
            updates={"email":email,"email_state":"VERIFIED","research_topics":"retrieval agent evaluation",
                     "affiliation_state":"CURRENT"})
        programme = ledger.create_programme("Example University",f"PhD Computer Science demo {number}")
        opportunity = ledger.create_opportunity("FACULTY_ENQUIRY",f"Faculty enquiry demo {number}",
            "Example University",programme_id=programme,contact_policy="ALLOWED",
            verification_state="VERIFIED",source_evidence_id=faculty_source)
        app = ledger.create_application("2026-27",programme_id=programme,opportunity_id=opportunity)
        discovery.link_to_application(app,faculty_id=professor,evidence_id=faculty_source)
        cv_req = ledger.create_requirement(app,"FACULTY_OUTREACH","REQUIRED","CV",requirement_source,
            normalized_document_type="CV",file_format="pdf")
        if proposal:
            paper_source = ledger.create_evidence(f"https://example.edu/paper/{number}","PUBLICATION",
                "A stored synthetic publication record for retrieval agent evaluation.","VERIFIED")
            with transaction(db_path) as db:
                publication = db.execute("""INSERT INTO publications
                    (faculty_profile_id,title,year,authors_json,topics_json,source_evidence_id,retrieved_at)
                    VALUES(?,?,?,?,?,?,?)""",(professor,"Evaluation of Retrieval Agents",2026,
                    json.dumps([name]),'["retrieval", "agents"]',paper_source,utc_now())).lastrowid
            proposal_req = ledger.create_requirement(app,"FACULTY_OUTREACH","REQUIRED","Research proposal",
                requirement_source,normalized_document_type="RESEARCH_PROPOSAL",file_format="pdf")
            cv_artifact = studio.tailor_cv(master,track_version,application_id=app,faculty_id=professor)
            studio.review_artifact(cv_artifact,"Demo reviewer",True)
            proposal_artifact = studio.create_proposal(profile_version,track_version,application_id=app,
                faculty_id=professor,requirement_id=proposal_req,publication_ids=[publication])
            studio.review_artifact(proposal_artifact,"Demo reviewer",True)
            with connect(db_path) as db:
                cv_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?",(cv_artifact,)).fetchone()[0]
                proposal_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?",(proposal_artifact,)).fetchone()[0]
            vault.link_to_application(app,cv_req,cv_version)
            vault.link_to_application(app,proposal_req,proposal_version)
        else:
            vault.link_to_application(app,cv_req,reusable_cv["version_id"])
        doc_package = builder.build(app,profile_version,track_version,context="FACULTY_OUTREACH")
        builder.mark_ready(doc_package,"Demo reviewer")
        return professor,app,doc_package

    a = candidate("Dr Ada One","ada.one@example.edu",1,proposal=True)
    b = candidate("Dr Ben Two","ben.two@example.edu",2)
    c = candidate("Dr Cara Three","cara.three@example.edu",3)
    outreach.record_manual_contact(b[0],"ben.two@example.edu","Earlier enquiry",utc_now(),"Demo reviewer")
    a_package = outreach.prepare(a[0],a[1],a[2],profile_version,track_version,campaign_id=campaign)
    outreach.approve(a_package,"Demo reviewer")
    c_package = outreach.prepare(c[0],c[1],c[2],profile_version,track_version,campaign_id=campaign)
    outreach.approve(c_package,"Demo reviewer")
    first_dry_run = outreach.dry_run(campaign,profile_version,track_version,
                                    now=datetime(2026,9,21,12,tzinfo=timezone.utc))
    outreach.reconcile_sent([],source="TEST")
    class TimeoutTransport:
        def send(self,*_):
            raise TimeoutError("Simulated uncertain network outcome")
    ambiguous = outreach.send(c_package,TimeoutTransport(),"Demo reviewer",
                              now=datetime(2026,9,21,12,tzinfo=timezone.utc))
    recovery = outreach.recover_ambiguous(ambiguous["message_id"],"Demo reviewer")
    second_dry_run = outreach.dry_run(campaign,profile_version,track_version,
                                     now=datetime(2026,9,21,12,tzinfo=timezone.utc))
    with connect(db_path) as db:
        a_gate = json.loads(db.execute("SELECT quality_json FROM outreach_packages WHERE id=?",(a_package,)).fetchone()[0])
        a_status = db.execute("SELECT status FROM outreach_packages WHERE id=?",(a_package,)).fetchone()[0]
        c_status = db.execute("SELECT status FROM outreach_packages WHERE id=?",(c_package,)).fetchone()[0]
    report = {"demo_db":str(db_path),"campaign_id":campaign,
        "professor_a":{"id":a[0],"package_id":a_package,"status":a_status,"quality":a_gate["status"],
                       "attachments":[x.document_type for x in outreach._unpack(outreach.get_package(a_package))[0].attachments]},
        "professor_b":{"id":b[0],"blocker":"PREVIOUS_GMAIL_CONTACT"},
        "professor_c":{"id":c[0],"package_id":c_package,"status":c_status,
                       "send_result":ambiguous,"recovery":recovery},
        "dry_run_before_ambiguous":first_dry_run["candidates"],
        "dry_run_after_ambiguous":second_dry_run["candidates"],
        "auto_send_enabled":False,"gmail_calls":0,
        "active_db_unchanged":source_db is None or hashlib.sha256(source_db.read_bytes()).hexdigest()==active_hash}
    (output_dir/"demo-report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("data/slice4-demo"))
    parser.add_argument("--source-db",type=Path,default=None)
    args=parser.parse_args()
    if args.source_db is not None and not args.source_db.is_file():
        raise SystemExit("--source-db must point to an existing database file")
    result=run(args.output_dir,args.source_db)
    print(json.dumps(result,indent=2))
