"""Isolated Stanford demonstration. Never writes to the active applicant database.

The real CV is read into a separate local Vault. Synthetic transcripts are marked
DEMO_ONLY and deterministically block READY until genuine uploads replace them.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sqlite3
from pathlib import Path

from PyPDF2 import PdfReader

from phd_agent.db import connect, utc_now, transaction
from phd_agent.documents import DocumentVault
from phd_agent.ledger import Ledger
from phd_agent.matching import MatchEngine
from phd_agent.materials import MaterialStudio, render_pdf
from phd_agent.packages import PackageBuilder
from phd_agent.profile import ApplicantTruth


def run(source_db: Path, output_dir: Path, *, liverpool_results: Path | None = None,
        galgotias_transcript: Path | None = None) -> dict:
    source_db = source_db.resolve()
    output_dir = output_dir.resolve()
    source_hash_before = hashlib.sha256(source_db.read_bytes()).hexdigest()
    if output_dir.exists():
        raise FileExistsError("Demo output already exists; choose a new directory")
    output_dir.mkdir(parents=True)
    db_path = output_dir / "phd_outreach.db"
    with sqlite3.connect(source_db) as source, sqlite3.connect(db_path) as target:
        source.backup(target)
    if (source_db.parent / "documents").exists():
        shutil.copytree(source_db.parent / "documents", output_dir / "documents")
    ledger, vault = Ledger(db_path), DocumentVault(db_path)
    truth, studio, builder = ApplicantTruth(db_path, vault), MaterialStudio(db_path, vault), PackageBuilder(db_path, vault)

    # Sandbox operator decisions based on the supplied CV, never the active database.
    vault.set_approval(1, True, "Slice 3 sandbox CV review")
    for claim_id in (1, 3, 4):
        truth.review_claim(claim_id, True, "Slice 3 sandbox CV review", verification_state="VERIFIED",
                           for_application=True, for_outreach=True,
                           notes="CV-supported sandbox approval; independent credential proof still needed")
    truth.review_claim(9, True, "Slice 3 sandbox aspiration review", for_application=True,
                       notes="Future aim; no achieved-result claim")
    profile_version = truth.create_profile_version(1, [1, 3, 4, 9], approve=True,
                                                    reviewer="Slice 3 sandbox review")
    track_version = truth.revise_track(1, supporting_claim_revision_ids=[3, 4])
    truth.approve_track(track_version, "Slice 3 sandbox review")

    # The source URLs were manually checked on 2026-09-21. No live scraping is needed
    # to reproduce this demonstration; the short paraphrases identify the checked facts.
    checklist = ledger.create_evidence(
        "https://www.cs.stanford.edu/admissions/graduate-application-checklists", "PROGRAMME",
        "Stanford CS lists a two-page SOP, resume/CV, unofficial transcripts for attended institutions, three online recommendations, and conditional English proficiency evidence.",
        "VERIFIED")
    work_evidence = ledger.create_evidence(
        "https://cs.stanford.edu/~diyiy/research.html", "FACULTY",
        "Diyi Yang's official Stanford research page lists CollabSkill: Evaluating Human-Agent Collaboration On Real-World Tasks (COLM 2026).",
        "VERIFIED")
    with transaction(db_path) as db:
        publication_id = db.execute("""INSERT INTO publications
            (faculty_profile_id,title,year,venue,authors_json,topics_json,source_evidence_id,retrieved_at,notes)
            VALUES(?,?,?,?,?,?,?,?,?)""", (67,
            "CollabSkill: Evaluating Human-Agent Collaboration On Real-World Tasks", 2026, "COLM 2026",
            json.dumps(["Yijia Shao", "Zora Zhiruo Wang", "Neel Ahuja", "Yicheng Wang", "Bowen Liu", "Diyi Yang"]),
            json.dumps(["human-agent collaboration", "agent evaluation"]),work_evidence,utc_now(),
            "Official faculty research-page review in isolated Slice 3 demo")).lastrowid

    application = ledger.create_application("2026-27", programme_id=1, opportunity_id=1,
        portal_url="https://gradadmissions.stanford.edu/apply", eligibility_state="UNKNOWN",
        next_action="Replace demo transcripts, verify eligibility and English exemption, arrange three referees")
    ledger.create_deadline(application, "APPLICATION", "2026-12-08", 40, verification_state="VERIFIED")
    cv_req = ledger.create_requirement(application, "FORMAL_APPLICATION", "REQUIRED", "Resume / CV", checklist,
                                       normalized_document_type="CV", file_format="pdf")
    sop_req = ledger.create_requirement(application, "FORMAL_APPLICATION", "REQUIRED", "Statement of purpose", checklist,
                                        normalized_document_type="SOP", page_limit=2, file_format="pdf")
    transcript_reqs = [ledger.create_requirement(application, "FORMAL_APPLICATION", "REQUIRED", f"Unofficial {school} transcript", checklist,
        normalized_document_type="TRANSCRIPT", file_format="pdf") for school in ("Liverpool", "Galgotias")]
    ledger.create_requirement(application, "FORMAL_APPLICATION", "UNKNOWN", "English test applicability / exemption", checklist,
        normalized_document_type="ENGLISH_TEST", condition_text="Review Stanford exemption rules and applicant evidence")
    ledger.create_requirement(application, "FORMAL_APPLICATION", "REQUIRED", "3 recommendation letters", checklist)
    ledger.create_requirement(application, "FORMAL_APPLICATION", "REQUIRED", "Application fee", checklist)

    with connect(db_path) as db:
        claim_texts = {r["id"]: r["claim_text"] for r in db.execute(
            "SELECT id,claim_text FROM claim_revisions WHERE id IN (1,3,4,9)")}
    master = studio.create_master_cv(profile_version, [
        {"name":"Education","bullets":[{"text":claim_texts[1],"claim_revision_ids":[1]}]},
        {"name":"Research Experience","bullets":[
            {"text":claim_texts[3],"claim_revision_ids":[3]},
            {"text":claim_texts[4],"claim_revision_ids":[4]}]},
        {"name":"Profile","bullets":[{"text":claim_texts[9],"claim_revision_ids":[9]}]},
    ])
    studio.review_master_cv(master, "Slice 3 sandbox review", True)
    cv = studio.tailor_cv(master, track_version, application_id=application,
                          selected_sections=["Research Experience","Education","Profile"])
    studio.review_artifact(cv, "Slice 3 sandbox review", True)
    module = studio.create_module(profile_version, "RESEARCH_IDENTITY",
        claim_texts[4] + "\n\n" + claim_texts[9],
        [4, 9])
    studio.review_module(module, "Slice 3 sandbox review", True)
    sop = studio.create_statement("SOP", profile_version, track_version, application, sop_req, [module])
    studio.review_artifact(sop, "Slice 3 sandbox review", True)
    proposal = studio.create_proposal(profile_version, track_version, faculty_id=67,
                                      publication_ids=[publication_id], format_name="CONCEPT_NOTE")
    studio.review_artifact(proposal, "Slice 3 sandbox review", True)
    with connect(db_path) as db:
        cv_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?", (cv,)).fetchone()[0]
        sop_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?", (sop,)).fetchone()[0]
    vault.link_to_application(application, cv_req, cv_version)
    vault.link_to_application(application, sop_req, sop_version)

    # Real files are copied into the isolated Vault when explicitly supplied.
    # Liverpool's assessment-results printout is provisional until the user
    # verifies that Stanford accepts it as an unofficial transcript.
    document_sources = {}
    for school, requirement_id, supplied in zip(("Liverpool", "Galgotias"), transcript_reqs,
                                                (liverpool_results, galgotias_transcript)):
        if supplied:
            supplied = supplied.resolve(strict=True)
            pdf = supplied.read_bytes()
            if not PdfReader(io.BytesIO(pdf)).pages:
                raise ValueError("Supplied transcript PDF has no pages")
            filename = supplied.name
            notes = "PROVISIONAL_TRANSCRIPT" if school == "Liverpool" else "Applicant-supplied transcript; sandbox review"
            title = f"{school} source document"
            document_sources[school] = str(supplied)
        else:
            pdf = render_pdf("DEMO ONLY — NOT A REAL TRANSCRIPT", f"{school} synthetic transcript placeholder. Replace with applicant document.")
            filename = f"DEMO_ONLY_{school}_transcript.pdf"
            notes = "DEMO_ONLY"
            title = f"DEMO ONLY {school} transcript"
            document_sources[school] = "SYNTHETIC_DEMO_ONLY"
        version = vault.upload(pdf, filename, "SOURCE", "TRANSCRIPT", title,
                               sensitivity="CONFIDENTIAL", notes=notes)
        vault.set_approval(version["version_id"], True, "Slice 3 sandbox review")
        vault.link_to_application(application, requirement_id, version["version_id"])

    match = MatchEngine(db_path).assess(67, profile_version, track_version, application)
    package = builder.build(application, profile_version, track_version, combine_pdf=True, zip_export=True)
    preflight = builder.preflight(package)
    try:
        builder.mark_ready(package, "Slice 3 sandbox review")
    except ValueError:
        ready_blocked = True
    else:
        ready_blocked = False
    with connect(db_path) as db:
        claim_counts = {r[0]:r[1] for r in db.execute("SELECT review_status,COUNT(*) FROM claim_revisions GROUP BY review_status")}
        historical = db.execute("SELECT COUNT(*) FROM professors").fetchone()[0]
        package_row = db.execute("SELECT export_path,package_sha256,status FROM application_packages WHERE id=?", (package,)).fetchone()
    report = {"demo_db":str(db_path),"application_id":application,"profile_version_id":profile_version,
        "track_version_id":track_version,"claim_counts":claim_counts,"faculty_id":67,"publication_id":publication_id,
        "research_fit":match["research_fit"],"research_fit_coverage":match["research_fit_coverage"],
        "research_fit_components":match["components"],"application_readiness":match["application_readiness"],
        "artifact_ids":{"CV":cv,"SOP":sop,"RESEARCH_PROPOSAL":proposal},"package_id":package,
        "package_status":package_row["status"],"package_sha256":package_row["package_sha256"],
        "export_path":package_row["export_path"],"preflight_status":preflight["status"],
        "preflight_blocks":[r for r in preflight["rules"] if r["severity"]=="BLOCK"],
        "ready_blocked":ready_blocked,"historical_professors":historical,
        "document_sources":document_sources,
        "synthetic_transcripts":any(v=="SYNTHETIC_DEMO_ONLY" for v in document_sources.values()),
        "production_database_unchanged":hashlib.sha256(source_db.read_bytes()).hexdigest()==source_hash_before}
    if not report["production_database_unchanged"]:
        raise RuntimeError("Active database changed while creating the demo")
    (output_dir / "demo-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=Path("data/phd_outreach.db"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/slice3-demo-2026-09-21"))
    parser.add_argument("--liverpool-results", type=Path)
    parser.add_argument("--galgotias-transcript", type=Path)
    args = parser.parse_args()
    result = run(args.source_db, args.output_dir, liverpool_results=args.liverpool_results,
                 galgotias_transcript=args.galgotias_transcript)
    print(json.dumps({k:v for k,v in result.items() if k not in {"research_fit_components","application_readiness","preflight_blocks"}}, indent=2))
    print("Blocking rules:", [x["rule_id"] for x in result["preflight_blocks"]])
