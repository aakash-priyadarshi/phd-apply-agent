"""Slice 3 gates use temporary SQLite and Vault storage only."""

import hashlib
import json
import sqlite3
from datetime import date, timedelta

import pytest

from phd_agent.db import connect, transaction, utc_now
from phd_agent.discovery import Discovery
from phd_agent.documents import DocumentVault
from phd_agent.ledger import Ledger
from phd_agent.matching import MatchEngine, contact_policy_state
from phd_agent.materials import MaterialStudio, render_pdf
from phd_agent.packages import PackageBuilder, combine_selected_pdfs, package_policy
from phd_agent.profile import ApplicantTruth


@pytest.fixture
def scenario(tmp_path):
    db_path = tmp_path / "test.db"
    ledger, vault = Ledger(db_path), DocumentVault(db_path)
    truth = ApplicantTruth(db_path, vault)
    discovery = Discovery(db_path)
    studio = MaterialStudio(db_path, vault)
    builder = PackageBuilder(db_path, vault)
    profile = truth.create_profile("Demo applicant")
    source = vault.upload(b"Reviewed demo CV source", "cv-source.txt", "SOURCE", "CV", "Reviewed CV")
    vault.set_approval(source["version_id"], True, "Demo reviewer")
    fact = truth.create_claim(profile, "RESEARCH_PROJECT", "I evaluated reliable agents with retrieval methods.",
                              "project", "FACT", source_document_version_id=source["version_id"])
    truth.review_claim(fact, True, "Demo reviewer", verification_state="VERIFIED", for_application=True, for_outreach=True)
    aim = truth.create_claim(profile, "OTHER", "I aim to study reliable retrieval agents.", "goal", "ASPIRATION")
    truth.review_claim(aim, True, "Demo reviewer", for_application=True)
    rejected = truth.create_claim(profile, "PUBLICATION", "I published an invented paper.", "publication", "FACT",
                                  source_document_version_id=source["version_id"])
    truth.review_claim(rejected, False, "Demo reviewer")
    pv = truth.create_profile_version(profile, [fact, aim], approve=True, reviewer="Demo reviewer")
    track_id, tv = truth.create_track(profile, "Reliable retrieval agents", research_problem="How can retrieval agents be reliable?",
                                      motivation="I aim to study reliable retrieval agents.",
                                      proposed_methodology="Evaluate retrieval agents with controlled benchmarks.",
                                      evaluation_strategy="Compare reliability on benchmark tasks.",
                                      expected_contribution="More reliable retrieval agents.",
                                      supporting_claim_revision_ids=[fact])
    truth.approve_track(tv, "Demo reviewer")
    base_master = studio.create_master_cv(pv, [
        {"name":"Research Experience","bullets":[{"text":"I evaluated reliable agents with retrieval methods.","claim_revision_ids":[fact]}]},
        {"name":"Profile","bullets":[{"text":"I aim to study reliable retrieval agents.","claim_revision_ids":[aim]}]},
    ])
    studio.review_master_cv(base_master, "Demo reviewer", True)
    evidence = ledger.create_evidence("https://example.edu/phd", "PROGRAMME", "Reviewed admissions and document requirements", "VERIFIED")
    faculty_evidence = ledger.create_evidence("https://example.edu/faculty/demo", "FACULTY", "Research on retrieval and agent reliability", "VERIFIED")
    publication_evidence = ledger.create_evidence("https://openalex.org/W999", "PUBLICATION", "Stored publication record", "VERIFIED")
    programme = ledger.create_programme("Example University", "PhD Computer Science", portal_url="https://example.edu/apply")
    app = ledger.create_application("2027", programme_id=programme, eligibility_state="ELIGIBLE")
    deadline = ledger.create_deadline(app, "APPLICATION", (date.today()+timedelta(days=90)).isoformat(), evidence,
                                      verification_state="VERIFIED")
    cv_req = ledger.create_requirement(app,"FORMAL_APPLICATION","REQUIRED","CV",evidence,normalized_document_type="CV",file_format="pdf")
    sop_req = ledger.create_requirement(app,"FORMAL_APPLICATION","REQUIRED","SOP",evidence,normalized_document_type="SOP",word_limit=250,file_format="pdf")
    transcript_req = ledger.create_requirement(app,"FORMAL_APPLICATION","REQUIRED","Transcript",evidence,normalized_document_type="TRANSCRIPT",file_format="pdf")
    faculty = discovery.create_faculty("Dr Demo", "Example University", evidence_id=faculty_evidence)
    discovery.verify_faculty(faculty, "VERIFIED", evidence_by_fact={"IDENTITY":faculty_evidence,
        "AFFILIATION":faculty_evidence,"TOPICS":faculty_evidence}, reviewer="Demo reviewer", reason="Reviewed official profile",
        updates={"research_topics":"reliable retrieval agents", "affiliation_state":"CURRENT"})
    with transaction(db_path) as db:
        publication = db.execute("""INSERT INTO publications
            (faculty_profile_id,title,year,authors_json,topics_json,source_evidence_id,retrieved_at)
            VALUES(?,?,?,?,?,?,?)""", (faculty,"Evaluation of Retrieval Agents",date.today().year,
            '["Dr Demo"]','["retrieval", "agents"]',publication_evidence,utc_now())).lastrowid
    return locals()


def _prepare_cv(s):
    with pytest.raises(ValueError, match="approved profile"):
        s["studio"].create_proposal(999,s["tv"],faculty_id=s["faculty"])
    with pytest.raises(ValueError, match="approved application claims"):
        s["studio"].create_master_cv(s["pv"],[{"name":"Research Experience","bullets":[{"text":"Invented","claim_revision_ids":[s["rejected"]]}]}])
    sections = [{"name":"Research Experience","bullets":[{"text":"I evaluated reliable agents with retrieval methods.","claim_revision_ids":[s["fact"]]}]},
                {"name":"Profile","bullets":[{"text":"I aim to study reliable retrieval agents.","claim_revision_ids":[s["aim"]]}]}]
    master = s["studio"].create_master_cv(s["pv"],sections)
    s["studio"].review_master_cv(master,"Demo reviewer",True)
    with connect(s["db_path"]) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE master_cv_versions SET sections_json='[]' WHERE id=?",(master,))
    master2 = s["studio"].create_master_cv(s["pv"],sections)
    assert master2 > master
    cv = s["studio"].tailor_cv(master,s["tv"],application_id=s["app"],selected_sections=["Research Experience"])
    with connect(s["db_path"]) as db:
        artifact = db.execute("SELECT * FROM generated_artifacts WHERE id=?",(cv,)).fetchone()
    assert "evaluated reliable agents" in artifact["content_text"]
    assert json.loads(artifact["diff_json"])["removed"] == ["Profile"]
    assert json.loads(artifact["claim_revision_ids_json"]) == [s["fact"]]
    s["studio"].review_artifact(cv,"Demo reviewer",True)
    with connect(s["db_path"]) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE document_versions SET sha256='tampered' WHERE id=?", (artifact["document_version_id"],))
    return cv


def test_review_gate_cv_versioning_and_diff(scenario):
    _prepare_cv(scenario)


def test_statement_constraints_proposal_citations_and_cover_policy(scenario):
    s = scenario
    module = s["studio"].create_module(s["pv"],"WHY_PHD","I evaluated reliable agents with retrieval methods.",[s["fact"]])
    s["studio"].review_module(module,"Demo reviewer",True)
    sop = s["studio"].create_statement("SOP",s["pv"],s["tv"],s["app"],s["sop_req"],[module])
    s["studio"].review_artifact(sop,"Demo reviewer",True)
    edited = s["studio"].revise_artifact(sop, "I published an unsupported paper at Stanford.")
    with connect(s["db_path"]) as db:
        edited_quality = json.loads(db.execute("SELECT quality_json FROM generated_artifacts WHERE id=?",(edited,)).fetchone()[0])
    assert any("UNREVIEWED_MANUAL_CONTENT" in x for x in edited_quality["blockers"])
    with pytest.raises(ValueError, match="quality blockers"):
        s["studio"].review_artifact(edited,"Demo reviewer",True)
    with pytest.raises(ValueError, match="Cover letter"):
        s["studio"].create_cover_letter(s["pv"],s["tv"],s["app"])
    assert s["studio"].create_cover_letter(s["pv"],s["tv"],s["app"],intentional=True)
    with pytest.raises(ValueError, match="publication"):
        s["studio"].create_proposal(s["pv"],s["tv"],faculty_id=s["faculty"],publication_ids=[999])
    proposal = s["studio"].create_proposal(s["pv"],s["tv"],faculty_id=s["faculty"],publication_ids=[s["publication"]])
    with connect(s["db_path"]) as db:
        artifact = db.execute("SELECT * FROM generated_artifacts WHERE id=?",(proposal,)).fetchone()
    assert artifact["research_track_version_id"] == s["tv"]
    assert "Evaluation of Retrieval Agents" in artifact["content_text"]
    assert "research_problem" in json.loads(artifact["diff_json"])["preserved"]
    # A shortened word limit is a blocking quality result, not silent truncation.
    s["ledger"].update_requirement(s["sop_req"],word_limit=3)
    short = s["studio"].create_statement("SOP",s["pv"],s["tv"],s["app"],s["sop_req"],[module])
    with connect(s["db_path"]) as db:
        quality = json.loads(db.execute("SELECT quality_json FROM generated_artifacts WHERE id=?",(short,)).fetchone()[0])
    assert "WORD_LIMIT" in quality["blockers"]
    with pytest.raises(ValueError, match="quality blockers"):
        s["studio"].review_artifact(short,"Demo reviewer",True)


def test_other_application_contamination_preserves_approved_education_fact(scenario):
    s = scenario
    s["ledger"].create_programme("Other University", "PhD AI")
    education = "I completed an MSc at Other University."
    education_id = s["truth"].create_claim(s["profile"], "EDUCATION", education, "degree", "FACT",
        source_document_version_id=s["source"]["version_id"])
    s["truth"].review_claim(education_id, True, "Demo reviewer", verification_state="VERIFIED", for_application=True)
    clean = s["studio"]._quality("CV", "# Education\n• " + education,
        s["app"], None, [education_id], [], render_pdf("CV", education))
    assert not clean["blockers"]
    contaminated = s["studio"]._quality("SOP", "I want to apply at Other University.",
        s["app"], s["sop_req"], [s["fact"]], [], render_pdf("SOP", "Wrong institution"))
    assert "OTHER_INSTITUTION:Other University" in contaminated["blockers"]


def test_match_components_unknowns_and_review_history(scenario):
    s = scenario
    match = MatchEngine(s["db_path"]).assess(s["faculty"],s["pv"],s["tv"],s["app"])
    assert match["research_fit"] is not None
    assert match["research_fit_coverage"] > 0
    assert match["components"]["recent_work"]["evidence_ids"] == [s["publication_evidence"]]
    assert match["application_readiness"]["contact_policy"]["state"] == "UNKNOWN"
    assert match["application_readiness"]["deadline"]["state"] == "PASS"
    assert match["application_readiness"]["eligibility"]["state"] == "PASS"
    engine = MatchEngine(s["db_path"])
    one = engine.review(match["id"],"Demo reviewer","Topic match needs review",{"research_fit":4.5})
    two = engine.review(match["id"],"Demo reviewer","Reconsidered")
    assert two > one
    with connect(s["db_path"]) as db:
        assert db.execute("SELECT research_fit FROM match_assessments WHERE id=?",(match["id"],)).fetchone()[0] == match["research_fit"]
    with transaction(s["db_path"]) as db:
        db.execute("DELETE FROM publications")
    newer = engine.assess(s["faculty"],s["pv"],s["tv"],s["app"])
    assert newer["components"]["recent_work"]["score"] is None
    assert "recent_work" in newer["unknowns"]


def test_policy_package_hash_manifest_combined_preflight(scenario, monkeypatch):
    s = scenario
    assert package_policy({"context":"FORMAL_APPLICATION","requirement_state":"UNKNOWN","document_state":"MISSING"},context="FORMAL_APPLICATION")["decision"] == "REVIEW"
    assert package_policy({"context":"FORMAL_APPLICATION","requirement_state":"OPTIONAL","document_state":"AVAILABLE"},context="FORMAL_APPLICATION")["decision"] == "EXCLUDE"
    with pytest.raises(ValueError, match="Required documents"):
        s["builder"].build(s["app"],s["pv"],s["tv"])
    cv = _prepare_cv(s)
    module = s["studio"].create_module(s["pv"],"WHY_PHD","I evaluated reliable agents with retrieval methods.",[s["fact"]])
    s["studio"].review_module(module,"Demo reviewer",True)
    sop = s["studio"].create_statement("SOP",s["pv"],s["tv"],s["app"],s["sop_req"],[module])
    s["studio"].review_artifact(sop,"Demo reviewer",True)
    with connect(s["db_path"]) as db:
        cv_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?",(cv,)).fetchone()[0]
        sop_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?",(sop,)).fetchone()[0]
    transcript = s["vault"].upload(render_pdf("Transcript","Completed degree"),"transcript.pdf","SOURCE","TRANSCRIPT","Transcript",sensitivity="CONFIDENTIAL")
    s["vault"].set_approval(transcript["version_id"],True,"Demo reviewer")
    for req,version in [(s["cv_req"],cv_version),(s["sop_req"],sop_version),(s["transcript_req"],transcript["version_id"])]:
        s["vault"].link_to_application(s["app"],req,version)
    package_id = s["builder"].build(s["app"],s["pv"],s["tv"],combine_pdf=True,zip_export=True)
    with connect(s["db_path"]) as db:
        p = db.execute("SELECT * FROM application_packages WHERE id=?",(package_id,)).fetchone()
        docs = db.execute("SELECT * FROM package_documents WHERE package_id=? ORDER BY sort_order",(package_id,)).fetchall()
    assert len(docs) == 3 and p["status"] == "DRAFT"
    manifest = json.loads(p["manifest_json"])
    assert hashlib.sha256((__import__("pathlib").Path(p["export_path"])/"manifest.json").read_bytes()).hexdigest() == p["package_sha256"]
    assert [x["filename"] for x in manifest["combined_pdf"]["components"]] == [d["canonical_filename"] for d in docs]
    preflight = s["builder"].preflight(package_id)
    assert preflight["status"] in {"PASS","WARNING"}
    s["builder"].mark_ready(package_id,"Demo reviewer")
    s["ledger"].update_application(s["app"],status="READY_TO_SUBMIT")
    monkeypatch.setattr("phd_agent.packages.freshness", lambda *_: "STALE")
    assert s["builder"].preflight(package_id)["status"] == "BLOCK"


def test_combining_preserves_order_and_rejects_non_pdf():
    a,b=render_pdf("First","Alpha"),render_pdf("Second","Beta")
    joined=combine_selected_pdfs([("a.pdf",a),("b.pdf",b)])
    assert len(__import__("PyPDF2").PdfReader(__import__("io").BytesIO(joined)).pages) == 2
    with pytest.raises(ValueError,match="PDFs only"):
        combine_selected_pdfs([("a.pdf",a),("b.txt",b)])


def test_unknown_and_non_file_requirements_stay_visible(scenario):
    s=scenario
    unknown=s["ledger"].create_requirement(s["app"],"FORMAL_APPLICATION","UNKNOWN","English exemption",s["evidence"],normalized_document_type="ENGLISH_TEST")
    admin=s["ledger"].create_requirement(s["app"],"FORMAL_APPLICATION","REQUIRED","Application fee",s["evidence"])
    preview=s["builder"].preview(s["app"])
    by_id={d["requirement_id"]:d for d in preview["decisions"]}
    assert by_id[unknown]["decision"] == "REVIEW"
    assert by_id[admin]["decision"] == "EXCLUDE"
    assert preview["completeness"]["unknown_requirements"] == 1


def test_outreach_preflight_uses_contact_rules_not_formal_submission_rules(scenario):
    s = scenario
    assert contact_policy_state("No unsolicited contact") == "BLOCK"
    assert contact_policy_state("See admissions page") == "UNKNOWN"
    assert contact_policy_state("DO_NOT_CONTACT") == "BLOCK"
    assert contact_policy_state("ALLOWED: official faculty page") == "PASS"
    opportunity = s["ledger"].create_opportunity(
        "FACULTY_ENQUIRY", "Faculty research enquiry", "Example University",
        programme_id=s["programme"], contact_policy="DO_NOT_CONTACT",
        verification_state="VERIFIED", source_evidence_id=s["faculty_evidence"])
    s["ledger"].create_requirement(s["app"], "FACULTY_OUTREACH", "REQUIRED", "CV",
                                   s["evidence"], normalized_document_type="CV", file_format="pdf")
    s["ledger"].update_application(s["app"], eligibility_state="UNKNOWN")
    with transaction(s["db_path"]) as db:
        db.execute("UPDATE applications SET opportunity_id=? WHERE id=?", (opportunity, s["app"]))
        db.execute("INSERT INTO application_faculty(application_id,faculty_profile_id,linked_at) VALUES(?,?,?)",
                   (s["app"], s["faculty"], utc_now()))
    cv = _prepare_cv(s)
    with connect(s["db_path"]) as db:
        cv_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?", (cv,)).fetchone()[0]
        outreach_req = db.execute("SELECT id FROM requirements WHERE application_id=? AND context='FACULTY_OUTREACH'", (s["app"],)).fetchone()[0]
    s["vault"].link_to_application(s["app"], outreach_req, cv_version)
    package = s["builder"].build(s["app"], s["pv"], s["tv"], context="FACULTY_OUTREACH")
    result = s["builder"].preflight(package)
    rule_ids = {r["rule_id"] for r in result["rules"]}
    assert not {"DEADLINE", "ROUTE", "ELIGIBILITY", "REFEREE_SCOPE"} & rule_ids
    assert any(r["rule_id"] == "OUTREACH_POLICY" and r["severity"] == "BLOCK" for r in result["rules"])


def test_missing_eligibility_stays_unknown(scenario):
    s = scenario
    readiness = MatchEngine._readiness(
        None, [], {"eligibility_state": None, "portal_url": None}, None, None, [], [])
    assert readiness["eligibility"]["state"] == "UNKNOWN"
    s["ledger"].update_application(s["app"], eligibility_state="UNKNOWN")
    match = MatchEngine(s["db_path"]).assess(s["faculty"], s["pv"], s["tv"], s["app"])
    assert match["application_readiness"]["eligibility"]["state"] == "UNKNOWN"


def test_preflight_invalid_filename_rule_and_unknown_referee_count(scenario):
    s = scenario
    s["ledger"].update_requirement(s["cv_req"], filename_rule="[unclosed")
    s["ledger"].create_requirement(s["app"], "FORMAL_APPLICATION", "REQUIRED",
                                   "Letters of recommendation", s["evidence"])
    cv = _prepare_cv(s)
    module = s["studio"].create_module(s["pv"], "WHY_PHD",
        "I evaluated reliable agents with retrieval methods.", [s["fact"]])
    s["studio"].review_module(module, "Demo reviewer", True)
    sop = s["studio"].create_statement("SOP", s["pv"], s["tv"], s["app"], s["sop_req"], [module])
    s["studio"].review_artifact(sop, "Demo reviewer", True)
    with connect(s["db_path"]) as db:
        cv_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?", (cv,)).fetchone()[0]
        sop_version = db.execute("SELECT document_version_id FROM generated_artifacts WHERE id=?", (sop,)).fetchone()[0]
    transcript = s["vault"].upload(render_pdf("Transcript", "Completed degree"), "transcript.pdf",
                                   "SOURCE", "TRANSCRIPT", "Transcript", sensitivity="CONFIDENTIAL")
    s["vault"].set_approval(transcript["version_id"], True, "Demo reviewer")
    for req, version in [(s["cv_req"], cv_version), (s["sop_req"], sop_version),
                         (s["transcript_req"], transcript["version_id"])]:
        s["vault"].link_to_application(s["app"], req, version)
    package_id = s["builder"].build(s["app"], s["pv"], s["tv"])
    rules = {r["rule_id"]: r for r in s["builder"].preflight(package_id)["rules"]}
    assert rules["FILENAME"]["severity"] == "BLOCK"
    assert "REFEREE_COUNT_UNKNOWN" in rules
    assert rules["REFEREE_COUNT_UNKNOWN"]["severity"] == "BLOCK"
