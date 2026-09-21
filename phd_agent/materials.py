"""Reviewed source-bound CV, statement, proposal, and letter versions."""

from __future__ import annotations

import html
import io
import json
import re
from pathlib import Path

from PyPDF2 import PdfReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.documents import DocumentVault


SECTIONS = {"Profile", "Education", "Research Experience", "Industry and Research Engineering", "Projects", "Publications", "Patents", "Teaching", "Awards", "Technical Skills"}
STATEMENTS = {"SOP", "PERSONAL_STATEMENT", "RESEARCH_STATEMENT"}
GENERATED = {"CV", *STATEMENTS, "RESEARCH_PROPOSAL", "COVER_LETTER"}
PLACEHOLDERS = re.compile(r"\[(?:insert|todo|name|university|professor|programme)[^]]*\]|\b(?:TBD|TODO|Lorem ipsum)\b", re.I)


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pdf_markup(value: str) -> str:
    return html.escape(value, quote=True)


def render_pdf(title: str, text: str) -> bytes:
    """Conservative paginated PDF; source remains editable in the database."""
    output = io.BytesIO()
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "Helvetica"
    styles["Normal"].fontSize = 10
    styles["Normal"].leading = 14
    doc = SimpleDocTemplate(output, pagesize=(595.28, 841.89), leftMargin=58,
                            rightMargin=58, topMargin=56, bottomMargin=56)
    story = [Paragraph(_pdf_markup(title), styles["Title"]), Spacer(1, 10)]
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for line in paragraph.splitlines():
            line = line.strip().replace("• ", "- ")
            if not line:
                continue
            if line.startswith("# "):
                story.append(Paragraph(_pdf_markup(line[2:]), styles["Heading2"]))
            else:
                story.append(Paragraph(_pdf_markup(line), styles["Normal"]))
                story.append(Spacer(1, 3))
        story.append(Spacer(1, 8))
    def page_number(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(537, 30, str(document.page))
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return output.getvalue()


class MaterialStudio:
    def __init__(self, db_path: Path | str, vault: DocumentVault | None = None):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.vault = vault or DocumentVault(self.db_path)

    def _approved_context(self, profile_version_id: int, track_version_id: int, use: str = "application") -> tuple[dict, dict, dict[int, dict]]:
        if use not in {"application", "outreach"}:
            raise ValueError("Invalid claim use")
        with connect(self.db_path) as db:
            profile = db.execute("SELECT * FROM profile_versions WHERE id=?", (profile_version_id,)).fetchone()
            track = db.execute("""SELECT v.*,t.profile_id,t.title,t.status AS track_status
                FROM research_track_versions v JOIN research_tracks t ON t.id=v.track_id WHERE v.id=?""", (track_version_id,)).fetchone()
            if not profile or profile["approval_state"] != "APPROVED" or not track or track["approval_state"] != "APPROVED" or track["profile_id"] != profile["profile_id"] or track["track_status"] != "ACTIVE":
                raise ValueError("An approved profile and active approved research track are required")
            claims = [dict(r) for r in db.execute(f"""SELECT cr.* FROM profile_version_claims pvc
                JOIN claim_revisions cr ON cr.id=pvc.claim_revision_id WHERE pvc.profile_version_id=?
                AND cr.review_status='APPROVED' AND cr.approved_for_{use}=1""", (profile_version_id,))]
        if not claims:
            raise ValueError(f"No {use}-approved claims in profile snapshot")
        claim_map = {c["id"]: c for c in claims}
        if not set(json.loads(track["supporting_claim_revision_ids_json"])) <= claim_map.keys():
            raise ValueError("Research track references claims outside the approved profile/use")
        return dict(profile), dict(track), claim_map

    def create_master_cv(self, profile_version_id: int, sections: list[dict]) -> int:
        with connect(self.db_path) as db:
            profile = db.execute("SELECT * FROM profile_versions WHERE id=?", (profile_version_id,)).fetchone()
            if not profile or profile["approval_state"] != "APPROVED":
                raise ValueError("Approved profile required")
            allowed = {r["claim_revision_id"] for r in db.execute("""SELECT pvc.claim_revision_id FROM profile_version_claims pvc
                JOIN claim_revisions cr ON cr.id=pvc.claim_revision_id WHERE pvc.profile_version_id=?
                AND cr.review_status='APPROVED' AND cr.approved_for_application=1""", (profile_version_id,))}
        if not sections:
            raise ValueError("Master CV needs sections")
        for section in sections:
            if section.get("name") not in SECTIONS or not section.get("bullets"):
                raise ValueError("Invalid or empty CV section")
            for bullet in section["bullets"]:
                if not bullet.get("text", "").strip() or not set(bullet.get("claim_revision_ids", [])) or not set(bullet["claim_revision_ids"]) <= allowed:
                    raise ValueError("Each CV bullet requires approved application claims")
        with transaction(self.db_path) as db:
            version = db.execute("SELECT COALESCE(MAX(version_number),0)+1 FROM master_cv_versions WHERE profile_id=?", (profile["profile_id"],)).fetchone()[0]
            return db.execute("""INSERT INTO master_cv_versions
                (profile_id,profile_version_id,version_number,sections_json,created_at) VALUES(?,?,?,?,?)""",
                (profile["profile_id"], profile_version_id, version, _dump(sections), utc_now())).lastrowid

    def review_master_cv(self, master_id: int, reviewer: str, approve: bool) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        with transaction(self.db_path) as db:
            row = db.execute("SELECT * FROM master_cv_versions WHERE id=?", (master_id,)).fetchone()
            if not row or row["approval_state"] != "DRAFT":
                raise ValueError("Only draft master CV versions can be reviewed")
            db.execute("UPDATE master_cv_versions SET approval_state=?,approved_at=?,approved_by=? WHERE id=?",
                       ("APPROVED" if approve else "REJECTED", utc_now(), reviewer.strip(), master_id))

    def create_module(self, profile_version_id: int, key: str, content: str, claim_ids: list[int]) -> int:
        key = key.strip().upper()
        with connect(self.db_path) as db:
            profile = db.execute("SELECT * FROM profile_versions WHERE id=? AND approval_state='APPROVED'", (profile_version_id,)).fetchone()
            allowed = {r[0] for r in db.execute("""SELECT pvc.claim_revision_id FROM profile_version_claims pvc
                JOIN claim_revisions cr ON cr.id=pvc.claim_revision_id WHERE pvc.profile_version_id=?
                AND cr.review_status='APPROVED' AND cr.approved_for_application=1""", (profile_version_id,))}
        if not profile or not key.strip() or not content.strip() or not claim_ids or not set(claim_ids) <= allowed:
            raise ValueError("Module needs an approved profile, content, and supporting application claims")
        with transaction(self.db_path) as db:
            version = db.execute("SELECT COALESCE(MAX(version_number),0)+1 FROM story_module_versions WHERE profile_id=? AND module_key=?", (profile["profile_id"], key)).fetchone()[0]
            return db.execute("""INSERT INTO story_module_versions
                (profile_id,module_key,version_number,content,claim_revision_ids_json,created_at)
                VALUES(?,?,?,?,?,?)""", (profile["profile_id"], key.strip().upper(), version, content.strip(), _dump(sorted(set(claim_ids))), utc_now())).lastrowid

    def review_module(self, module_id: int, reviewer: str, approve: bool) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        with transaction(self.db_path) as db:
            row = db.execute("SELECT approval_state FROM story_module_versions WHERE id=?", (module_id,)).fetchone()
            if not row or row["approval_state"] != "DRAFT":
                raise ValueError("Only draft modules can be reviewed")
            db.execute("UPDATE story_module_versions SET approval_state=?,approved_at=?,approved_by=? WHERE id=?",
                       ("APPROVED" if approve else "REJECTED", utc_now(), reviewer.strip(), module_id))

    def tailor_cv(self, master_id: int, track_version_id: int, *, application_id: int | None = None,
                  faculty_id: int | None = None, selected_sections: list[str] | None = None) -> int:
        with connect(self.db_path) as db:
            master = db.execute("SELECT * FROM master_cv_versions WHERE id=?", (master_id,)).fetchone()
        if not master or master["approval_state"] != "APPROVED":
            raise ValueError("Approved Master CV required")
        _, _, claims = self._approved_context(master["profile_version_id"], track_version_id)
        sections = json.loads(master["sections_json"])
        order = selected_sections or [s["name"] for s in sections]
        if len(order) != len(set(order)) or not set(order) <= {s["name"] for s in sections}:
            raise ValueError("CV variant may only select approved master sections once")
        chosen = [next(s for s in sections if s["name"] == name) for name in order]
        ids = sorted({i for s in chosen for b in s["bullets"] for i in b["claim_revision_ids"]})
        if not set(ids) <= claims.keys():
            raise ValueError("Master CV claims are no longer in approved profile")
        content = "\n\n".join("# " + s["name"] + "\n" + "\n".join("• " + b["text"] for b in s["bullets"]) for s in chosen)
        diff = {"preserved": order, "removed": [s["name"] for s in sections if s["name"] not in order],
                "reordered": order != [s["name"] for s in sections if s["name"] in order],
                "added_or_rephrased": [], "claim_revision_ids": ids}
        return self._save("CV", content, master["profile_version_id"], track_version_id,
                          application_id=application_id, faculty_id=faculty_id, master_cv_version_id=master_id,
                          claim_ids=ids, diff=diff, template="cv-selection-v1")

    def create_statement(self, kind: str, profile_version_id: int, track_version_id: int,
                         application_id: int, requirement_id: int, module_ids: list[int]) -> int:
        if kind not in STATEMENTS:
            raise ValueError("Statement type must be explicit")
        _, track, claims = self._approved_context(profile_version_id, track_version_id)
        app, requirement, programme, _, evidence = self._context(application_id, requirement_id, kind)
        with connect(self.db_path) as db:
            modules = [dict(r) for r in db.execute(f"SELECT * FROM story_module_versions WHERE id IN ({','.join('?' for _ in module_ids)})", module_ids)] if module_ids else []
        if len(modules) != len(set(module_ids)) or not modules or any(m["approval_state"] != "APPROVED" for m in modules):
            raise ValueError("Approved story modules required")
        with connect(self.db_path) as db:
            profile_id = db.execute("SELECT profile_id FROM profile_versions WHERE id=?", (profile_version_id,)).fetchone()[0]
        if any(m["profile_id"] != profile_id or not set(json.loads(m["claim_revision_ids_json"])) <= claims.keys() for m in modules):
            raise ValueError("Story modules must trace to this approved profile")
        ids = sorted({i for m in modules for i in json.loads(m["claim_revision_ids_json"])})
        title = programme["university"] + " — " + programme["programme_name"] if programme else app["cycle"]
        content = title + "\n\n" + "\n\n".join(m["content"] for m in modules) + "\n\nProposed research direction: " + track["research_problem"]
        return self._save(kind, content, profile_version_id, track_version_id, application_id=application_id,
                          requirement_id=requirement_id, claim_ids=ids, module_ids=module_ids,
                          evidence_ids=evidence, template="reviewed-modules-v1")

    def create_proposal(self, profile_version_id: int, track_version_id: int,
                        *, application_id: int | None = None, faculty_id: int | None = None,
                        requirement_id: int | None = None, publication_ids: list[int] | None = None,
                        format_name: str = "CONCEPT_NOTE") -> int:
        _, track, claims = self._approved_context(profile_version_id, track_version_id)
        if format_name not in {"SHORT_STATEMENT", "CONCEPT_NOTE", "ONE_PAGE", "TWO_PAGE", "THREE_PAGE", "FULL", "INSTITUTION_TEMPLATE"}:
            raise ValueError("Invalid proposal format")
        evidence = []
        if application_id:
            _, _, programme, opportunity, evidence = self._context(application_id, requirement_id, "RESEARCH_PROPOSAL")
        elif not faculty_id:
            raise ValueError("Proposal needs an application or verified faculty context")
        with connect(self.db_path) as db:
            faculty = db.execute("SELECT * FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone() if faculty_id else None
            linked_faculty = db.execute("SELECT 1 FROM application_faculty WHERE application_id=? AND faculty_profile_id=?",
                                        (application_id, faculty_id)).fetchone() if application_id and faculty_id else None
            pubs = [dict(r) for r in db.execute(f"""SELECT p.*,e.verification_state FROM publications p
                JOIN source_evidence e ON e.id=p.source_evidence_id WHERE p.id IN ({','.join('?' for _ in publication_ids)})""", publication_ids)] if publication_ids else []
        if faculty_id and (not faculty or faculty["verification_state"] != "VERIFIED"):
            raise ValueError("Professor must be verified")
        if application_id and faculty_id and (not linked_faculty or
                (programme and faculty["institution"].casefold() != programme["university"].casefold()) or
                (opportunity and faculty["institution"].casefold() != opportunity["institution"].casefold())):
            raise ValueError("Professor must be linked to this application and institution")
        if len(pubs) != len(set(publication_ids or [])) or any(p["verification_state"] != "VERIFIED" or (faculty_id and p["faculty_profile_id"] != faculty_id) for p in pubs):
            raise ValueError("Every cited publication must resolve to verified stored evidence for this professor")
        sections = [("Core research question", track["research_problem"]),
                    ("Motivation", track["motivation"]),
                    ("Methodology", track["proposed_methodology"]),
                    ("Evaluation", track["evaluation_strategy"]),
                    ("Expected contribution", track["expected_contribution"])]
        content = "\n\n".join("# " + h + "\n" + v for h, v in sections if v)
        if faculty:
            content += "\n\n# Professor connection\n" + faculty["name"] + " — " + faculty["institution"]
        if pubs:
            content += "\n\n# Verified related work\n" + "\n".join(f"{p['title']} ({p['year'] or 'year unknown'})." for p in pubs)
        ids = json.loads(track["supporting_claim_revision_ids_json"])
        diff = {"preserved": ["research_problem", "motivation", "proposed_methodology", "expected_contribution"],
                "changed": ["professor_connection", "verified_related_work"] if faculty else [],
                "publication_ids": publication_ids or [], "format": format_name}
        return self._save("RESEARCH_PROPOSAL", content, profile_version_id, track_version_id,
                          application_id=application_id, faculty_id=faculty_id, requirement_id=requirement_id,
                          claim_ids=ids, publication_ids=publication_ids or [],
                          evidence_ids=evidence + [p["source_evidence_id"] for p in pubs], diff=diff,
                          template="track-proposal-v1")

    def create_cover_letter(self, profile_version_id: int, track_version_id: int, application_id: int,
                            requirement_id: int | None = None, *, intentional: bool = False) -> int:
        _, track, claims = self._approved_context(profile_version_id, track_version_id)
        app, requirement, programme, _, evidence = self._context(application_id, requirement_id, "COVER_LETTER")
        if not intentional and (not requirement or requirement["requirement_state"] != "REQUIRED"):
            raise ValueError("Cover letter requires REQUIRED status or explicit operator selection")
        if requirement and requirement["requirement_state"] in {"NOT_REQUESTED", "UNKNOWN"}:
            raise ValueError("Cover letter requirement is unresolved or not requested")
        name = programme["university"] if programme else "application committee"
        factual = [c for c in claims.values() if c["classification"] == "FACT"][:3]
        content = f"Dear {name} admissions committee,\n\n" + "\n\n".join(c["claim_text"] for c in factual)
        content += "\n\nMy proposed research direction is " + track["research_problem"]
        return self._save("COVER_LETTER", content, profile_version_id, track_version_id,
                          application_id=application_id, requirement_id=requirement_id,
                          claim_ids=[c["id"] for c in factual], evidence_ids=evidence,
                          template="cover-letter-v1")

    def _context(self, application_id, requirement_id, kind):
        with connect(self.db_path) as db:
            app = db.execute("SELECT * FROM applications WHERE id=?", (application_id,)).fetchone()
            if not app:
                raise ValueError("Application not found")
            programme = db.execute("SELECT * FROM programmes WHERE id=?", (app["programme_id"],)).fetchone() if app["programme_id"] else None
            opportunity = db.execute("SELECT * FROM opportunities WHERE id=?", (app["opportunity_id"],)).fetchone() if app["opportunity_id"] else None
            requirement = db.execute("SELECT * FROM requirements WHERE id=? AND application_id=?", (requirement_id, application_id)).fetchone() if requirement_id else None
            evidence = db.execute("SELECT * FROM source_evidence WHERE id=?", (requirement["source_evidence_id"],)).fetchone() if requirement else None
        if requirement_id and (not requirement or requirement["normalized_document_type"] != kind or requirement["requirement_state"] not in {"REQUIRED", "OPTIONAL"} or not evidence or evidence["verification_state"] != "VERIFIED"):
            raise ValueError("Matching sourced and reviewed requirement required")
        if kind in STATEMENTS and not requirement:
            raise ValueError("A sourced exact statement requirement is required")
        if opportunity and opportunity["verification_state"] != "VERIFIED":
            raise ValueError("Opportunity needs reviewed source evidence")
        return dict(app), dict(requirement) if requirement else None, dict(programme) if programme else None, dict(opportunity) if opportunity else None, [evidence["id"]] if evidence else []

    def _quality(self, kind, content, application_id, requirement_id, claim_ids, publication_ids, pdf):
        issues = []
        words = len(content.split())
        pages = len(PdfReader(io.BytesIO(pdf)).pages)
        if PLACEHOLDERS.search(content):
            issues.append("PLACEHOLDER")
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if len(paragraphs) != len(set(paragraphs)):
            issues.append("DUPLICATE_PARAGRAPH")
        if application_id:
            with connect(self.db_path) as db:
                app = db.execute("SELECT programme_id,opportunity_id FROM applications WHERE id=?", (application_id,)).fetchone()
                target = db.execute("SELECT university FROM programmes WHERE id=?", (app["programme_id"],)).fetchone() if app["programme_id"] else None
                others = [r[0] for r in db.execute("SELECT DISTINCT university FROM programmes WHERE id!=?", (app["programme_id"] or -1,))]
                req = db.execute("SELECT * FROM requirements WHERE id=?", (requirement_id,)).fetchone() if requirement_id else None
                approved_claim_lines = {r[0].strip().casefold() for r in db.execute(
                    f"SELECT claim_text FROM claim_revisions WHERE id IN ({','.join('?' for _ in claim_ids)})", claim_ids)} if claim_ids else set()
            for university in others:
                if not university or (target and university.casefold() == target["university"].casefold()):
                    continue
                other_lines = [line.strip().lstrip("•- ").strip() for line in content.splitlines()
                               if line.strip().lstrip("•- ").strip().casefold() not in approved_claim_lines]
                if any(re.search(r"\b" + re.escape(university) + r"\b", line, re.I) for line in other_lines):
                    issues.append("OTHER_INSTITUTION:" + university)
            if req:
                if req["word_limit"] and words > req["word_limit"]:
                    issues.append("WORD_LIMIT")
                if req["page_limit"] and pages > req["page_limit"]:
                    issues.append("PAGE_LIMIT")
        warnings = []
        if kind == "CV" and words < 80:
            warnings.append("THIN_CV: review completeness against the full master CV")
        if kind in STATEMENTS and words < 150:
            warnings.append("THIN_STATEMENT: review narrative depth before submission")
        if kind == "RESEARCH_PROPOSAL" and words < 150:
            warnings.append("THIN_PROPOSAL: review methodological detail before use")
        return {"word_count": words, "page_count": pages, "blockers": issues, "warnings": warnings,
                "claim_revision_ids": claim_ids, "publication_ids": publication_ids}

    def _save(self, kind, content, profile_version_id, track_version_id, *, application_id=None,
              faculty_id=None, master_cv_version_id=None, requirement_id=None, claim_ids=None,
              module_ids=None, publication_ids=None, evidence_ids=None, diff=None, template="manual-v1",
              parent_artifact_id=None):
        if kind not in GENERATED or not content.strip():
            raise ValueError("Invalid generated material")
        _, _, approved = self._approved_context(profile_version_id, track_version_id)
        claim_ids = sorted(set(claim_ids or []))
        if not set(claim_ids) <= approved.keys():
            raise ValueError("Rejected or unsupported applicant claim")
        title = kind if kind in {"CV", "SOP"} else kind.replace("_", " ").title()
        pdf = render_pdf(title, content)
        quality = self._quality(kind, content, application_id, requirement_id, claim_ids, publication_ids or [], pdf)
        if parent_artifact_id:
            with connect(self.db_path) as db:
                parent = db.execute("SELECT content_text FROM generated_artifacts WHERE id=?", (parent_artifact_id,)).fetchone()
            if not parent:
                raise ValueError("Parent material version not found")
            approved_paragraphs = {re.sub(r"\s+", " ", p).strip() for p in parent["content_text"].split("\n\n")}
            changed = [p for p in content.split("\n\n") if re.sub(r"\s+", " ", p).strip() not in approved_paragraphs]
            if changed:
                quality["blockers"].append("UNREVIEWED_MANUAL_CONTENT: approve a source module or claim, then regenerate")
        filename = f"{kind.lower()}-{application_id or faculty_id or 'master'}-{utc_now().replace(':','')}.pdf"
        saved = self.vault.upload(pdf, filename, "GENERATED", kind, kind.replace("_", " ").title())
        if saved["status"] == "duplicate":
            raise ValueError("Identical PDF already stored; make a substantive new version")
        with transaction(self.db_path) as db:
            db.execute("""UPDATE document_versions SET generation_context=?,prompt_version=?,model_used=?,
                master_document_version_id=?,claim_ids_json=?,source_evidence_ids_json=?,manual_edits=?,generated_at=?
                WHERE id=?""", (_dump({"application_id": application_id,"faculty_id": faculty_id,
                "profile_version_id": profile_version_id,"research_track_version_id": track_version_id,
                "requirement_id": requirement_id,"publication_ids": publication_ids or [],
                "module_ids": module_ids or []}), template, "manual",
                None, _dump(claim_ids), _dump(evidence_ids or []),
                "" if not parent_artifact_id else "new version", utc_now(), saved["version_id"]))
            return db.execute("""INSERT INTO generated_artifacts
                (document_version_id,parent_artifact_id,kind,application_id,faculty_profile_id,profile_version_id,
                 research_track_version_id,master_cv_version_id,requirement_id,claim_revision_ids_json,
                 story_module_version_ids_json,publication_ids_json,evidence_ids_json,content_text,diff_json,
                 quality_json,template_id,template_version,provider,model,generated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (saved["version_id"],parent_artifact_id,kind,
                application_id,faculty_id,profile_version_id,track_version_id,master_cv_version_id,requirement_id,
                _dump(claim_ids),_dump(module_ids or []),_dump(publication_ids or []),_dump(evidence_ids or []),
                content,_dump(diff or {}),_dump(quality),template,"1","manual","manual",utc_now())).lastrowid

    def revise_artifact(self, artifact_id: int, content: str) -> int:
        with connect(self.db_path) as db:
            old = db.execute("SELECT * FROM generated_artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not old:
            raise ValueError("Artifact not found")
        if old["kind"] == "CV" or old["kind"] == "RESEARCH_PROPOSAL":
            raise ValueError("Revise the approved master CV or track, then regenerate this material")
        # Freeform edits get a draft and must pass checks and an explicit reviewer decision.
        return self._save(old["kind"], content, old["profile_version_id"], old["research_track_version_id"],
            application_id=old["application_id"],faculty_id=old["faculty_profile_id"],
            master_cv_version_id=old["master_cv_version_id"],requirement_id=old["requirement_id"],
            claim_ids=json.loads(old["claim_revision_ids_json"]),module_ids=json.loads(old["story_module_version_ids_json"]),
            publication_ids=json.loads(old["publication_ids_json"]),evidence_ids=json.loads(old["evidence_ids_json"]),
            diff={"previous_artifact_id": artifact_id,"manual_edit": True},template="manual-revision-v1",
            parent_artifact_id=artifact_id)

    def review_artifact(self, artifact_id: int, reviewer: str, approve: bool) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        with connect(self.db_path) as db:
            row = db.execute("SELECT * FROM generated_artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not row or row["approval_state"] != "DRAFT":
            raise ValueError("Only draft artifacts can be reviewed")
        quality = json.loads(row["quality_json"])
        if approve and quality["blockers"]:
            raise ValueError("Resolve document quality blockers first: " + ", ".join(quality["blockers"]))
        if approve:
            self._approved_context(row["profile_version_id"], row["research_track_version_id"])
            self.vault.set_approval(row["document_version_id"], True, reviewer)
        with transaction(self.db_path) as db:
            db.execute("UPDATE generated_artifacts SET approval_state=?,approved_at=?,approved_by=? WHERE id=?",
                       ("APPROVED" if approve else "REJECTED",utc_now(),reviewer.strip(),artifact_id))
