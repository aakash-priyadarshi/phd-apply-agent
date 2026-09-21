"""Deterministic document selection, local exports, and reusable preflight."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import date
from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.discovery import freshness
from phd_agent.documents import DocumentVault
from phd_agent.ledger import Ledger, REQUIREMENT_CONTEXTS
from phd_agent.materials import MaterialStudio
from phd_agent.matching import contact_policy_state


RULE_VERSION = "application-outreach-preflight-v1"


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")[:80] or "Application"


def package_policy(requirement: dict, *, context: str, selected_optional: bool = False) -> dict:
    """Pure inclusion rule. Unknown or conflicting evidence always asks for review."""
    if context not in REQUIREMENT_CONTEXTS:
        raise ValueError("Invalid package context")
    state = requirement["requirement_state"]
    document_state = requirement["document_state"]
    if requirement.get("context") != context:
        return {"decision": "EXCLUDE", "reason": "Different application context"}
    if state == "UNKNOWN" or requirement.get("conflict"):
        return {"decision": "REVIEW", "reason": "Requirement is unknown or conflicting"}
    if state == "NOT_REQUESTED":
        return {"decision": "EXCLUDE", "reason": "Official requirement says not requested"}
    if not requirement.get("normalized_document_type"):
        return {"decision": "EXCLUDE", "reason": "Non-file requirement is checked by preflight"}
    if state == "OPTIONAL" and not selected_optional:
        return {"decision": "EXCLUDE", "reason": "Optional item not selected by operator"}
    if requirement.get("sensitivity") == "HIGHLY_SENSITIVE" and context == "FACULTY_OUTREACH":
        return {"decision": "REVIEW", "reason": "Sensitive identity file needs explicit outreach review"}
    if document_state != "AVAILABLE":
        return {"decision": "REVIEW", "reason": f"Document state is {document_state}"}
    return {"decision": "INCLUDE", "reason": "Required item approved" if state == "REQUIRED" else "Optional item selected and approved"}


def combine_selected_pdfs(parts: list[tuple[str, bytes]]) -> bytes:
    if not parts:
        raise ValueError("Select PDF components")
    writer = PdfWriter()
    for filename, data in parts:
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Combined file may include PDFs only")
        reader = PdfReader(io.BytesIO(data))
        if not reader.pages:
            raise ValueError("Empty PDF component")
        for page in reader.pages:
            writer.add_page(page)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class PackageBuilder:
    def __init__(self, db_path: Path | str, vault: DocumentVault | None = None):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.vault = vault or DocumentVault(self.db_path)
        self.ledger = Ledger(self.db_path)
        self.materials = MaterialStudio(self.db_path, self.vault)

    def preview(self, application_id: int, context: str = "FORMAL_APPLICATION",
                selected_optional: set[int] | None = None) -> dict:
        selected_optional = selected_optional or set()
        rows = self.ledger.requirement_rows(application_id)
        with connect(self.db_path) as db:
            for row in rows:
                if row["document_version_id"]:
                    version = self.vault.get_version(row["document_version_id"])
                    row["sensitivity"] = version["sensitivity"]
                else:
                    row["sensitivity"] = None
        decisions = []
        for row in rows:
            rule = package_policy(row, context=context, selected_optional=row["id"] in selected_optional)
            decisions.append({"requirement_id": row["id"], "label": row["original_label"],
                              "state": row["requirement_state"], "document_state": row["document_state"],
                              "document_version_id": row["document_version_id"], **rule})
        return {"decisions": decisions, "completeness": self.ledger.readiness(application_id, context),
                "awaiting_approval": sum(r["document_state"] == "NEEDS_APPROVAL" for r in rows if r["context"] == context)}

    def build(self, application_id: int, profile_version_id: int, track_version_id: int,
              *, context: str = "FORMAL_APPLICATION", selected_optional: set[int] | None = None,
              combine_pdf: bool = False, zip_export: bool = False) -> int:
        self.materials._approved_context(profile_version_id, track_version_id,
                                         "outreach" if context == "FACULTY_OUTREACH" else "application")
        app = self.ledger.get("applications", application_id)
        if not app:
            raise ValueError("Application not found")
        if context not in REQUIREMENT_CONTEXTS:
            raise ValueError("Invalid context")
        rows = self.ledger.requirement_rows(application_id)
        scoped = [r for r in rows if r["context"] == context]
        if not scoped:
            raise ValueError("No sourced requirements for this context")
        preview = self.preview(application_id, context, selected_optional)
        decisions = preview["decisions"]
        document_requirements = {r["id"] for r in scoped if r["normalized_document_type"]}
        blockers = [d for d in decisions if d["requirement_id"] in document_requirements and d["state"] == "REQUIRED" and d["decision"] != "INCLUDE"]
        if blockers:
            raise ValueError("Required documents missing, unapproved, or unresolved: " + ", ".join(d["label"] for d in blockers))
        included = [(r, next(d for d in decisions if d["requirement_id"] == r["id"])) for r in scoped
                    if next(d for d in decisions if d["requirement_id"] == r["id"])["decision"] == "INCLUDE"]
        if not included:
            raise ValueError("Package has no approved selected documents")
        with connect(self.db_path) as db:
            version = db.execute("SELECT COALESCE(MAX(version_number),0)+1 FROM application_packages WHERE application_id=?", (application_id,)).fetchone()[0]
            programme = db.execute("SELECT * FROM programmes WHERE id=?", (app["programme_id"],)).fetchone() if app["programme_id"] else None
            opportunity = db.execute("SELECT * FROM opportunities WHERE id=?", (app["opportunity_id"],)).fetchone() if app["opportunity_id"] else None
            evidence_ids = sorted({r["source_evidence_id"] for r in scoped})
            evidence = [dict(r) for r in db.execute(f"SELECT * FROM source_evidence WHERE id IN ({','.join('?' for _ in evidence_ids)})", evidence_ids)]
        name = (programme["university"] + "_" + programme["programme_name"] if programme else opportunity["institution"] + "_" + opportunity["title"]) + "_" + app["cycle"]
        folder = self.db_path.parent / "exports" / _slug(name) / f"package-v{version}"
        manifest_docs = []
        parts = []
        prepared = []
        for index, (row, decision) in enumerate(included, 1):
            doc = self.vault.get_version(row["document_version_id"])
            if not doc or doc["approval_state"] != "APPROVED" or not self.vault.storage.verify_hash(doc["storage_key"], doc["sha256"]):
                raise ValueError("Selected document changed before export")
            if doc["document_class"] == "GENERATED":
                with connect(self.db_path) as db:
                    artifact = db.execute("SELECT approval_state,profile_version_id,research_track_version_id FROM generated_artifacts WHERE document_version_id=?", (doc["id"],)).fetchone()
                if not artifact or artifact["approval_state"] != "APPROVED" or artifact["profile_version_id"] != profile_version_id or artifact["research_track_version_id"] != track_version_id:
                    raise ValueError("Generated document is unapproved or from a different applicant context")
            suffix = Path(doc["canonical_filename"]).suffix.lower()
            filename = f"{index:02d}_{_slug(row['normalized_document_type'] or row['original_label'])}_v{doc['version_number']}{suffix}"
            data = self.vault.storage.get(doc["storage_key"])
            prepared.append((filename, data))
            manifest_docs.append({"document_id": doc["document_id"], "document_version_id": doc["id"],
                "requirement_id": row["id"], "filename": filename, "sha256": doc["sha256"],
                "reason": decision["reason"], "order": index})
            if suffix == ".pdf":
                parts.append((filename, data))
        combined = None
        combined_data = None
        if combine_pdf:
            if len(parts) != len(manifest_docs):
                raise ValueError("A combined PDF requires every selected component to be PDF")
            combined_data = combine_selected_pdfs(parts)
            combined = {"filename": "combined_selected_documents.pdf", "sha256": hashlib.sha256(combined_data).hexdigest(),
                        "components": [{"filename": x["filename"], "sha256": x["sha256"]} for x in manifest_docs]}
        manifest = {"application_id": application_id, "version": version, "context": context,
                    "profile_version_id": profile_version_id, "research_track_version_id": track_version_id,
                    "documents": manifest_docs, "combined_pdf": combined,
                    "requirements": [{"id": r["id"], "state": r["requirement_state"], "label": r["original_label"]} for r in scoped],
                    "evidence_ids": evidence_ids}
        manifest_bytes = _dump(manifest).encode("utf-8")
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        folder.mkdir(parents=True, exist_ok=False)
        for filename, data in prepared:
            (folder / filename).write_bytes(data)
        if combined_data is not None:
            (folder / "combined_selected_documents.pdf").write_bytes(combined_data)
        (folder / "manifest.json").write_bytes(manifest_bytes)
        checklist = [f"Application package v{version} — {name}", "", "Selected files:"]
        checklist += [f"- {d['filename']} (requirement #{d['requirement_id']}; SHA-256 {d['sha256']})" for d in manifest_docs]
        checklist += ["", "Review required:"] + [f"- {d['label']}: {d['reason']}" for d in decisions if d["decision"] == "REVIEW"]
        (folder / "checklist.txt").write_text("\n".join(checklist), encoding="utf-8")
        if zip_export:
            with zipfile.ZipFile(folder / "package.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(folder.iterdir()):
                    if path.name != "package.zip":
                        archive.write(path, path.name)
        with transaction(self.db_path) as db:
            package_id = db.execute("""INSERT INTO application_packages
                (application_id,version_number,context,profile_version_id,research_track_version_id,
                 requirements_json,evidence_json,decisions_json,manifest_json,package_sha256,export_path,built_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (application_id,version,context,profile_version_id,
                track_version_id,_dump(scoped),_dump(evidence),_dump(decisions),_dump(manifest),digest,
                str(folder.resolve()),utc_now())).lastrowid
            for doc in manifest_docs:
                db.execute("""INSERT INTO package_documents
                    (package_id,document_id,document_version_id,requirement_id,canonical_filename,sha256,inclusion_reason,sort_order)
                    VALUES(?,?,?,?,?,?,?,?)""", (package_id,doc["document_id"],doc["document_version_id"],
                    doc["requirement_id"],doc["filename"],doc["sha256"],doc["reason"],doc["order"]))
        return package_id

    def preflight(self, package_id: int) -> dict:
        with connect(self.db_path) as db:
            package = db.execute("SELECT * FROM application_packages WHERE id=?", (package_id,)).fetchone()
            if not package:
                raise ValueError("Package not found")
            documents = [dict(r) for r in db.execute("SELECT * FROM package_documents WHERE package_id=? ORDER BY sort_order", (package_id,))]
            app = db.execute("SELECT * FROM applications WHERE id=?", (package["application_id"],)).fetchone()
            programme = db.execute("SELECT * FROM programmes WHERE id=?", (app["programme_id"],)).fetchone() if app["programme_id"] else None
            opportunity = db.execute("SELECT * FROM opportunities WHERE id=?", (app["opportunity_id"],)).fetchone() if app["opportunity_id"] else None
            deadlines = [dict(r) for r in db.execute("SELECT * FROM deadlines WHERE application_id=? AND deadline_type='APPLICATION'", (app["id"],))]
            referees = [dict(r) for r in db.execute("SELECT * FROM application_referees WHERE application_id=?", (app["id"],))]
            current_reqs = [dict(r) for r in db.execute("SELECT * FROM requirements WHERE application_id=? AND context=? ORDER BY context,id", (app["id"],package["context"]))]
        rules = []
        def add(rule_id, severity, message, affected=None, action=""):
            rules.append({"rule_id": rule_id, "severity": severity, "message": message,
                          "affected": affected, "action": action})
        formal = package["context"] == "FORMAL_APPLICATION"
        if formal:
            current_deadline = next((d for d in deadlines if d["due_at"][:10] >= date.today().isoformat() and d["verification_state"] == "VERIFIED"), None)
            add("DEADLINE", "PASS" if current_deadline else "BLOCK", "Current verified application deadline" if current_deadline else "Application deadline absent, expired, or unverified", None, "Verify official deadline")
            route = app["portal_url"] or (opportunity["application_route"] if opportunity else None) or (programme["portal_url"] if programme else None)
            add("ROUTE", "PASS" if route else "BLOCK", "Application route recorded" if route else "Application route unknown", None, "Verify route from official source")
            add("ELIGIBILITY", "PASS" if app["eligibility_state"] == "ELIGIBLE" else "BLOCK", "Eligibility confirmed" if app["eligibility_state"] == "ELIGIBLE" else "Eligibility unresolved or ineligible", None, "Review eligibility")
        if opportunity and (formal or opportunity["opportunity_type"] == "ADVERTISED_POSITION"):
            good = opportunity["verification_state"] == "VERIFIED" and opportunity["opening_status"] == "OPEN"
            add("OPENING", "PASS" if good else "BLOCK", "Verified open opportunity" if good else "Opening not verified as open", opportunity["id"], "Reverify opening")
        scoped = [r for r in json.loads(package["requirements_json"]) if r["context"] == package["context"]]
        current_rows = {r["id"]: r for r in self.ledger.requirement_rows(app["id"])}
        snap = [{k:v for k,v in r.items() if k not in {"source_url","mapping_id","document_version_id","mapping_notes","version_number","sha256","storage_key","approval_state","document_id","document_title","document_type","expiry_date","document_state","sensitivity"}} for r in scoped]
        # Compare mutable requirement fields with frozen source rows.
        changed = len(current_reqs) != len(snap) or any(any(r.get(k) != old.get(k) for k in r) for r,old in zip(current_reqs,snap))
        add("REQUIREMENT_SNAPSHOT", "BLOCK" if changed else "PASS", "Requirements changed since package build" if changed else "Requirements match frozen snapshot", None, "Rebuild package" if changed else "")
        mapped = {d["requirement_id"] for d in documents}
        for r in scoped:
            if r["requirement_state"] == "UNKNOWN":
                add("UNKNOWN_REQUIREMENT", "BLOCK", f"Unknown requirement: {r['original_label']}", r["id"], "Resolve from official source")
            if r["requirement_state"] == "REQUIRED" and r["normalized_document_type"]:
                add("REQUIRED_DOCUMENT", "PASS" if r["id"] in mapped else "BLOCK", f"Required item: {r['original_label']}", r["id"], "Link approved document")
            if r["requirement_state"] == "REQUIRED" and not r["normalized_document_type"] and "referee" not in r["original_label"].casefold() and "recommendation" not in r["original_label"].casefold():
                add("ADMIN_REQUIREMENT", "PASS" if r["fulfilled_at"] else "BLOCK", f"Administrative item: {r['original_label']}", r["id"], "Complete the required task")
            if r["id"] in mapped:
                current_state = current_rows.get(r["id"], {}).get("document_state")
                add("CURRENT_DOCUMENT_STATE", "PASS" if current_state == "AVAILABLE" else "BLOCK",
                    f"Current document state: {current_state or 'UNKNOWN'}", r["id"], "Refresh or relink approved document")
        export = Path(package["export_path"])
        manifest = json.loads(package["manifest_json"])
        actual_manifest = _dump(manifest).encode()
        good_manifest = (export / "manifest.json").is_file() and (export / "manifest.json").read_bytes() == actual_manifest and hashlib.sha256(actual_manifest).hexdigest() == package["package_sha256"]
        good_manifest &= len(documents) == len(manifest["documents"])
        add("MANIFEST", "PASS" if good_manifest else "BLOCK", "Manifest and package hash verified" if good_manifest else "Manifest incomplete or changed", None, "Rebuild package")
        if manifest.get("combined_pdf"):
            combined = manifest["combined_pdf"]
            file = export / combined["filename"]
            ordered = [{"filename": d["canonical_filename"], "sha256": d["sha256"]} for d in documents]
            good = file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest() == combined["sha256"] and combined["components"] == ordered
            add("COMBINED_PDF", "PASS" if good else "BLOCK", "Combined PDF and component order verified" if good else "Combined PDF or component order changed", None, "Rebuild package")
        with connect(self.db_path) as db:
            profile = db.execute("SELECT approval_state FROM profile_versions WHERE id=?", (package["profile_version_id"],)).fetchone()
            track = db.execute("SELECT approval_state FROM research_track_versions WHERE id=?", (package["research_track_version_id"],)).fetchone()
        approved_context = bool(profile and profile["approval_state"] == "APPROVED" and track and track["approval_state"] == "APPROVED")
        add("APPLICANT_CONTEXT", "PASS" if approved_context else "BLOCK", "Applicant context approved" if approved_context else "Applicant context unapproved", None, "Review profile and track")
        with connect(self.db_path) as db:
            for d in documents:
                version = self.vault.get_version(d["document_version_id"])
                file = export / d["canonical_filename"]
                good_hash = bool(version and version["sha256"] == d["sha256"] and file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest() == d["sha256"] and self.vault.storage.verify_hash(version["storage_key"], d["sha256"]))
                add("DOCUMENT_HASH", "PASS" if good_hash else "BLOCK", d["canonical_filename"] + (" hash verified" if good_hash else " changed or missing"), d["document_version_id"], "Rebuild package")
                approved = bool(version and version["approval_state"] == "APPROVED")
                add("DOCUMENT_APPROVAL", "PASS" if approved else "BLOCK", "Document approved" if approved else "Document unapproved", d["document_version_id"], "Approve reviewed version")
                if version and "DEMO_ONLY" in (version["notes"] or ""):
                    add("DEMO_DOCUMENT", "BLOCK", "Synthetic demonstration document cannot be submitted", d["document_version_id"], "Replace with the applicant's genuine document")
                if version and "PROVISIONAL_TRANSCRIPT" in (version["notes"] or ""):
                    add("TRANSCRIPT_ACCEPTABILITY", "BLOCK", "Assessment-results printout needs transcript acceptance review", d["document_version_id"], "Obtain an official or accepted unofficial transcript")
                req = next((r for r in scoped if r["id"] == d["requirement_id"]), None)
                if req and version:
                    suffix = Path(d["canonical_filename"]).suffix.lower().lstrip(".")
                    accepted = not req["file_format"] or suffix in {x.strip().lower().lstrip(".") for x in re.split(r"[,;/ ]+", req["file_format"])}
                    add("FILE_FORMAT", "PASS" if accepted else "BLOCK", "File type accepted" if accepted else "File type not accepted", req["id"], "Export accepted type")
                    if req["filename_rule"]:
                        try:
                            valid_name = re.fullmatch(req["filename_rule"], d["canonical_filename"]) is not None
                        except re.error:
                            add("FILENAME", "BLOCK", "Filename rule is not a valid pattern", req["id"],
                                "Correct the filename pattern from the official source")
                        else:
                            add("FILENAME", "PASS" if valid_name else "BLOCK",
                                "Filename rule met" if valid_name else "Filename rule unmet", req["id"], "Use required filename")
                    if req["page_limit"] and suffix == "pdf" and file.is_file():
                        page_count = len(PdfReader(io.BytesIO(file.read_bytes())).pages)
                        add("PAGE_LIMIT", "PASS" if page_count <= req["page_limit"] else "BLOCK", f"{page_count} pages; limit {req['page_limit']}", req["id"], "Shorten document")
                    if req["word_limit"] and suffix == "pdf" and file.is_file():
                        reader = PdfReader(io.BytesIO(file.read_bytes()))
                        count = len(" ".join(page.extract_text() or "" for page in reader.pages).split())
                        add("WORD_LIMIT", "PASS" if count <= req["word_limit"] else "BLOCK", f"{count} PDF words; limit {req['word_limit']}", req["id"], "Shorten document")
                artifact = db.execute("SELECT * FROM generated_artifacts WHERE document_version_id=?", (d["document_version_id"],)).fetchone()
                if version and version["document_class"] == "GENERATED":
                    valid = bool(artifact and artifact["approval_state"] == "APPROVED" and artifact["profile_version_id"] == package["profile_version_id"] and artifact["research_track_version_id"] == package["research_track_version_id"])
                    add("GENERATED_LINEAGE", "PASS" if valid else "BLOCK", "Generated material lineage approved" if valid else "Generated material lineage invalid", d["document_version_id"], "Regenerate from approved context")
                    if artifact:
                        quality = json.loads(artifact["quality_json"])
                        add("CONTENT_QUALITY", "PASS" if not quality["blockers"] else "BLOCK", "Content quality checked" if not quality["blockers"] else ", ".join(quality["blockers"]), d["document_version_id"], "Revise source and regenerate")
                        for warning in quality.get("warnings", []):
                            add("CONTENT_WARNING", "WARNING", warning, d["document_version_id"], "Review content depth")
                        for pub_id in json.loads(artifact["publication_ids_json"]):
                            pub = db.execute("""SELECT e.verification_state FROM publications p JOIN source_evidence e ON e.id=p.source_evidence_id WHERE p.id=?""", (pub_id,)).fetchone()
                            add("CITATION", "PASS" if pub and pub[0] == "VERIFIED" else "BLOCK", "Stored citation verified" if pub and pub[0] == "VERIFIED" else "Unsupported citation", pub_id, "Use verified publication")
                        for claim_id in json.loads(artifact["claim_revision_ids_json"]):
                            claim = db.execute("SELECT review_status,approved_for_application FROM claim_revisions WHERE id=?", (claim_id,)).fetchone()
                            good = bool(claim and claim["review_status"] == "APPROVED" and claim["approved_for_application"])
                            add("CLAIM", "PASS" if good else "BLOCK", "Claim approved" if good else "Claim unapproved", claim_id, "Review claim")
        for e in json.loads(package["evidence_json"]):
            kind = "PROGRAMME_REQUIREMENT"
            state = freshness(e["retrieved_at"], kind)
            verified = e["verification_state"] == "VERIFIED"
            add("EVIDENCE_FRESHNESS", "PASS" if state == "CURRENT" and verified else "BLOCK", "Requirement evidence current" if state == "CURRENT" and verified else "Requirement evidence stale or unverified", e["id"], "Reverify source")
        for d in (deadlines if formal else []):
            with connect(self.db_path) as db:
                e = db.execute("SELECT * FROM source_evidence WHERE id=?", (d["source_evidence_id"],)).fetchone()
            current = bool(e and e["verification_state"] == "VERIFIED" and freshness(e["retrieved_at"], "DEADLINE") == "CURRENT")
            add("DEADLINE_EVIDENCE", "PASS" if current else "BLOCK", "Deadline evidence current" if current else "Deadline evidence stale or unverified", d["source_evidence_id"], "Reverify deadline")
        for r in (referees if formal else []):
            add("REFEREE", "PASS" if r["submission_state"] == "SUBMITTED" else "BLOCK", f"Referee {r['referee_name']}: {r['submission_state']}", r["id"], "Confirm referee submission")
        referee_requirements = [r for r in scoped if r["requirement_state"] == "REQUIRED" and
                                ("referee" in r["original_label"].casefold() or "recommendation" in r["original_label"].casefold())]
        if formal and referee_requirements:
            counts = [int(m.group()) for r in referee_requirements if (m := re.search(r"\d+", r["original_label"]))]
            if not counts:
                add("REFEREE_COUNT_UNKNOWN", "BLOCK", "Required referee count is unresolved", None,
                    "Record the numeric referee requirement from the official source")
            else:
                needed = max(counts)
                submitted = sum(r["submission_state"] == "SUBMITTED" for r in referees)
                add("REFEREE_COUNT", "PASS" if submitted >= needed else "BLOCK",
                    f"Submitted referees: {submitted}/{needed}", None, "Confirm required recommendations")
        elif formal and not referees:
            add("REFEREE_SCOPE", "WARNING", "No referee records; confirm whether required", None, "Review official referee requirement")
        if package["context"] == "FACULTY_OUTREACH":
            with connect(self.db_path) as db:
                linked = [dict(r) for r in db.execute("""SELECT f.*,e.retrieved_at FROM application_faculty af
                    JOIN faculty_profiles f ON f.id=af.faculty_profile_id
                    LEFT JOIN faculty_evidence_links l ON l.faculty_profile_id=f.id AND l.fact_type='AFFILIATION'
                    LEFT JOIN source_evidence e ON e.id=l.source_evidence_id
                    WHERE af.application_id=?""", (app["id"],))]
                research_sources = [dict(r) for r in db.execute("""SELECT e.id,e.retrieved_at,e.verification_state
                    FROM application_faculty af JOIN faculty_evidence_links l ON l.faculty_profile_id=af.faculty_profile_id
                    JOIN source_evidence e ON e.id=l.source_evidence_id
                    WHERE af.application_id=? AND l.fact_type='TOPICS'""", (app["id"],))]
            current = any(f["verification_state"] == "VERIFIED" and f["affiliation_state"] == "CURRENT"
                          and freshness(f["retrieved_at"], "FACULTY_AFFILIATION") == "CURRENT" for f in linked)
            add("OUTREACH_FACULTY", "PASS" if current else "BLOCK", "Current professor affiliation verified" if current else "Professor affiliation unknown or stale", None, "Reverify shortlisted professor")
            research_current = any(e["verification_state"] == "VERIFIED" and freshness(e["retrieved_at"], "PUBLICATION") == "CURRENT" for e in research_sources)
            add("OUTREACH_RESEARCH", "PASS" if research_current else "BLOCK",
                "Current faculty research evidence" if research_current else "Faculty research evidence missing or stale",
                [e["id"] for e in research_sources], "Refresh official faculty research evidence")
            contact = bool(linked and all(f["email_state"] == "VERIFIED" for f in linked))
            add("OUTREACH_CONTACT", "PASS" if contact else "BLOCK", "Faculty email verified" if contact else "Faculty email unverified", None, "Verify email from an official source")
            policy = contact_policy_state(opportunity["contact_policy"] if opportunity else None)
            with connect(self.db_path) as db:
                policy_source = db.execute("SELECT * FROM source_evidence WHERE id=?", (opportunity["source_evidence_id"],)).fetchone() if opportunity and opportunity["source_evidence_id"] else None
            policy_current = bool(policy_source and policy_source["verification_state"] == "VERIFIED" and freshness(policy_source["retrieved_at"], "CONTACT_POLICY") == "CURRENT")
            add("OUTREACH_POLICY", "PASS" if policy == "PASS" and policy_current else "BLOCK",
                "Current source explicitly allows contact" if policy == "PASS" and policy_current else "Contact policy is prohibited, unknown, or stale",
                opportunity["source_evidence_id"] if opportunity else None, "Record current official contact policy as ALLOWED or CONTACT_ALLOWED, or do not contact")
        status = "BLOCK" if any(r["severity"] == "BLOCK" for r in rules) else "WARNING" if any(r["severity"] == "WARNING" for r in rules) else "PASS"
        result = {"rule_version": RULE_VERSION, "status": status, "rules": rules,
                  "completeness": self.ledger.readiness(app["id"], package["context"]),
                  "awaiting_approval": sum(r["document_state"] == "NEEDS_APPROVAL" for r in self.ledger.requirement_rows(app["id"]) if r["context"] == package["context"])}
        with transaction(self.db_path) as db:
            run_id = db.execute("INSERT INTO preflight_runs(package_id,rule_version,result_json,status,run_at) VALUES(?,?,?,?,?)",
                                (package_id,RULE_VERSION,_dump(result),status,utc_now())).lastrowid
            db.execute("UPDATE application_packages SET preflight_run_id=?,status=CASE WHEN ?='BLOCK' THEN 'DRAFT' ELSE status END WHERE id=?", (run_id,status,package_id))
            if status == "BLOCK" and app["status"] == "READY_TO_SUBMIT":
                db.execute("UPDATE applications SET status='IN_PROGRESS',updated_at=? WHERE id=?", (utc_now(),app["id"]))
        return {"id": run_id, **result}

    def mark_ready(self, package_id: int, reviewer: str) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        result = self.preflight(package_id)
        if result["status"] == "BLOCK":
            raise ValueError("Preflight has blocking rules; package cannot be READY")
        with transaction(self.db_path) as db:
            db.execute("UPDATE application_packages SET status='READY',approved_at=?,approved_by=? WHERE id=?",
                       (utc_now(),reviewer.strip(),package_id))
