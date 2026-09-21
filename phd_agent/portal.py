"""Approved portal answer library, supervised checklists, and submission archives."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.documents import DocumentVault
from phd_agent.ledger import Ledger


ANSWER_KEYS = (
    "FULL_NAME", "EMAIL", "PHONE", "NATIONALITY", "EDUCATION", "DEGREE_DATES",
    "RESEARCH_INTERESTS", "AWARDS", "PUBLICATIONS_SUMMARY", "REFEREES",
    "ENGLISH_TEST", "FUNDING_STATEMENT", "OTHER",
)
PAYMENT_STATES = ("NOT_APPLICABLE", "PENDING_USER", "RECORDED")


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class PortalAssistance:
    """Copy-ready answers and submission records. Never submits or pays on a portal."""

    def __init__(self, db_path: Path | str, vault: DocumentVault | None = None):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.ledger = Ledger(self.db_path)
        self.vault = vault or DocumentVault(self.db_path)

    def save_answer(self, field_key: str, label: str, value_text: str, *,
                    claim_revision_id: int | None = None, source_evidence_id: int | None = None) -> int:
        if field_key not in ANSWER_KEYS:
            raise ValueError("Unknown answer field")
        if not label.strip() or not value_text.strip():
            raise ValueError("Answer label and value are required")
        with transaction(self.db_path) as db:
            version = db.execute("SELECT COALESCE(MAX(version_number),0)+1 FROM answer_library WHERE field_key=?",
                                 (field_key,)).fetchone()[0]
            return db.execute("""INSERT INTO answer_library
                (field_key,label,value_text,version_number,claim_revision_id,source_evidence_id,
                 approval_state,created_at)
                VALUES(?,?,?,?,?,?, 'DRAFT', ?)""",
                (field_key, label.strip(), value_text.strip(), version, claim_revision_id,
                 source_evidence_id, utc_now())).lastrowid

    def approve_answer(self, answer_id: int, reviewer: str) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        with transaction(self.db_path) as db:
            row = db.execute("SELECT * FROM answer_library WHERE id=?", (answer_id,)).fetchone()
            if not row or row["approval_state"] != "DRAFT":
                raise ValueError("Only a draft answer can be approved")
            db.execute("""UPDATE answer_library SET approval_state='APPROVED',approved_by=?,approved_at=?
                WHERE id=?""", (reviewer.strip(), utc_now(), answer_id))

    def list_answers(self) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM answer_library ORDER BY field_key, version_number DESC")]

    def add_checklist_field(self, application_id: int, field_key: str, portal_label: str, *,
                            required: bool = True) -> int:
        if field_key not in ANSWER_KEYS:
            raise ValueError("Unknown answer field")
        if not portal_label.strip():
            raise ValueError("Portal field label is required")
        if not self.ledger.get("applications", application_id):
            raise ValueError("Application not found")
        with transaction(self.db_path) as db:
            return db.execute("""INSERT INTO portal_checklist_fields
                (application_id,field_key,portal_label,required,status)
                VALUES(?,?,?,?,'EMPTY')""",
                (application_id, field_key, portal_label.strip(), int(required))).lastrowid

    def fill_field(self, checklist_id: int, answer_id: int) -> None:
        with connect(self.db_path) as db:
            field = db.execute("SELECT * FROM portal_checklist_fields WHERE id=?", (checklist_id,)).fetchone()
            answer = db.execute("SELECT * FROM answer_library WHERE id=?", (answer_id,)).fetchone()
        if not field:
            raise ValueError("Checklist field not found")
        if not answer or answer["approval_state"] != "APPROVED":
            raise ValueError("Fill checklist fields from an approved answer")
        if answer["field_key"] != field["field_key"]:
            raise ValueError("Answer field does not match the portal field")
        sources = []
        if answer["claim_revision_id"]:
            sources.append({"type": "claim_revision", "id": answer["claim_revision_id"]})
        if answer["source_evidence_id"]:
            sources.append({"type": "source_evidence", "id": answer["source_evidence_id"]})
        with transaction(self.db_path) as db:
            db.execute("""UPDATE portal_checklist_fields SET answer_id=?,value_snapshot=?,source_json=?,
                status='FILLED',reviewed_by=NULL,reviewed_at=NULL WHERE id=?""",
                (answer_id, answer["value_text"], _dump(sources), checklist_id))

    def review_field(self, checklist_id: int, reviewer: str, *, exclude: bool = False) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        with transaction(self.db_path) as db:
            field = db.execute("SELECT * FROM portal_checklist_fields WHERE id=?", (checklist_id,)).fetchone()
            if not field:
                raise ValueError("Checklist field not found")
            if exclude:
                db.execute("""UPDATE portal_checklist_fields SET status='EXCLUDED',reviewed_by=?,reviewed_at=?
                    WHERE id=?""", (reviewer.strip(), utc_now(), checklist_id))
                return
            if field["status"] != "FILLED":
                raise ValueError("Review a filled portal field before marking it reviewed")
            db.execute("""UPDATE portal_checklist_fields SET status='REVIEWED',reviewed_by=?,reviewed_at=?
                WHERE id=?""", (reviewer.strip(), utc_now(), checklist_id))

    def checklist(self, application_id: int) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM portal_checklist_fields WHERE application_id=? ORDER BY id",
                (application_id,))]

    def copy_ready(self, application_id: int) -> list[dict]:
        rows = []
        for field in self.checklist(application_id):
            rows.append({
                "portal_label": field["portal_label"],
                "field_key": field["field_key"],
                "value": field["value_snapshot"] or "",
                "sources": json.loads(field["source_json"] or "[]"),
                "status": field["status"],
                "required": bool(field["required"]),
            })
        return rows

    def counts(self) -> dict:
        with connect(self.db_path) as db:
            incomplete = db.execute("""SELECT COUNT(*) FROM portal_checklist_fields
                WHERE required=1 AND status NOT IN ('REVIEWED','EXCLUDED')""").fetchone()[0]
            archives = db.execute("SELECT COUNT(*) FROM submission_archives").fetchone()[0]
        return {"incomplete_portal_fields": incomplete, "submission_archives": archives}

    def record_submission(self, application_id: int, package_id: int, confirmation_number: str,
                          submitted_at: str, actor: str, *, payment_state: str = "NOT_APPLICABLE",
                          payment_reference: str | None = None, notes: str = "",
                          acknowledge_empty_checklist: bool = False) -> int:
        if not actor.strip() or not confirmation_number.strip():
            raise ValueError("Operator and portal confirmation number are required")
        if payment_state not in PAYMENT_STATES:
            raise ValueError("Invalid payment state")
        if payment_state == "RECORDED" and not (payment_reference or "").strip():
            raise ValueError("A user-recorded payment needs a payment reference")
        try:
            datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("Submission time must be an ISO date or date/time") from error
        with connect(self.db_path) as db:
            package = db.execute("SELECT * FROM application_packages WHERE id=?", (package_id,)).fetchone()
            existing = db.execute("SELECT id FROM submission_archives WHERE application_id=?",
                                  (application_id,)).fetchone()
            documents = [dict(r) for r in db.execute(
                "SELECT * FROM package_documents WHERE package_id=? ORDER BY sort_order", (package_id,))]
            referees = [dict(r) for r in db.execute(
                "SELECT * FROM application_referees WHERE application_id=?", (application_id,))]
        if existing:
            raise ValueError("This application already has a frozen submission archive")
        if not package or package["application_id"] != application_id:
            raise ValueError("Package does not belong to this application")
        if package["context"] != "FORMAL_APPLICATION" or package["status"] != "READY":
            raise ValueError("Archive a READY formal application package after you submit it yourself")
        fields = self.checklist(application_id)
        pending = [f for f in fields if f["required"] and f["status"] not in {"REVIEWED", "EXCLUDED"}]
        if pending:
            raise ValueError("Required portal fields still need review")
        if not fields and not acknowledge_empty_checklist:
            raise ValueError("Confirm that the portal had no fields to record")
        for document in documents:
            version = self.vault.get_version(document["document_version_id"])
            if not version or version["sha256"] != document["sha256"] or not self.vault.storage.verify_hash(
                    version["storage_key"], document["sha256"]):
                raise ValueError("Package document hash no longer matches the Vault")
        answers = self.copy_ready(application_id)
        requirements = self.ledger.requirement_rows(application_id, verify=False)
        payload = {
            "application_id": application_id,
            "document_package_id": package_id,
            "confirmation_number": confirmation_number.strip(),
            "submitted_at": submitted_at,
            "payment_state": payment_state,
            "payment_reference": (payment_reference or "").strip() or None,
            "answers": answers,
            "referees": referees,
            "requirements": requirements,
            "package_sha256": package["package_sha256"],
            "documents": [{"document_version_id": d["document_version_id"], "sha256": d["sha256"],
                           "filename": d["canonical_filename"]} for d in documents],
        }
        digest = hashlib.sha256(_dump(payload).encode()).hexdigest()
        with transaction(self.db_path) as db:
            archive_id = db.execute("""INSERT INTO submission_archives
                (application_id,document_package_id,confirmation_number,submitted_at,submitted_by,payment_state,
                 payment_reference,payment_recorded_at,answers_json,referee_json,requirement_snapshot_json,
                 package_manifest_json,package_sha256,archive_sha256,notes,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (application_id, package_id, confirmation_number.strip(), submitted_at, actor.strip(),
                 payment_state, (payment_reference or "").strip() or None,
                 utc_now() if payment_state == "RECORDED" else None,
                 _dump(answers), _dump(referees), _dump(requirements), package["manifest_json"],
                 package["package_sha256"], digest, notes, utc_now())).lastrowid
            db.execute("UPDATE applications SET status='SUBMITTED',updated_at=? WHERE id=?",
                       (utc_now(), application_id))
        return archive_id

    def get_archive(self, archive_id: int) -> dict | None:
        with connect(self.db_path) as db:
            row = db.execute("SELECT * FROM submission_archives WHERE id=?", (archive_id,)).fetchone()
        return dict(row) if row else None
