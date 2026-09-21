"""Manual application ledger and count-based administrative readiness."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.documents import DOCUMENT_TYPES, LocalDocumentStorage


OPPORTUNITY_TYPES = ("ADVERTISED_POSITION", "PROGRAMME_APPLICATION", "FACULTY_ENQUIRY")
REQUIREMENT_STATES = ("REQUIRED", "OPTIONAL", "NOT_REQUESTED", "UNKNOWN")
REQUIREMENT_CONTEXTS = ("FACULTY_OUTREACH", "FORMAL_APPLICATION", "REPLY_REQUEST")
DEADLINE_TYPES = ("APPLICATION", "FUNDING", "SCHOLARSHIP", "DOCUMENT", "REFEREE", "INTERVIEW", "OTHER")
APPLICATION_STATUSES = (
    "PLANNING", "IN_PROGRESS", "WAITING_ON_REFEREE", "READY_TO_SUBMIT",
    "SUBMITTED", "INTERVIEW", "OFFER", "REJECTED", "WITHDRAWN",
)


def _required(value: str | None, label: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"{label} is required")
    return value


def _url(value: str | None) -> str:
    value = _required(value, "Source URL")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Source URL must start with http:// or https://")
    return value


def _row(row):
    return dict(row) if row is not None else None


def _date_or_datetime(value: str, label: str) -> str:
    value = _required(value, label)
    try:
        datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO date or date/time") from error
    return value


class Ledger:
    def __init__(self, db_path: Path | str, document_root: Path | None = None):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.storage = LocalDocumentStorage(document_root or self.db_path.parent / "documents")

    def _insert(self, table: str, values: dict) -> int:
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with transaction(self.db_path) as db:
            cursor = db.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(values.values())
            )
            return cursor.lastrowid

    def _update(self, table: str, row_id: int, changes: dict, allowed: set[str]) -> None:
        unexpected = set(changes) - allowed
        if unexpected:
            raise ValueError(f"Unsupported {table} fields: {sorted(unexpected)}")
        if not changes:
            return
        if table in {"programmes", "opportunities", "applications", "requirements", "application_referees"}:
            changes["updated_at"] = utc_now()
        assignments = ", ".join(f"{name} = ?" for name in changes)
        with transaction(self.db_path) as db:
            cursor = db.execute(
                f"UPDATE {table} SET {assignments} WHERE id = ?",
                (*changes.values(), row_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"{table} record {row_id} does not exist")

    def get(self, table: str, row_id: int) -> dict | None:
        if table not in {
            "programmes", "opportunities", "applications", "deadlines", "requirements",
            "application_tasks", "source_evidence", "application_referees",
        }:
            raise ValueError("Unsupported table")
        with connect(self.db_path) as db:
            return _row(db.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone())

    def create_evidence(
        self, canonical_url: str, source_type: str, relevant_excerpt: str = "",
        verification_state: str = "UNVERIFIED", retrieved_at: str | None = None,
        last_manually_verified_at: str | None = None,
    ) -> int:
        if verification_state not in {"UNVERIFIED", "VERIFIED", "NEEDS_REVIEW"}:
            raise ValueError("Invalid verification state")
        excerpt = relevant_excerpt.strip()
        now = utc_now()
        return self._insert("source_evidence", {
            "canonical_url": _url(canonical_url),
            "source_type": _required(source_type, "Source type"),
            "retrieved_at": retrieved_at or now,
            "relevant_excerpt": excerpt,
            "content_hash": hashlib.sha256(excerpt.encode("utf-8")).hexdigest() if excerpt else None,
            "verification_state": verification_state,
            "last_manually_verified_at": last_manually_verified_at or (now if verification_state == "VERIFIED" else None),
            "created_at": now,
        })

    def list_evidence(self) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute("SELECT * FROM source_evidence ORDER BY id DESC")]

    def create_programme(self, university: str, programme_name: str, **details) -> int:
        allowed = {
            "department", "degree_type", "cycle", "programme_url", "admissions_url", "portal_url", "notes",
            "country", "country_code", "country_source", "country_match_state",
            "qs_ranking_system", "qs_ranking_year", "qs_rank_display", "qs_rank_numeric",
            "qs_rank_band_low", "qs_rank_band_high", "qs_source_url", "qs_source_evidence_id",
            "qs_checked_at", "qs_match_state",
        }
        if set(details) - allowed:
            raise ValueError("Unsupported programme field")
        now = utc_now()
        return self._insert("programmes", {
            "university": _required(university, "University"),
            "programme_name": _required(programme_name, "Programme name"),
            **details, "created_at": now, "updated_at": now,
        })

    def update_programme(self, programme_id: int, **changes) -> None:
        self._update("programmes", programme_id, changes, {
            "university", "programme_name", "department", "degree_type", "cycle",
            "programme_url", "admissions_url", "portal_url", "notes",
            "country", "country_code", "country_source", "country_match_state",
            "qs_ranking_system", "qs_ranking_year", "qs_rank_display", "qs_rank_numeric",
            "qs_rank_band_low", "qs_rank_band_high", "qs_source_url", "qs_source_evidence_id",
            "qs_checked_at", "qs_match_state",
        })

    def list_programmes(self) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute("SELECT * FROM programmes ORDER BY university, programme_name")]

    def create_opportunity(self, opportunity_type: str, title: str, institution: str, **details) -> int:
        allowed = {
            "programme_id", "canonical_url", "department_lab", "supervisor_name", "research_area",
            "funding_text", "eligibility_text", "deadline_at", "deadline_timezone", "application_route",
            "contact_policy", "opening_status", "verification_state", "last_checked_at",
            "source_evidence_id", "notes",
        }
        if set(details) - allowed:
            raise ValueError("Unsupported opportunity field")
        if opportunity_type not in OPPORTUNITY_TYPES:
            raise ValueError("Invalid opportunity type")
        now = utc_now()
        return self._insert("opportunities", {
            "opportunity_type": opportunity_type, "title": _required(title, "Title"),
            "institution": _required(institution, "Institution"), **details,
            "created_at": now, "updated_at": now,
        })

    def update_opportunity(self, opportunity_id: int, **changes) -> None:
        self._update("opportunities", opportunity_id, changes, {
            "programme_id", "opportunity_type", "title", "canonical_url", "institution",
            "department_lab", "supervisor_name", "research_area", "funding_text",
            "eligibility_text", "deadline_at", "deadline_timezone", "application_route",
            "contact_policy", "opening_status", "verification_state", "last_checked_at",
            "source_evidence_id", "notes",
        })

    def list_opportunities(self) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute("SELECT * FROM opportunities ORDER BY institution, title")]

    def create_application(
        self, cycle: str, programme_id: int | None = None,
        opportunity_id: int | None = None, **details,
    ) -> int:
        allowed = {
            "status", "portal_url", "funding_state", "eligibility_state",
            "supervisor_contact_state", "next_action", "owner_notes",
        }
        if set(details) - allowed:
            raise ValueError("Unsupported application field")
        if not (programme_id or opportunity_id):
            raise ValueError("A programme or opportunity is required")
        if details.get("status", "PLANNING") not in APPLICATION_STATUSES:
            raise ValueError("Invalid application status")
        if programme_id and opportunity_id:
            opportunity = self.get("opportunities", opportunity_id)
            if not opportunity:
                raise ValueError("Opportunity does not exist")
            if opportunity["programme_id"] and opportunity["programme_id"] != programme_id:
                raise ValueError("Opportunity is linked to a different programme")
        now = utc_now()
        return self._insert("applications", {
            "programme_id": programme_id, "opportunity_id": opportunity_id,
            "cycle": _required(cycle, "Cycle"), **details,
            "created_at": now, "updated_at": now,
        })

    def update_application(self, application_id: int, **changes) -> None:
        if "status" in changes and changes["status"] not in APPLICATION_STATUSES:
            raise ValueError("Invalid application status")
        if changes.get("status") == "READY_TO_SUBMIT":
            with connect(self.db_path) as db:
                ready = db.execute("""SELECT 1 FROM application_packages WHERE application_id=?
                    AND context='FORMAL_APPLICATION' AND status='READY' ORDER BY version_number DESC LIMIT 1""",
                    (application_id,)).fetchone()
            if not ready:
                raise ValueError("A formal application package must pass preflight and be marked READY first")
        self._update("applications", application_id, changes, {
            "cycle", "status", "portal_url", "funding_state", "eligibility_state",
            "supervisor_contact_state", "next_action", "owner_notes",
        })

    def delete_application(self, application_id: int) -> None:
        with transaction(self.db_path) as db:
            cursor = db.execute("DELETE FROM applications WHERE id = ?", (application_id,))
            if cursor.rowcount != 1:
                raise ValueError("Application does not exist")

    def list_applications(self) -> list[dict]:
        sql = """SELECT a.*, COALESCE(p.university, o.institution) AS institution,
            COALESCE(p.programme_name, o.title) AS application_name,
            p.country, p.country_code, p.qs_rank_display, p.qs_ranking_year, p.qs_match_state,
            (SELECT MIN(d.due_at) FROM deadlines d WHERE d.application_id = a.id
             AND substr(d.due_at, 1, 10) >= ?) AS nearest_deadline
            FROM applications a
            LEFT JOIN programmes p ON p.id = a.programme_id
            LEFT JOIN opportunities o ON o.id = a.opportunity_id
            ORDER BY nearest_deadline IS NULL, nearest_deadline, a.id"""
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute(sql, (date.today().isoformat(),))]

    def create_deadline(
        self, application_id: int, deadline_type: str, due_at: str,
        source_evidence_id: int, **details,
    ) -> int:
        if deadline_type not in DEADLINE_TYPES:
            raise ValueError("Invalid deadline type")
        allowed = {"timezone", "verification_state", "last_checked_at", "notes"}
        if set(details) - allowed:
            raise ValueError("Unsupported deadline field")
        return self._insert("deadlines", {
            "application_id": application_id, "deadline_type": deadline_type,
            "due_at": _date_or_datetime(due_at, "Deadline"), "source_evidence_id": source_evidence_id,
            **details, "created_at": utc_now(),
        })

    def list_deadlines(self, application_id: int) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute(
                """SELECT d.*, e.canonical_url AS source_url FROM deadlines d
                JOIN source_evidence e ON e.id = d.source_evidence_id
                WHERE d.application_id = ? ORDER BY d.due_at""", (application_id,)
            )]

    def update_deadline(self, deadline_id: int, **changes) -> None:
        if "due_at" in changes:
            changes["due_at"] = _date_or_datetime(changes["due_at"], "Deadline")
        self._update("deadlines", deadline_id, changes, {
            "deadline_type", "due_at", "timezone", "source_evidence_id",
            "verification_state", "last_checked_at", "notes",
        })

    def create_requirement(
        self, application_id: int, context: str, requirement_state: str,
        original_label: str, source_evidence_id: int, **details,
    ) -> int:
        if context not in REQUIREMENT_CONTEXTS or requirement_state not in REQUIREMENT_STATES:
            raise ValueError("Invalid requirement context or state")
        allowed = {
            "normalized_document_type", "condition_text", "page_limit", "word_limit",
            "file_format", "filename_rule", "upload_field", "last_checked_at", "notes",
        }
        if set(details) - allowed:
            raise ValueError("Unsupported requirement field")
        doc_type = details.get("normalized_document_type")
        if doc_type and doc_type not in DOCUMENT_TYPES:
            raise ValueError("Invalid document type")
        now = utc_now()
        return self._insert("requirements", {
            "application_id": application_id, "context": context,
            "requirement_state": requirement_state,
            "original_label": _required(original_label, "Requirement label"),
            "source_evidence_id": source_evidence_id, **details,
            "created_at": now, "updated_at": now,
        })

    def update_requirement(self, requirement_id: int, **changes) -> None:
        if "context" in changes and changes["context"] not in REQUIREMENT_CONTEXTS:
            raise ValueError("Invalid context")
        if "requirement_state" in changes and changes["requirement_state"] not in REQUIREMENT_STATES:
            raise ValueError("Invalid requirement state")
        if changes.get("normalized_document_type") and changes["normalized_document_type"] not in DOCUMENT_TYPES:
            raise ValueError("Invalid document type")
        self._update("requirements", requirement_id, changes, {
            "context", "requirement_state", "original_label", "normalized_document_type",
            "condition_text", "page_limit", "word_limit", "file_format", "filename_rule",
            "upload_field", "source_evidence_id", "last_checked_at", "fulfilled_at", "notes",
        })

    def mark_requirement_fulfilled(self, requirement_id: int, fulfilled: bool) -> None:
        requirement = self.get("requirements", requirement_id)
        if not requirement:
            raise ValueError("Requirement does not exist")
        if requirement["normalized_document_type"]:
            raise ValueError("Document requirements must be linked to an approved version")
        self.update_requirement(requirement_id, fulfilled_at=utc_now() if fulfilled else None)

    def requirement_rows(self, application_id: int, *, verify: bool = True) -> list[dict]:
        sql = """SELECT r.*, e.canonical_url AS source_url,
            ad.id AS mapping_id, ad.document_version_id, ad.notes AS mapping_notes,
            v.version_number, v.sha256, v.storage_key, v.approval_state,
            d.id AS document_id, d.title AS document_title, d.document_type,
            v.expiry_date
            FROM requirements r
            JOIN source_evidence e ON e.id = r.source_evidence_id
            LEFT JOIN application_documents ad ON ad.requirement_id = r.id
                AND ad.application_id = r.application_id
            LEFT JOIN document_versions v ON v.id = ad.document_version_id
            LEFT JOIN documents d ON d.id = v.document_id
            WHERE r.application_id = ? ORDER BY r.context, r.id"""
        with connect(self.db_path) as db:
            rows = [dict(r) for r in db.execute(sql, (application_id,))]
        today = date.today().isoformat()
        for row in rows:
            if not row["normalized_document_type"]:
                row["document_state"] = "AVAILABLE" if row["fulfilled_at"] else "MISSING"
            elif not row["document_version_id"]:
                row["document_state"] = "MISSING"
            elif row["document_type"] != row["normalized_document_type"]:
                row["document_state"] = "NEEDS_UPDATE"
            elif row["expiry_date"] and row["expiry_date"] < today:
                row["document_state"] = "NEEDS_UPDATE"
            elif row["approval_state"] != "APPROVED":
                row["document_state"] = "NEEDS_APPROVAL"
            elif not (
                self.storage.verify_hash(row["storage_key"], row["sha256"]) if verify
                else self.storage.exists(row["storage_key"])
            ):
                row["document_state"] = "NEEDS_UPDATE"
            else:
                row["document_state"] = "AVAILABLE"
        return rows

    def readiness(self, application_id: int, context: str = "FORMAL_APPLICATION") -> dict:
        if context not in REQUIREMENT_CONTEXTS:
            raise ValueError("Invalid requirement context")
        rows = [r for r in self.requirement_rows(application_id) if r["context"] == context]
        required = [r for r in rows if r["requirement_state"] == "REQUIRED"]
        complete = sum(r["document_state"] == "AVAILABLE" for r in required)
        return {
            "required_complete": complete,
            "required_total": len(required),
            "unknown_requirements": sum(r["requirement_state"] == "UNKNOWN" for r in rows),
            "conditional_requirements": sum(bool(r["condition_text"]) for r in rows),
        }

    def create_task(
        self, application_id: int, task_type: str, description: str, **details,
    ) -> int:
        allowed = {"status", "due_at", "priority", "source_context", "notes"}
        if set(details) - allowed:
            raise ValueError("Unsupported task field")
        if details.get("due_at"):
            details["due_at"] = _date_or_datetime(details["due_at"], "Task due date")
        status = details.get("status", "TODO")
        return self._insert("application_tasks", {
            "application_id": application_id,
            "task_type": _required(task_type, "Task type"),
            "description": _required(description, "Description"), **details,
            "created_at": utc_now(),
            "completed_at": utc_now() if status == "DONE" else None,
        })

    def update_task(self, task_id: int, **changes) -> None:
        if changes.get("due_at"):
            changes["due_at"] = _date_or_datetime(changes["due_at"], "Task due date")
        if "status" in changes:
            changes["completed_at"] = utc_now() if changes["status"] == "DONE" else None
        self._update("application_tasks", task_id, changes, {
            "task_type", "description", "status", "due_at", "priority",
            "source_context", "completed_at", "notes",
        })

    def delete_task(self, task_id: int) -> None:
        with transaction(self.db_path) as db:
            cursor = db.execute("DELETE FROM application_tasks WHERE id = ?", (task_id,))
            if cursor.rowcount != 1:
                raise ValueError("Task does not exist")

    def list_tasks(self, application_id: int) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM application_tasks WHERE application_id = ? ORDER BY status, due_at, id",
                (application_id,),
            )]

    def create_referee(self, application_id: int, referee_name: str, **details) -> int:
        allowed = {
            "institution", "email", "relationship", "invitation_state", "requested_at",
            "deadline_at", "submission_state", "submitted_at", "notes",
        }
        if set(details) - allowed:
            raise ValueError("Unsupported referee field")
        if details.get("deadline_at"):
            details["deadline_at"] = _date_or_datetime(details["deadline_at"], "Referee deadline")
        now = utc_now()
        return self._insert("application_referees", {
            "application_id": application_id,
            "referee_name": _required(referee_name, "Referee name"),
            **details, "created_at": now, "updated_at": now,
        })

    def update_referee(self, referee_id: int, **changes) -> None:
        if changes.get("deadline_at"):
            changes["deadline_at"] = _date_or_datetime(changes["deadline_at"], "Referee deadline")
        self._update("application_referees", referee_id, changes, {
            "referee_name", "institution", "email", "relationship", "invitation_state",
            "requested_at", "deadline_at", "submission_state", "submitted_at", "notes",
        })

    def list_referees(self, application_id: int) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM application_referees WHERE application_id = ? ORDER BY id",
                (application_id,),
            )]

    def today(self) -> dict:
        today = date.today().isoformat()
        cutoff = (date.today() + timedelta(days=60)).isoformat()
        with connect(self.db_path) as db:
            upcoming = [dict(r) for r in db.execute("""
                SELECT d.*, COALESCE(p.university, o.institution) AS institution,
                e.canonical_url AS source_url
                FROM deadlines d JOIN applications a ON a.id = d.application_id
                LEFT JOIN programmes p ON p.id = a.programme_id
                LEFT JOIN opportunities o ON o.id = a.opportunity_id
                JOIN source_evidence e ON e.id = d.source_evidence_id
                WHERE substr(d.due_at, 1, 10) BETWEEN ? AND ?
                ORDER BY d.due_at LIMIT 50""", (today, cutoff))]
            overdue = [dict(r) for r in db.execute("""
                SELECT t.*, COALESCE(p.university, o.institution) AS institution
                FROM application_tasks t JOIN applications a ON a.id = t.application_id
                LEFT JOIN programmes p ON p.id = a.programme_id
                LEFT JOIN opportunities o ON o.id = a.opportunity_id
                WHERE t.due_at IS NOT NULL AND substr(t.due_at, 1, 10) < ?
                  AND t.status NOT IN ('DONE','CANCELLED')
                ORDER BY t.due_at""", (today,))]
            documents = [dict(r) for r in db.execute("""
                SELECT d.id, d.title, d.expiry_date, v.id AS version_id, v.approval_state
                FROM documents d JOIN document_versions v ON v.document_id = d.id
                WHERE v.version_number = (SELECT MAX(v2.version_number) FROM document_versions v2
                                          WHERE v2.document_id = d.id)
                  AND (v.approval_state != 'APPROVED' OR
                       (d.expiry_date IS NOT NULL AND d.expiry_date <= ?))
                ORDER BY d.expiry_date""", (cutoff,))]
        missing = []
        unknown = []
        for app in self.list_applications():
            for requirement in self.requirement_rows(app["id"], verify=False):
                item = {"application_id": app["id"], "institution": app["institution"], **requirement}
                if requirement["requirement_state"] == "REQUIRED" and requirement["document_state"] != "AVAILABLE":
                    missing.append(item)
                if requirement["requirement_state"] == "UNKNOWN":
                    unknown.append(item)
        return {
            "upcoming_deadlines": upcoming, "missing_required": missing,
            "unknown_requirements": unknown, "overdue_tasks": overdue,
            "document_alerts": documents,
        }
