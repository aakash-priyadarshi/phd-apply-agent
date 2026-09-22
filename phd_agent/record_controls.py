"""Applicant-facing corrections and reversible cleanup over the existing ledger."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.ledger import Ledger
from phd_agent.university_enrichment import candidate_matches_filters, funding_looks_funded, resolve_country_label


PROGRAMME_TYPES = (
    "PHD", "DPHIL", "CDT", "DOCTORAL_TRAINING", "INDUSTRIAL_PHD",
    "PROJECT_SPECIFIC_PHD", "OPEN_RESEARCH_PHD", "RA_TO_PHD", "OTHER",
)


def programme_type(title: str, degree: str | None = None) -> str:
    name = f"{title} {degree or ''}".casefold()
    if "dphil" in name:
        return "DPHIL"
    if "cdt" in name or "centre for doctoral training" in name:
        return "CDT"
    if "industrial" in name:
        return "INDUSTRIAL_PHD"
    if "doctoral training" in name:
        return "DOCTORAL_TRAINING"
    if "research assistant" in name or "ra to phd" in name:
        return "RA_TO_PHD"
    if "project" in name and ("phd" in name or "doctoral" in name):
        return "PROJECT_SPECIFIC_PHD"
    if "open research" in name and ("phd" in name or "doctoral" in name):
        return "OPEN_RESEARCH_PHD"
    if "phd" in name or "doctor" in name:
        return "PHD"
    return "OTHER"


def group_programmes(items: list[dict], *, view: str = "Country") -> dict:
    if view not in {"Country", "University"}:
        raise ValueError("Choose Country or University")
    result = defaultdict(lambda: defaultdict(list))
    for item in items:
        payload = item.get("payload", item)
        country = payload.get("country") or "Country unknown"
        university = payload.get("university") or item.get("institution") or "University unknown"
        result[country if view == "Country" else university][university if view == "Country" else country].append(item)
    return {key: dict(sorted(children.items())) for key, children in sorted(result.items())}


def filter_programmes(items: list[dict], criteria: dict) -> list[dict]:
    """Apply explicit search criteria without mistaking unknowns for matches."""
    base = {key: criteria[key] for key in ("country_codes", "qs_max", "min_research_fit")
            if criteria.get(key) not in (None, [], "")}
    kinds = set(criteria.get("programme_types") or [])
    funding = criteria.get("funding", "Any")
    result = []
    for item in items:
        payload = item.get("payload", item)
        if not candidate_matches_filters(payload, base):
            continue
        if kinds and programme_type(payload.get("programme") or payload.get("programme_name") or "",
                                    payload.get("degree") or payload.get("degree_type")) not in kinds:
            continue
        if funding == "Funded" and not funding_looks_funded(payload.get("funding")):
            continue
        funding_text = str(payload.get("funding") or "").strip().upper()
        if funding == "Unknown" and funding_text not in {"", "UNKNOWN"}:
            continue
        result.append(item)
    return result


class RecordControls:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.ledger = Ledger(self.db_path)

    def _audit(self, db, entity: str, row_id: int, action: str, changes: dict) -> None:
        db.execute("""INSERT INTO record_change_events(entity_type,entity_id,action,changes_json,created_at)
            VALUES(?,?,?,?,?)""", (entity, row_id, action, json.dumps(changes, sort_keys=True), utc_now()))

    def edit_application(self, application_id: int, *, cycle: str, status: str,
                         portal_url: str, next_action: str, owner_notes: str,
                         funding_state: str | None = None, eligibility_state: str | None = None,
                         supervisor_contact_state: str | None = None,
                         source_url: str | None = None) -> None:
        before = self.ledger.get("applications", application_id)
        if not before or before.get("archived_at"):
            raise ValueError("Application is unavailable")
        changes = {"cycle": cycle.strip(), "status": status, "portal_url": portal_url.strip() or None,
                   "next_action": next_action.strip(), "owner_notes": owner_notes.strip()}
        choices = {"funding_state": ("UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING", "CONFIRMED"),
                   "eligibility_state": ("UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING"),
                   "supervisor_contact_state": ("UNKNOWN", "NOT_CONTACTED", "CONTACTED", "REPLIED", "NOT_REQUIRED")}
        for field, value in (("funding_state", funding_state), ("eligibility_state", eligibility_state),
                             ("supervisor_contact_state", supervisor_contact_state)):
            if value is not None:
                if value not in choices[field]:
                    raise ValueError("Invalid application detail")
                changes[field] = value
        if source_url:
            from phd_agent.ledger import _url
            _url(source_url)
        if (changes.get("eligibility_state") == "ELIGIBLE" and before["eligibility_state"] != "ELIGIBLE"
                or changes.get("funding_state") == "CONFIRMED" and before["funding_state"] != "CONFIRMED") and not source_url:
            raise ValueError("Provide an official source before confirming eligibility or funding")
        if not changes["cycle"]:
            raise ValueError("Application cycle is required")
        if changes["portal_url"]:
            from phd_agent.ledger import _url
            _url(changes["portal_url"])
        with transaction(self.db_path) as db:
            self.ledger.update_application(application_id, db=db, **changes)
            for field in choices:
                if field in changes and before[field] != changes[field]:
                    db.execute("""INSERT INTO record_field_reviews
                        (entity_type,entity_id,field_name,field_value,verification_state,source_url,recorded_at)
                        VALUES('APPLICATION',?,?,?,?,?,?)""",
                        (application_id, field, changes[field],
                         "OPERATOR_CONFIRMED" if source_url else "NEEDS_REVIEW", source_url, utc_now()))
            self._audit(db, "APPLICATION", application_id, "EDIT",
                        {key: {"before": before[key], "after": value} for key, value in changes.items()
                         if before[key] != value})

    def edit_programme(self, programme_id: int, *, university: str, programme_name: str,
                       country: str, department: str, degree_type: str, programme_url: str,
                       source_url: str | None = None, verified: bool = False) -> None:
        before = self.ledger.get("programmes", programme_id)
        if not before:
            raise ValueError("Programme does not exist")
        if not university.strip() or not programme_name.strip():
            raise ValueError("University and programme name are required")
        if source_url:
            from phd_agent.ledger import _url
            _url(source_url)
        if verified and not source_url:
            raise ValueError("Provide the official source URL before confirming facts")
        normalized_country, code = resolve_country_label(country)
        if country.strip() and not code:
            raise ValueError("Choose a recognized country")
        changes = {
            "university": university.strip(), "programme_name": programme_name.strip(),
            "country": normalized_country, "country_code": code,
            "department": department.strip() or None, "degree_type": degree_type.strip() or None,
            "programme_url": programme_url.strip() or None,
        }
        if changes["programme_url"]:
            from phd_agent.ledger import _url
            _url(changes["programme_url"])
        if before["country"] != changes["country"]:
            changes["country_source"] = "OPERATOR" if verified else "MANUAL_UNVERIFIED"
            changes["country_match_state"] = "CONFIRMED" if verified else "UNKNOWN"
        with transaction(self.db_path) as db:
            self.ledger.update_programme(programme_id, db=db, **changes)
            differences = {}
            for key, value in changes.items():
                if before[key] == value:
                    continue
                differences[key] = {"before": before[key], "after": value}
                db.execute("""INSERT INTO record_field_reviews
                    (entity_type,entity_id,field_name,field_value,verification_state,source_url,recorded_at)
                    VALUES('PROGRAMME',?,?,?,?,?,?)""",
                    (programme_id, key, str(value) if value is not None else None,
                     "OPERATOR_CONFIRMED" if verified else "NEEDS_REVIEW", source_url, utc_now()))
            self._audit(db, "PROGRAMME", programme_id, "EDIT", differences)

    def set_deadline(self, application_id: int, due_at: str, *, source_url: str,
                     verified: bool = False) -> int:
        if not self.ledger.get("applications", application_id):
            raise ValueError("Application does not exist")
        with transaction(self.db_path) as db:
            source = self.ledger.create_evidence(source_url, "OPERATOR_DEADLINE", due_at,
                                                 "VERIFIED" if verified else "NEEDS_REVIEW", db=db)
            deadline = self.ledger.create_deadline(application_id, "APPLICATION", due_at, source,
                                                   verification_state="VERIFIED" if verified else "NEEDS_REVIEW",
                                                   last_checked_at=utc_now(), db=db)
            self._audit(db, "APPLICATION", application_id, "DEADLINE_ADDED",
                        {"deadline_id": deadline, "verified": verified, "source_url": source_url})
        return deadline

    def impact(self, application_id: int) -> dict:
        app = self.ledger.get("applications", application_id)
        if not app:
            raise ValueError("Application does not exist")
        with connect(self.db_path) as db:
            return {"professors": db.execute(
                        "SELECT COUNT(*) FROM application_faculty WHERE application_id=?", (application_id,)).fetchone()[0],
                    "tasks": db.execute(
                        "SELECT COUNT(*) FROM application_tasks WHERE application_id=?", (application_id,)).fetchone()[0],
                    "documents": db.execute(
                        "SELECT COUNT(*) FROM application_documents WHERE application_id=?", (application_id,)).fetchone()[0]}

    def archive_application(self, application_id: int, archived: bool = True) -> None:
        self.impact(application_id)
        with transaction(self.db_path) as db:
            db.execute("UPDATE applications SET archived_at=?,updated_at=? WHERE id=?",
                       (utc_now() if archived else None, utc_now(), application_id))
            self._audit(db, "APPLICATION", application_id, "ARCHIVE" if archived else "RESTORE", {})

    def archive_candidate(self, candidate_id: int, archived: bool = True) -> None:
        with transaction(self.db_path) as db:
            row = db.execute("SELECT review_state FROM programme_candidates WHERE id=?", (candidate_id,)).fetchone()
            if not row or row["review_state"] == "ACCEPTED":
                raise ValueError("Accepted results are managed through their application")
            db.execute("UPDATE programme_candidates SET archived_at=? WHERE id=?",
                       (utc_now() if archived else None, candidate_id))
            self._audit(db, "CANDIDATE", candidate_id, "ARCHIVE" if archived else "RESTORE", {})

    def archive_candidates(self, candidate_ids: list[int]) -> int:
        ids = list(dict.fromkeys(candidate_ids))
        if not ids:
            return 0
        with transaction(self.db_path) as db:
            rows = db.execute("SELECT id,review_state FROM programme_candidates WHERE id IN (" +
                              ",".join("?" for _ in ids) + ")", ids).fetchall()
            if len(rows) != len(ids) or any(row["review_state"] == "ACCEPTED" for row in rows):
                raise ValueError("Select only unaccepted programme results")
            for candidate_id in ids:
                db.execute("UPDATE programme_candidates SET archived_at=? WHERE id=?",
                           (utc_now(), candidate_id))
                self._audit(db, "CANDIDATE", candidate_id, "ARCHIVE", {})
        return len(ids)
