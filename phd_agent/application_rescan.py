"""Scan official programme pages for missing application facts.

Extracted values stay EXTRACTED until an applicant accepts them. Confirmed and
verified values are never replaced in place, and a frozen submission archive is
left untouched.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.ledger import Ledger
from phd_agent.model_router import TIERS, ModelRouter


PAGE_BUDGET = 12
STALE_DAYS = 180
SCAN_MISSING = "APPLICATION_DETAIL_SCAN"
SCAN_FULL = "APPLICATION_FULL_REFRESH"
ACTIVE_SCAN = ("QUEUED", "RUNNING", "CANCEL_REQUESTED", "PAUSED")
HUMAN_MESSAGE = "I could not read this page reliably."
LINK_HINTS = (
    "admission", "apply", "entry-requirement", "entry-requirements", "funding",
    "fee", "english", "supervisor", "document", "reference", "referee", "phd", "dphil",
)
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
FIELDS = {
    "deadline": {"label": "Deadline", "category": "Programme", "high_impact": True},
    "deadline_timezone": {"label": "Deadline timezone", "category": "Programme", "high_impact": False},
    "intake": {"label": "Intake", "category": "Programme", "high_impact": False},
    "opening_status": {"label": "Opening status", "category": "Programme", "high_impact": False},
    "funding": {"label": "Funding", "category": "Funding", "high_impact": True},
    "funding_deadline": {"label": "Funding deadline", "category": "Funding", "high_impact": True},
    "english_requirements": {"label": "English requirement", "category": "Requirements", "high_impact": True},
    "referee_count": {"label": "Referees", "category": "Requirements", "high_impact": True},
    "research_proposal": {"label": "Research proposal", "category": "Requirements", "high_impact": True},
    "application_fee": {"label": "Application fee", "category": "Application", "high_impact": True},
    "portal_url": {"label": "Portal URL", "category": "Application", "high_impact": False},
    "application_route": {"label": "Application route", "category": "Application", "high_impact": True},
    "supervisor_contact_policy": {"label": "Supervisor contact policy", "category": "Supervisor", "high_impact": True},
}


@dataclass
class FieldFact:
    name: str
    value: str | None
    state: str
    source: str | None
    checked_at: str | None
    protected: bool
    stale: bool

    @property
    def label(self) -> str:
        return FIELDS[self.name]["label"]

    @property
    def category(self) -> str:
        return FIELDS[self.name]["category"]

    @property
    def high_impact(self) -> bool:
        return FIELDS[self.name]["high_impact"]


@dataclass
class Parsed:
    value: str
    excerpt: str
    ambiguous: bool = False


def _blank(name: str) -> FieldFact:
    return FieldFact(name, None, "UNKNOWN", None, None, False, False)


def _known(fact: FieldFact) -> bool:
    return bool(fact.value and fact.value.strip().upper() != "UNKNOWN" and fact.state not in {"UNKNOWN", "CONFLICT"})


def _stale(checked_at: str | None, *, today: date | None = None) -> bool:
    if not checked_at:
        return False
    try:
        checked = date.fromisoformat(checked_at[:10])
    except ValueError:
        return False
    return ((today or date.today()) - checked).days > STALE_DAYS


def _same(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    return left.strip().casefold()[:10] == right.strip().casefold()[:10] if _looks_date(left) and _looks_date(right) else left.strip().casefold() == right.strip().casefold()


def _looks_date(value: str) -> bool:
    return bool(re.fullmatch(r"20\d{2}-\d{2}-\d{2}", value.strip()[:10]))


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n+", text) if part.strip()]


def _snippet(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip()[:500]


def _long_date(match: re.Match) -> str:
    return date(int(match.group(3)), MONTHS[match.group(2).casefold()], int(match.group(1))).isoformat()


def _dates_in(sentence: str) -> list[str]:
    found = [match.group(0) for match in re.finditer(r"\b(20\d{2}-\d{2}-\d{2})\b", sentence)]
    found.extend(_long_date(match) for match in re.finditer(
        r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b",
        sentence, flags=re.I))
    return found


def _dated(text: str, keywords: tuple[str, ...]) -> Parsed | None:
    hits: list[tuple[str, str]] = []
    for sentence in _sentences(text):
        folded = sentence.casefold()
        if not any(keyword in folded for keyword in keywords):
            continue
        for value in _dates_in(sentence):
            hits.append((value, sentence))
    unique = list(dict.fromkeys(value for value, _ in hits))
    if not unique:
        return None
    sentence = hits[0][1]
    return Parsed(unique[0], _snippet(sentence), ambiguous=len(unique) > 1)


def _portal(html: str, base_url: str) -> str | None:
    if not html or not base_url:
        return None
    host = (urlparse(base_url).hostname or "").casefold()
    for anchor in BeautifulSoup(html, "html.parser").select("a[href]"):
        href = urljoin(base_url, anchor.get("href") or "").split("#", 1)[0]
        parsed = urlparse(href)
        if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").casefold() != host:
            continue
        haystack = f"{parsed.path} {anchor.get_text(' ', strip=True)}".casefold()
        if "apply" in haystack or "portal" in haystack:
            return href
    return None


def official_links(html: str, base_url: str, *, limit: int) -> list[str]:
    """Same-institution links that look like admissions, funding, or requirements pages."""
    if not html or limit < 1:
        return []
    host = (urlparse(base_url).hostname or "").casefold()
    found: list[str] = []
    seen = {base_url.split("#", 1)[0].casefold()}
    for anchor in BeautifulSoup(html, "html.parser").select("a[href]"):
        href = urljoin(base_url, anchor.get("href") or "").split("#", 1)[0]
        parsed = urlparse(href)
        if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").casefold() != host:
            continue
        haystack = f"{parsed.path} {anchor.get_text(' ', strip=True)}".casefold()
        if not any(hint in haystack for hint in LINK_HINTS):
            continue
        if href.casefold() in seen:
            continue
        seen.add(href.casefold())
        found.append(href)
        if len(found) >= limit:
            break
    return found


def parse_page(text: str, html: str = "", base_url: str = "") -> dict[str, Parsed]:
    """Deterministic extraction. Ambiguous prose is flagged instead of guessed."""
    found: dict[str, Parsed] = {}
    deadline = _dated(text, ("deadline", "applications close", "apply by", "closing date"))
    if deadline:
        found["deadline"] = deadline
        for sentence in _sentences(text):
            if deadline.value[:10] in sentence or "deadline" in sentence.casefold() or "close" in sentence.casefold():
                zone = re.search(r"\b(UTC|GMT|BST|UK time)\b", sentence, flags=re.I)
                if zone:
                    found["deadline_timezone"] = Parsed(zone.group(1).upper(), _snippet(sentence))
                    break
    funding_deadline = _dated(text, ("funding deadline", "scholarship deadline", "studentship deadline"))
    if funding_deadline:
        found["funding_deadline"] = funding_deadline
    for sentence in _sentences(text):
        folded = sentence.casefold()
        if "intake" not in found and re.search(r"\b(20\d{2})\s+entry\b", sentence, flags=re.I):
            month = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})", sentence, flags=re.I)
            found["intake"] = Parsed(month.group(0) if month else re.search(r"20\d{2}", sentence).group(0) + " entry", _snippet(sentence))
        if "opening_status" not in found and "applications are open" in folded:
            found["opening_status"] = Parsed("OPEN", _snippet(sentence))
        elif "opening_status" not in found and "applications are closed" in folded:
            found["opening_status"] = Parsed("CLOSED", _snippet(sentence))
        if "funding" not in found and any(token in folded for token in ("studentship", "scholarship", "fully funded", "funding is available")):
            found["funding"] = Parsed(_snippet(sentence), _snippet(sentence))
        if "english_requirements" not in found:
            score = re.search(r"IELTS[^\n.]{0,24}?(\d(?:\.\d)?)", sentence, flags=re.I)
            if score:
                found["english_requirements"] = Parsed(f"IELTS {score.group(1)}", _snippet(sentence))
        if "referee_count" not in found:
            count = re.search(r"\b(\d)\s+referees?\b", sentence, flags=re.I)
            words = re.search(r"\b(one|two|three|four|five)\s+referees?\b", sentence, flags=re.I)
            if count or words:
                found["referee_count"] = Parsed(count.group(1) if count else NUMBER_WORDS[words.group(1).casefold()], _snippet(sentence))
        if "research_proposal" not in found and "research proposal" in folded:
            found["research_proposal"] = Parsed("Not requested" if "no research proposal" in folded else "Required", _snippet(sentence))
        if "application_fee" not in found:
            fee = re.search(r"(?:application fee|fee)[^\n£$€]{0,40}([£$€]\s?\d{2,5})", sentence, flags=re.I)
            if fee:
                found["application_fee"] = Parsed(re.sub(r"\s+", "", fee.group(1)), _snippet(sentence))
        if "supervisor_contact_policy" not in found and "supervisor" in folded:
            if "discouraged" in folded:
                policy = "discouraged"
            elif "not required" in folded or "no need to contact" in folded or "not requested" in folded:
                policy = "not requested"
            elif "encouraged" in folded:
                policy = "encouraged"
            elif "optional" in folded:
                policy = "optional"
            elif "must contact" in folded or "required to contact" in folded or "pre-contact" in folded:
                policy = "pre-contact required"
            else:
                policy = ""
            if policy:
                found["supervisor_contact_policy"] = Parsed(policy, _snippet(sentence))
    portal = _portal(html, base_url)
    if portal:
        found["portal_url"] = Parsed(portal, portal)
        found.setdefault("application_route", Parsed("Official online application", portal))
    return {key: value for key, value in found.items() if key in FIELDS}


def stage_for(url: str) -> str:
    path = urlparse(url).path.casefold()
    for label, token in (
        ("Admissions page", "admission"), ("Entry requirements", "entry"), ("Funding page", "funding"),
        ("Fees page", "fee"), ("English requirements", "english"), ("Supervisor information", "supervisor"),
        ("References", "reference"), ("Application page", "apply"),
    ):
        if token in path:
            return label
    return "Programme page"


class ApplicationRescan:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.ledger = Ledger(self.db_path)
        self.router = ModelRouter()

    def _audit(self, db, application_id: int, action: str, changes: dict) -> None:
        db.execute("""INSERT INTO record_change_events(entity_type,entity_id,action,changes_json,created_at)
            VALUES('APPLICATION',?,?,?,?)""",
            (application_id, action, json.dumps(changes, sort_keys=True), utc_now()))

    def _record(self, application_id: int) -> dict:
        with connect(self.db_path) as db:
            row = db.execute("""SELECT a.*, COALESCE(p.university,o.institution) AS institution,
                COALESCE(p.programme_name,o.title) AS programme_name,
                p.programme_url, p.admissions_url, p.portal_url AS programme_portal, p.id AS linked_programme_id,
                o.deadline_at AS opportunity_deadline, o.deadline_timezone AS opportunity_zone,
                o.funding_text, o.contact_policy, o.application_route AS opportunity_route,
                o.opening_status AS opportunity_opening, o.canonical_url AS opportunity_url,
                o.last_checked_at AS opportunity_checked, o.verification_state AS opportunity_state
                FROM applications a
                LEFT JOIN programmes p ON p.id=a.programme_id
                LEFT JOIN opportunities o ON o.id=a.opportunity_id
                WHERE a.id=?""", (application_id,)).fetchone()
        if not row or row["archived_at"]:
            raise ValueError("Application is unavailable")
        return dict(row)

    def _facts(self, db, application_id: int, record: dict | None = None) -> dict[str, FieldFact]:
        record = record or dict(db.execute("""SELECT a.*, COALESCE(p.university,o.institution) AS institution,
            p.programme_url, p.admissions_url, p.portal_url AS programme_portal,
            o.funding_text, o.contact_policy, o.application_route AS opportunity_route,
            o.opening_status AS opportunity_opening, o.canonical_url AS opportunity_url,
            o.last_checked_at AS opportunity_checked, o.verification_state AS opportunity_state
            FROM applications a
            LEFT JOIN programmes p ON p.id=a.programme_id
            LEFT JOIN opportunities o ON o.id=a.opportunity_id
            WHERE a.id=?""", (application_id,)).fetchone())
        facts = {name: _blank(name) for name in FIELDS}
        reviews = {}
        for row in db.execute("""SELECT field_name, field_value, verification_state, source_url, recorded_at
                FROM record_field_reviews WHERE entity_type='APPLICATION' AND entity_id=? ORDER BY id""",
                (application_id,)):
            reviews[row["field_name"]] = row
        stored = {row["field_name"]: row for row in db.execute(
            "SELECT * FROM application_field_states WHERE application_id=?", (application_id,))}
        deadlines = [dict(row) for row in db.execute("""SELECT d.*, e.canonical_url AS source_url
            FROM deadlines d JOIN source_evidence e ON e.id=d.source_evidence_id
            WHERE d.application_id=? ORDER BY d.id""", (application_id,))]
        open_conflicts = {row[0] for row in db.execute(
            "SELECT field_name FROM application_field_conflicts WHERE application_id=? AND status='OPEN'",
            (application_id,))}
        ledger_values = {
            "portal_url": (record.get("portal_url") or record.get("programme_portal"), "UNVERIFIED", None, record.get("programme_url")),
            "application_route": (record.get("opportunity_route"), record.get("opportunity_state") or "UNKNOWN", record.get("opportunity_checked"), record.get("opportunity_url")),
            "opening_status": (None if (record.get("opportunity_opening") or "UNKNOWN") == "UNKNOWN" else record.get("opportunity_opening"),
                               record.get("opportunity_state") or "UNKNOWN", record.get("opportunity_checked"), record.get("opportunity_url")),
            "funding": (record.get("funding_text"), record.get("opportunity_state") or "UNKNOWN", record.get("opportunity_checked"), record.get("opportunity_url")),
            "supervisor_contact_policy": (record.get("contact_policy"), record.get("opportunity_state") or "UNKNOWN", record.get("opportunity_checked"), record.get("opportunity_url")),
        }
        for deadline in deadlines:
            key = "funding_deadline" if deadline["deadline_type"] == "FUNDING" else "deadline" if deadline["deadline_type"] == "APPLICATION" else ""
            if key:
                ledger_values[key] = (deadline["due_at"][:10], deadline["verification_state"], deadline["last_checked_at"], deadline["source_url"])
                if key == "deadline" and deadline["timezone"]:
                    ledger_values["deadline_timezone"] = (deadline["timezone"], deadline["verification_state"], deadline["last_checked_at"], deadline["source_url"])
        for name, fact in facts.items():
            value, state, checked, source = ledger_values.get(name, (None, "UNKNOWN", None, None))
            if value and str(value).strip().upper() != "UNKNOWN":
                fact.value, fact.state, fact.checked_at, fact.source = str(value).strip(), state or "UNVERIFIED", checked, source
            review = reviews.get(name)
            if review and review["verification_state"] == "OPERATOR_CONFIRMED":
                fact.protected = True
                fact.state = "OPERATOR_CONFIRMED"
                fact.value = review["field_value"] or fact.value
                fact.source = review["source_url"] or fact.source
                fact.checked_at = review["recorded_at"]
            elif fact.state == "VERIFIED":
                fact.protected = True
            elif name in stored and not fact.protected:
                row = stored[name]
                fact.value = row["field_value"]
                fact.state = row["state"]
                fact.source = row["source_url"]
                fact.checked_at = row["checked_at"]
                fact.protected = row["state"] in {"OPERATOR_CONFIRMED", "VERIFIED"}
            if name in open_conflicts:
                fact.state = "CONFLICT"
            fact.stale = _stale(fact.checked_at)
        return facts

    def facts(self, application_id: int) -> dict[str, FieldFact]:
        record = self._record(application_id)
        with connect(self.db_path) as db:
            return self._facts(db, application_id, record)

    def plan_targets(self, application_id: int, *, mode: str = "MISSING", fields: list[str] | None = None) -> list[str]:
        if mode not in {"MISSING", "FULL"}:
            raise ValueError("Choose a missing-detail scan or a full refresh")
        requested = list(fields or [])
        if any(name not in FIELDS for name in requested):
            raise ValueError("Unknown application field")
        current = self.facts(application_id)
        if mode == "FULL":
            names = requested or list(FIELDS)
        else:
            names = [name for name, fact in current.items()
                     if not (fact.protected and not fact.stale)
                     and (not _known(fact) or fact.stale or fact.state in {"NEEDS_REVIEW", "CONFLICT", "UNKNOWN"})]
            if requested:
                names = [name for name in names if name in requested]
        return names

    def completeness(self, application_id: int) -> dict:
        current = self.facts(application_id)
        data = [fact for fact in current.values() if fact.category != "Requirements"]
        requirements = [fact for fact in current.values() if fact.category == "Requirements"]
        missing = [fact for fact in current.values() if not _known(fact) or fact.stale or fact.state == "CONFLICT"]
        groups: dict[str, list[dict]] = {}
        for fact in missing:
            groups.setdefault(fact.category, []).append({"field": fact.name, "label": fact.label, "state": fact.state})
        return {
            "known": sum(_known(fact) and fact.state != "CONFLICT" and not fact.stale for fact in data),
            "total": len(data),
            "requirements_known": sum(_known(fact) and fact.state != "CONFLICT" for fact in requirements),
            "requirements_total": len(requirements),
            "missing": [{"field": fact.name, "label": fact.label, "category": fact.category, "state": fact.state} for fact in missing],
            "groups": groups,
        }

    def active_operation(self, application_id: int, operation_type: str | None = None) -> int | None:
        sql = """SELECT id FROM operations WHERE application_id=? AND operation_type IN (?,?)
            AND status IN ('QUEUED','RUNNING','CANCEL_REQUESTED','PAUSED') ORDER BY id DESC LIMIT 1"""
        values: tuple = (application_id, SCAN_MISSING, SCAN_FULL)
        if operation_type:
            sql = """SELECT id FROM operations WHERE application_id=? AND operation_type=?
                AND status IN ('QUEUED','RUNNING','CANCEL_REQUESTED','PAUSED') ORDER BY id DESC LIMIT 1"""
            values = (application_id, operation_type)
        with connect(self.db_path) as db:
            row = db.execute(sql, values).fetchone()
        return row[0] if row else None

    def queue(self, application_id: int, *, mode: str = "MISSING", fields: list[str] | None = None,
              max_pages: int = PAGE_BUDGET) -> int:
        if not 1 <= int(max_pages) <= PAGE_BUDGET:
            raise ValueError("Scan page budget must be between 1 and 12")
        targets = self.plan_targets(application_id, mode=mode, fields=fields)
        if mode == "MISSING" and not targets:
            raise ValueError("Nothing missing to scan")
        from phd_agent.operations import OperationService
        operation_type = SCAN_FULL if mode == "FULL" else SCAN_MISSING
        if self.active_operation(application_id, operation_type):
            raise ValueError("Scan already running")
        operation_id = OperationService(self.db_path).queue(operation_type, application_id=application_id)
        checkpoint = {"mode": mode, "targets": targets, "fields": list(fields or []), "page_budget": int(max_pages),
                      "pages_checked": [], "resolved": [], "needs_review": [], "conflicts": [],
                      "supplied": {}, "human_input": [], "planned_urls": [], "index": 0}
        now = utc_now()
        with transaction(self.db_path) as db:
            db.execute("UPDATE operations SET checkpoint_json=?,updated_at=? WHERE id=?",
                       (json.dumps(checkpoint), now, operation_id))
            db.execute("""INSERT INTO application_scan_reports
                (operation_id,application_id,mode,target_fields_json,requested_fields_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?)""",
                (operation_id, application_id, mode, json.dumps(targets), json.dumps(list(fields or [])), now, now))
            self._audit(db, application_id, "SCAN_STARTED", {"operation_id": operation_id, "mode": mode, "targets": targets})
        return operation_id

    def queue_selected(self, application_ids: list[int], *, mode: str = "MISSING") -> dict:
        queued, skipped = [], []
        for application_id in list(dict.fromkeys(application_ids)):
            try:
                queued.append(self.queue(application_id, mode=mode))
            except ValueError as error:
                skipped.append({"application_id": application_id, "reason": str(error)})
                if "already active" in str(error):
                    break
        return {"queued": queued, "skipped": skipped}

    def seed_urls(self, application_id: int) -> list[str]:
        record = self._record(application_id)
        urls = []
        for url in (record.get("programme_url"), record.get("admissions_url"), record.get("programme_portal"), record.get("portal_url"), record.get("opportunity_url")):
            if url and url not in urls:
                urls.append(url)
        return urls

    def history(self, application_id: int) -> list[dict]:
        with connect(self.db_path) as db:
            rows = db.execute("""SELECT r.*, o.status, o.created_at AS started
                FROM application_scan_reports r JOIN operations o ON o.id=r.operation_id
                WHERE r.application_id=? ORDER BY r.id DESC""", (application_id,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["summary"] = json.loads(item.pop("summary_json") or "{}")
            item["targets"] = json.loads(item.pop("target_fields_json") or "[]")
            result.append(item)
        return result

    def latest_report(self, application_id: int) -> dict | None:
        rows = self.history(application_id)
        return rows[0] if rows else None

    def conflicts(self, application_id: int, *, status: str = "OPEN") -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(row) for row in db.execute(
                """SELECT * FROM application_field_conflicts WHERE application_id=? AND status=? ORDER BY id""",
                (application_id, status))]

    def notifications(self, *, unread_only: bool = True) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM applicant_notifications WHERE (?=0 OR read_at IS NULL) ORDER BY id DESC",
                (int(unread_only),))]

    def mark_notifications_read(self, notification_ids: list[int] | None = None) -> None:
        with transaction(self.db_path) as db:
            if notification_ids is None:
                db.execute("UPDATE applicant_notifications SET read_at=? WHERE read_at IS NULL", (utc_now(),))
                return
            ids = list(dict.fromkeys(notification_ids))
            if not ids:
                return
            db.execute("UPDATE applicant_notifications SET read_at=? WHERE read_at IS NULL AND id IN (" +
                       ",".join("?" for _ in ids) + ")", (utc_now(), *ids))

    def supply(self, operation_id: int, url: str, content: str | bytes, *, method: str = "PASTED_TEXT") -> None:
        if method not in {"PASTED_TEXT", "UPLOADED_HTML", "UPLOADED_PDF"}:
            raise ValueError("Unsupported human-assisted ingestion method")
        from phd_agent.ledger import _url
        safe_url = _url(url)
        if method == "UPLOADED_PDF":
            from io import BytesIO
            from PyPDF2 import PdfReader
            data = content if isinstance(content, bytes) else content.encode()
            text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
            html = ""
        elif method == "UPLOADED_HTML":
            html = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
            text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
        else:
            text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
            html = ""
        if len(text.strip()) < 80:
            raise ValueError("Supply enough page content to identify the missing details")
        from phd_agent.operations import OperationService
        operation = OperationService(self.db_path).get(operation_id)
        if operation["operation_type"] not in {SCAN_MISSING, SCAN_FULL}:
            raise ValueError("Operation is not an application scan")
        checkpoint = operation["checkpoint"]
        checkpoint.setdefault("supplied", {})[safe_url] = {"text": text[:80_000], "html": html[:80_000]}
        if safe_url not in checkpoint.setdefault("planned_urls", []):
            checkpoint["planned_urls"].append(safe_url)
        record = self._record(operation["application_id"])
        with transaction(self.db_path) as db:
            if record.get("linked_programme_id") and not record.get("programme_url"):
                self.ledger.update_programme(record["linked_programme_id"], db=db, programme_url=safe_url)
            db.execute("UPDATE operations SET checkpoint_json=?,updated_at=? WHERE id=?",
                       (json.dumps(checkpoint), utc_now(), operation_id))
            self._audit(db, operation["application_id"], "SCAN_PAGE_SUPPLIED", {"operation_id": operation_id, "url": safe_url, "method": method})

    def resolve_conflict(self, conflict_id: int, action: str) -> None:
        if action not in {"accept", "keep", "unresolved"}:
            raise ValueError("Choose accept, keep, or unresolved")
        with transaction(self.db_path) as db:
            row = db.execute("SELECT * FROM application_field_conflicts WHERE id=? AND status='OPEN'", (conflict_id,)).fetchone()
            if not row:
                raise ValueError("Conflict is not open")
            status = {"accept": "ACCEPTED", "keep": "KEPT", "unresolved": "UNRESOLVED"}[action]
            now = utc_now()
            if action == "accept":
                self._store_state(db, row["application_id"], row["field_name"], row["new_value"], "OPERATOR_CONFIRMED",
                                  row["new_source"], row["evidence_id"], row["new_value"] or "", "APPLICANT_ACCEPTED",
                                  row["operation_id"], None)
                self._mirror(db, row["application_id"], row["field_name"], row["new_value"], row["new_source"], row["evidence_id"], confirmed=True)
                db.execute("""INSERT INTO record_field_reviews
                    (entity_type,entity_id,field_name,field_value,verification_state,source_url,recorded_at)
                    VALUES('APPLICATION',?,?,?,'OPERATOR_CONFIRMED',?,?)""",
                    (row["application_id"], row["field_name"], row["new_value"], row["new_source"], now))
            db.execute("UPDATE application_field_conflicts SET status=?,resolved_at=? WHERE id=?", (status, now, conflict_id))
            self._audit(db, row["application_id"], "CONFLICT_" + status, {
                "conflict_id": conflict_id, "field": row["field_name"], "old": row["old_value"], "new": row["new_value"]})

    def _store_state(self, db, application_id, name, value, state, source, evidence_id, excerpt, method, operation_id, route) -> None:
        now = utc_now()
        db.execute("""INSERT INTO application_field_states
            (application_id,field_name,field_value,state,source_url,evidence_id,excerpt,extraction_method,checked_at,operation_id,model_route,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(application_id,field_name) DO UPDATE SET
            field_value=excluded.field_value,state=excluded.state,source_url=excluded.source_url,
            evidence_id=excluded.evidence_id,excerpt=excluded.excerpt,extraction_method=excluded.extraction_method,
            checked_at=excluded.checked_at,operation_id=excluded.operation_id,model_route=excluded.model_route,
            updated_at=excluded.updated_at""",
            (application_id, name, value, state, source, evidence_id, excerpt, method, now, operation_id, route, now))

    def _evidence(self, db, url: str, excerpt: str) -> int:
        excerpt = excerpt.strip()[:500]
        digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest() if excerpt else None
        if digest:
            existing = db.execute("SELECT id FROM source_evidence WHERE canonical_url=? AND content_hash=?", (url, digest)).fetchone()
            if existing:
                return existing[0]
        return self.ledger.create_evidence(url, "PROGRAMME", excerpt, "UNVERIFIED", db=db)

    def _mirror(self, db, application_id: int, name: str, value: str, source: str | None, evidence_id: int | None, *, confirmed: bool) -> None:
        state = "VERIFIED" if confirmed else "UNVERIFIED"
        if name in {"deadline", "funding_deadline"} and value:
            deadline_type = "FUNDING" if name == "funding_deadline" else "APPLICATION"
            current = db.execute("SELECT id,due_at FROM deadlines WHERE application_id=? AND deadline_type=? ORDER BY id DESC LIMIT 1",
                                 (application_id, deadline_type)).fetchone()
            if current and not confirmed and current["due_at"][:10] != value[:10]:
                return
            if evidence_id is None and source:
                evidence_id = self._evidence(db, source, value)
            if current:
                changes = {"due_at": value[:10], "verification_state": state, "last_checked_at": utc_now()}
                if evidence_id:
                    changes["source_evidence_id"] = evidence_id
                self.ledger.update_deadline(current["id"], db=db, **changes)
            elif evidence_id:
                self.ledger.create_deadline(application_id, deadline_type, value[:10], evidence_id,
                                            verification_state=state, last_checked_at=utc_now(), db=db)
        elif name == "portal_url" and value:
            current = db.execute("SELECT portal_url FROM applications WHERE id=?", (application_id,)).fetchone()
            if current and not current["portal_url"]:
                self.ledger.update_application(application_id, db=db, portal_url=value)
            elif confirmed:
                self.ledger.update_application(application_id, db=db, portal_url=value)

    def _apply_parsed(self, db, application_id: int, operation_id: int, targets: list[str], parsed: dict[str, Parsed],
                      url: str, facts: dict[str, FieldFact], checkpoint: dict) -> str | None:
        for name in targets:
            if name in checkpoint["resolved"] or name not in parsed:
                continue
            item = parsed[name]
            fact = facts[name]
            high = fact.high_impact
            differs = bool(fact.value) and not _same(fact.value, item.value)
            protected_conflict = fact.protected and differs and not item.ambiguous
            decision = self.router.route(
                "CONFLICTING_REQUIREMENTS" if protected_conflict else "AMBIGUOUS_EXTRACTION" if item.ambiguous else "PAGE_EXTRACTION",
                conflicting_evidence=protected_conflict)
            if self._model_tier is None or TIERS.index(decision.tier) > TIERS.index(self._model_tier):
                self._model_tier = decision.tier
                self._model_label = f"{decision.provider}/{decision.model}"
            evidence_id = self._evidence(db, url, item.excerpt)
            now = utc_now()
            if fact.protected and not item.ambiguous and _same(fact.value, item.value):
                self._store_state(db, application_id, name, fact.value, fact.state, fact.source or url, evidence_id,
                                  item.excerpt, "DETERMINISTIC", operation_id, self._model_label)
                if name not in checkpoint["resolved"]:
                    checkpoint["resolved"].append(name)
                continue
            if item.ambiguous or (differs and (fact.protected or _known(fact))):
                existing = db.execute("""SELECT id FROM application_field_conflicts
                    WHERE application_id=? AND field_name=? AND status='OPEN'""", (application_id, name)).fetchone()
                if existing:
                    db.execute("""UPDATE application_field_conflicts SET new_value=?,new_source=?,new_checked_at=?,evidence_id=?,operation_id=?
                        WHERE id=?""", (item.value, url, now, evidence_id, operation_id, existing[0]))
                else:
                    db.execute("""INSERT INTO application_field_conflicts
                        (application_id,field_name,old_value,new_value,old_source,new_source,old_checked_at,new_checked_at,operation_id,evidence_id,created_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (application_id, name, fact.value, item.value, fact.source, url, fact.checked_at, now, operation_id, evidence_id, now))
                    self._audit(db, application_id, "CONFLICT_CREATED", {"field": name, "old": fact.value, "new": item.value, "operation_id": operation_id})
                if name not in checkpoint["conflicts"]:
                    checkpoint["conflicts"].append(name)
                if item.ambiguous and name not in checkpoint["needs_review"]:
                    checkpoint["needs_review"].append(name)
                continue
            state = "NEEDS_REVIEW" if high and fact.stale else "EXTRACTED"
            self._store_state(db, application_id, name, item.value, state, url, evidence_id, item.excerpt, "DETERMINISTIC", operation_id, self._model_label)
            db.execute("""INSERT INTO application_scan_findings
                (operation_id,application_id,field_name,field_value,state,source_url,evidence_id,excerpt,extraction_method,checked_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (operation_id, application_id, name, item.value, state, url, evidence_id, item.excerpt, "DETERMINISTIC", now))
            if not fact.value or not _known(fact):
                self._mirror(db, application_id, name, item.value, url, evidence_id, confirmed=False)
            checkpoint["resolved"].append(name)
            self._audit(db, application_id, "FIELD_EXTRACTED", {"field": name, "value": item.value, "operation_id": operation_id, "state": state})
        return self._model_label

    def _tick(self, service, operation_id: int, checkpoint: dict, *, stage: str, completed: int, total: int,
              event: str | None = None, item: str = "") -> bool:
        running = service._progress(operation_id, stage=stage, item=item, completed=completed, total=total,
                                    checkpoint=checkpoint, event=event, model=getattr(self, "_model_label", None))
        if running:
            return True
        with transaction(self.db_path) as db:
            current = db.execute("SELECT status FROM operations WHERE id=?", (operation_id,)).fetchone()
            if not current or current["status"] in {"INTERRUPTED", "CANCELLED", "COMPLETED", "FAILED"}:
                return False
            db.execute("""UPDATE operations SET checkpoint_json=?,completed_units=?,total_units=?,stage=?,current_item=?,updated_at=?
                WHERE id=?""", (json.dumps(checkpoint), completed, total, stage, item[:300], utc_now(), operation_id))
            if event:
                db.execute("INSERT INTO operation_events(operation_id,event_text,created_at) VALUES(?,?,?)",
                           (operation_id, event[:500], utc_now()))
        return False

    def _save_report(self, operation_id: int, application_id: int, checkpoint: dict, institution: str, *, notify: bool) -> None:
        targets = checkpoint.get("targets") or []
        resolved = checkpoint.get("resolved") or []
        review = checkpoint.get("needs_review") or []
        conflicts = checkpoint.get("conflicts") or []
        current = self.facts(application_id)
        summary = {
            "targets": len(targets),
            "resolved": [{"field": name, "label": FIELDS[name]["label"], "value": current[name].value} for name in resolved if name in current],
            "needs_review": [{"field": name, "label": FIELDS[name]["label"]} for name in review],
            "conflicts": [{"field": name, "label": FIELDS[name]["label"]} for name in conflicts],
            "still_unknown": [{"field": name, "label": FIELDS[name]["label"]} for name in targets
                              if name not in resolved and name not in review and name not in conflicts],
            "human_input": checkpoint.get("human_input") or [],
            "sources_checked": len(checkpoint.get("pages_checked") or []),
        }
        message = f"{institution} scan completed — {len(resolved)} fields resolved, {len(review) + len(conflicts)} needs review."
        with transaction(self.db_path) as db:
            db.execute("UPDATE application_scan_reports SET summary_json=?,updated_at=? WHERE operation_id=?",
                       (json.dumps(summary), utc_now(), operation_id))
            if notify:
                db.execute("""INSERT INTO applicant_notifications(application_id,operation_id,message,created_at)
                    VALUES(?,?,?,?)""", (application_id, operation_id, message, utc_now()))

    def fetch(self, url: str) -> dict:
        from phd_agent.orchestration import ProgrammeOrchestrator
        page = ProgrammeOrchestrator(self.db_path).acquire_page(url)
        return {"status": page.status, "url": page.url, "text": page.text or "", "html": page.html or "", "reason": page.reason or ""}

    def execute(self, operation_id: int, service, fetcher=None) -> None:
        operation = service.get(operation_id)
        if operation["status"] != "RUNNING" or operation["operation_type"] not in {SCAN_MISSING, SCAN_FULL}:
            return
        checkpoint = operation["checkpoint"] or {}
        application_id = operation["application_id"]
        record = self._record(application_id)
        mode = checkpoint.get("mode") or ("FULL" if operation["operation_type"] == SCAN_FULL else "MISSING")
        if not checkpoint.get("targets"):
            checkpoint["targets"] = self.plan_targets(application_id, mode=mode, fields=checkpoint.get("fields") or None)
        checkpoint.setdefault("pages_checked", [])
        checkpoint.setdefault("resolved", [])
        checkpoint.setdefault("needs_review", [])
        checkpoint.setdefault("conflicts", [])
        checkpoint.setdefault("supplied", {})
        checkpoint.setdefault("human_input", [])
        checkpoint["index"] = 0
        budget = int(checkpoint.get("page_budget") or PAGE_BUDGET)
        targets = list(checkpoint["targets"])
        self._model_tier = None
        self._model_label = None
        if not self._tick(service, operation_id, checkpoint, stage="Prepare scan", completed=len(checkpoint["pages_checked"]),
                         total=budget, event="Prepare scan"):
            self._save_report(operation_id, application_id, checkpoint, record["institution"], notify=False)
            service._finish(operation_id, "CANCELLED", results=len(checkpoint["resolved"]))
            return
        if not checkpoint.get("budget_logged"):
            checkpoint["budget_logged"] = True
            if not self._tick(service, operation_id, checkpoint, stage="Load known official sources",
                             completed=len(checkpoint["pages_checked"]), total=budget, event=f"Page budget: {budget}"):
                self._save_report(operation_id, application_id, checkpoint, record["institution"], notify=False)
                service._finish(operation_id, "CANCELLED", results=len(checkpoint["resolved"]))
                return
        if not checkpoint.get("planned_urls"):
            checkpoint["planned_urls"] = self.seed_urls(application_id)
            if not self._tick(service, operation_id, checkpoint, stage="Load known official sources",
                             completed=0, total=budget, event="Load known official sources"):
                self._save_report(operation_id, application_id, checkpoint, record["institution"], notify=False)
                service._finish(operation_id, "CANCELLED", results=len(checkpoint["resolved"]))
                return
        fetcher = fetcher or self.fetch
        while checkpoint["index"] < len(checkpoint["planned_urls"]) and len(checkpoint["pages_checked"]) < budget:
            finished = set(checkpoint["resolved"]) | set(checkpoint["conflicts"]) | set(checkpoint["needs_review"])
            if set(targets) <= finished:
                self._tick(service, operation_id, checkpoint, stage="Extract target fields",
                           completed=len(checkpoint["pages_checked"]), total=budget, event="All targeted fields resolved")
                break
            url = checkpoint["planned_urls"][checkpoint["index"]]
            if url in checkpoint["pages_checked"]:
                checkpoint["index"] += 1
                continue
            if url in checkpoint["human_input"] and url not in checkpoint["supplied"]:
                checkpoint["index"] += 1
                continue
            stage = stage_for(url)
            if not self._tick(service, operation_id, checkpoint, stage=stage, item=url,
                             completed=len(checkpoint["pages_checked"]), total=budget):
                self._save_report(operation_id, application_id, checkpoint, record["institution"], notify=False)
                service._finish(operation_id, "CANCELLED", results=len(checkpoint["resolved"]))
                return
            supplied = checkpoint["supplied"].get(url)
            if supplied:
                page = {"status": "ACQUIRED", "url": url, "text": supplied.get("text") or "", "html": supplied.get("html") or ""}
            else:
                try:
                    page = fetcher(url)
                except (OSError, RuntimeError, ValueError):
                    page = {"status": "HUMAN_INPUT_REQUIRED", "url": url, "text": "", "html": ""}
            if page.get("status") != "ACQUIRED" or len(page.get("text") or "") < 80:
                if url not in checkpoint["human_input"]:
                    checkpoint["human_input"].append(url)
                checkpoint["index"] += 1
                self._tick(service, operation_id, checkpoint, stage=stage, item=url,
                           completed=len(checkpoint["pages_checked"]), total=budget, event=HUMAN_MESSAGE)
                continue
            parsed = parse_page(page.get("text") or "", page.get("html") or "", page.get("url") or url)
            with transaction(self.db_path) as db:
                facts = self._facts(db, application_id)
                label = self._apply_parsed(db, application_id, operation_id, targets, parsed, page.get("url") or url, facts, checkpoint)
            if label:
                self._model_label = label
            for link in official_links(page.get("html") or "", page.get("url") or url, limit=budget):
                if link not in checkpoint["planned_urls"] and len(checkpoint["planned_urls"]) < budget:
                    checkpoint["planned_urls"].append(link)
            checkpoint["pages_checked"].append(url)
            checkpoint["index"] += 1
            checked = len(checkpoint["pages_checked"])
            if not self._tick(service, operation_id, checkpoint, stage="Extract target fields", item=url, completed=checked, total=budget,
                             event=f"{checked}/{budget} pages checked"):
                self._save_report(operation_id, application_id, checkpoint, record["institution"], notify=False)
                service._finish(operation_id, "CANCELLED", results=len(checkpoint["resolved"]))
                return
            self._tick(service, operation_id, checkpoint, stage="Extract target fields", completed=checked, total=budget,
                       event=f"Extract target fields {len(checkpoint['resolved'])}/{len(targets)}")
        unresolved = [name for name in targets if name not in checkpoint["resolved"] and name not in checkpoint["conflicts"]]
        waiting = [url for url in checkpoint["human_input"] if url not in checkpoint["supplied"]]
        self._tick(service, operation_id, checkpoint, stage="Validate evidence", completed=len(checkpoint["pages_checked"]),
                   total=budget, event="Validate evidence")
        self._save_report(operation_id, application_id, checkpoint, record["institution"], notify=not (waiting and unresolved))
        if waiting and unresolved:
            with transaction(self.db_path) as db:
                current = db.execute("SELECT status FROM operations WHERE id=?", (operation_id,)).fetchone()
                if not current or current["status"] != "RUNNING":
                    service._finish(operation_id, "CANCELLED", results=len(checkpoint["resolved"]))
                    return
                db.execute("""UPDATE operations SET status='PAUSED',stage='Human input required',current_item=?,checkpoint_json=?,updated_at=?
                    WHERE id=?""", ((waiting[-1] or "")[:300], json.dumps(checkpoint), utc_now(), operation_id))
                db.execute("INSERT INTO operation_events(operation_id,event_text,created_at) VALUES(?,?,?)",
                           (operation_id, HUMAN_MESSAGE, utc_now()))
            return
        self._tick(service, operation_id, checkpoint, stage="Save findings", completed=len(checkpoint["pages_checked"]),
                   total=budget, event="Save findings")
        service._finish(operation_id, "COMPLETED", results=len(checkpoint["resolved"]))
