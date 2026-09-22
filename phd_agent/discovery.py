"""Manual-first, source-backed discovery and conservative verification."""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from phd_agent.config import FRESHNESS_DAYS
from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.ledger import Ledger


TARGET_STATES = ("ACTIVE", "CONSIDERING", "PAUSED", "ARCHIVED")
FACULTY_STATES = ("VERIFIED", "PARTIALLY_VERIFIED", "NEEDS_REVERIFICATION", "CONFLICT", "INACTIVE", "UNKNOWN")
SOURCE_TYPES = ("PROGRAMME", "OPPORTUNITY", "FACULTY", "LAB", "FUNDING", "OTHER")
FACULTY_FACTS = {"IDENTITY", "AFFILIATION", "TITLE", "DEPARTMENT", "LAB", "EMAIL", "TOPICS", "SUPERVISION"}
OPPORTUNITY_TYPES = ("ADVERTISED_POSITION", "PROGRAMME_APPLICATION", "FACULTY_ENQUIRY")


def canonical_url(url: str) -> str:
    value = url.strip()
    parts = urlparse(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("A public http(s) source URL is required")
    if parts.hostname.casefold() == "localhost" or parts.hostname.casefold().endswith(".localhost"):
        raise ValueError("Localhost cannot be a discovery source")
    try:
        if not ipaddress.ip_address(parts.hostname).is_global:
            raise ValueError("A public source URL is required")
    except ValueError as error:
        if "public source URL" in str(error):
            raise
    return urlunparse((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", "", parts.query, ""))


def _safe_canonical_url(url: str | None) -> str | None:
    if not url or not str(url).strip():
        return None
    try:
        return canonical_url(url)
    except ValueError:
        return None


def freshness(retrieved_at: str | None, fact_kind: str, *, now: datetime | None = None) -> str:
    if not retrieved_at:
        return "UNKNOWN"
    if fact_kind not in FRESHNESS_DAYS:
        raise ValueError("Unknown freshness policy")
    checked = datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
    if checked.tzinfo is None:
        checked = checked.replace(tzinfo=timezone.utc)
    age = (now or datetime.now(timezone.utc)) - checked
    return "STALE" if age > timedelta(days=FRESHNESS_DAYS[fact_kind]) else "CURRENT"


def _name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Discovery:
    def __init__(self, db_path: Path | str, session=None):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.session = session or requests.Session()
        self.ledger = Ledger(self.db_path)

    def import_targets(self, csv_path: Path | str) -> int:
        """Import historical preferences as CONSIDERING, never current applications."""
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        count = 0
        for row in rows:
            name = (row.get("University Name") or "").strip()
            if not name:
                continue
            with transaction(self.db_path) as db:
                if db.execute("SELECT 1 FROM target_institutions WHERE name=?", (name,)).fetchone():
                    continue
                now = utc_now()
                db.execute("""INSERT INTO target_institutions
                    (name,state,departments,legacy_priority,origin,notes,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?)""", (
                    name, "CONSIDERING", (row.get("Departments to Search") or "").strip(),
                    (row.get("Priority") or "").strip(), "LEGACY_CSV",
                    (row.get("Notes") or "").strip(), now, now,
                ))
                count += 1
        return count

    def add_target(self, name: str, *, state: str = "CONSIDERING", departments: str = "", notes: str = "") -> int:
        if not name.strip() or state not in TARGET_STATES:
            raise ValueError("Target name and valid state are required")
        now = utc_now()
        with transaction(self.db_path) as db:
            return db.execute("""INSERT INTO target_institutions
                (name,state,departments,origin,notes,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?)""", (name.strip(), state, departments, "MANUAL", notes, now, now)).lastrowid

    def set_target_state(self, target_id: int, state: str) -> None:
        if state not in TARGET_STATES:
            raise ValueError("Invalid target state")
        with transaction(self.db_path) as db:
            if db.execute("UPDATE target_institutions SET state=?,updated_at=? WHERE id=?",
                          (state, utc_now(), target_id)).rowcount != 1:
                raise ValueError("Target not found")

    def list_targets(self) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute("SELECT * FROM target_institutions ORDER BY name")]

    def add_source(self, university: str, source_type: str, url: str, *, target_id: int | None = None,
                   department: str | None = None, strategy: str = "MANUAL", enabled: bool = True,
                   notes: str = "") -> int:
        if source_type not in SOURCE_TYPES or strategy not in {"MANUAL", "STATIC_HTML"}:
            raise ValueError("Invalid source type or discovery strategy")
        if not university.strip():
            raise ValueError("University is required")
        now = utc_now()
        with transaction(self.db_path) as db:
            return db.execute("""INSERT INTO source_catalogue
                (target_id,university,department,source_type,canonical_url,discovery_strategy,
                 enabled,notes,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)""", (
                target_id, university.strip(), department or None, source_type,
                canonical_url(url), strategy, int(enabled), notes, now, now,
            )).lastrowid

    def list_sources(self) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute("SELECT * FROM source_catalogue ORDER BY university,source_type,id")]

    def update_source(self, source_id: int, *, strategy: str | None = None,
                      enabled: bool | None = None, notes: str | None = None) -> None:
        if strategy is not None and strategy not in {"MANUAL", "STATIC_HTML"}:
            raise ValueError("Invalid discovery strategy")
        with transaction(self.db_path) as db:
            row = db.execute("SELECT * FROM source_catalogue WHERE id=?", (source_id,)).fetchone()
            if not row:
                raise ValueError("Catalogue source not found")
            db.execute("""UPDATE source_catalogue SET discovery_strategy=?,enabled=?,notes=?,updated_at=?
                WHERE id=?""", (
                strategy if strategy is not None else row["discovery_strategy"],
                int(enabled) if enabled is not None else row["enabled"],
                notes if notes is not None else row["notes"], utc_now(), source_id,
            ))

    def snapshot_source(self, source_id: int, *, excerpt: str | None = None,
                        manually_verified: bool = False) -> int:
        with connect(self.db_path) as db:
            source = db.execute("SELECT * FROM source_catalogue WHERE id=?", (source_id,)).fetchone()
        if not source or not source["enabled"]:
            raise ValueError("Source is missing or disabled")
        full_text = None
        if source["discovery_strategy"] == "STATIC_HTML":
            response = self.session.get(source["canonical_url"], timeout=20,
                                        headers={"User-Agent": "PhD-Apply-Agent/2.0 (research review)"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            full_text = soup.get_text(" ", strip=True)
            if excerpt is not None and re.sub(r"\s+", " ", excerpt).strip().casefold() not in full_text.casefold():
                raise ValueError("Reviewed excerpt was not found in the fetched source page")
            excerpt = excerpt or full_text[:6000]
        elif excerpt is None:
            raise ValueError("Enter a reviewed excerpt for a manual source")
        excerpt = excerpt.strip()
        if not excerpt:
            raise ValueError("Source excerpt cannot be empty")
        now = utc_now()
        digest = hashlib.sha256((full_text or excerpt).encode("utf-8")).hexdigest()
        with transaction(self.db_path) as db:
            evidence_id = db.execute("""INSERT INTO source_evidence
                (canonical_url,source_type,retrieved_at,relevant_excerpt,content_hash,
                 verification_state,last_manually_verified_at,created_at)
                VALUES(?,?,?,?,?,?,?,?)""", (
                source["canonical_url"], source["source_type"], now, excerpt, digest,
                "VERIFIED" if manually_verified else "UNVERIFIED",
                now if manually_verified else None, now,
            )).lastrowid
            if manually_verified:
                db.execute("UPDATE source_catalogue SET last_verified_at=?,updated_at=? WHERE id=?",
                           (now, now, source_id))
            return evidence_id

    def get_evidence(self, evidence_id: int) -> dict:
        row = self.ledger.get("source_evidence", evidence_id)
        if not row:
            raise ValueError("Evidence does not exist")
        return row

    def add_faculty_candidate(self, source_id: int, name: str, evidence_id: int,
                              profile_url: str | None = None, department: str | None = None) -> int:
        evidence = self.get_evidence(evidence_id)
        with transaction(self.db_path) as db:
            source = db.execute("SELECT * FROM source_catalogue WHERE id=?", (source_id,)).fetchone()
            if not source or source["source_type"] not in {"FACULTY", "LAB"}:
                raise ValueError("A faculty/lab catalogue source is required")
            if evidence["canonical_url"] != source["canonical_url"]:
                raise ValueError("Candidate evidence must come from the selected source")
            return db.execute("""INSERT INTO faculty_candidates
                (source_catalogue_id,name,institution,department,profile_url,source_evidence_id,created_at)
                VALUES(?,?,?,?,?,?,?)""", (
                source_id, name.strip(), source["university"], department,
                canonical_url(profile_url) if profile_url else None, evidence_id, utc_now(),
            )).lastrowid

    def list_faculty_candidates(self) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute("""SELECT * FROM faculty_candidates
                ORDER BY review_state,id DESC""")]

    def review_faculty_candidate(self, candidate_id: int, *, dismiss: bool = False) -> int | None:
        with connect(self.db_path) as db:
            candidate = db.execute("SELECT * FROM faculty_candidates WHERE id=?", (candidate_id,)).fetchone()
        if not candidate or candidate["review_state"] != "NEW":
            raise ValueError("Candidate is missing or already reviewed")
        if dismiss:
            with transaction(self.db_path) as db:
                db.execute("UPDATE faculty_candidates SET review_state='DISMISSED' WHERE id=?", (candidate_id,))
            return None
        return self.create_faculty(candidate["name"], candidate["institution"],
                                   profile_url=candidate["profile_url"], department=candidate["department"],
                                   evidence_id=candidate["source_evidence_id"], candidate_id=candidate_id)

    def duplicate_signals(self, name: str, institution: str, profile_url: str | None = None) -> list[dict]:
        requested_url = _safe_canonical_url(profile_url)
        with connect(self.db_path) as db:
            rows = [dict(r) for r in db.execute("SELECT id,name,institution,official_profile_url,email FROM faculty_profiles")]
        result = []
        for row in rows:
            exact_url = bool(requested_url and _safe_canonical_url(row["official_profile_url"]) == requested_url)
            same_name = _name(row["name"]) == _name(name)
            same_institution = _name(row["institution"]) == _name(institution)
            if exact_url or same_name:
                result.append({**row, "signal": "EXACT_URL" if exact_url else
                               "SAME_NAME_INSTITUTION" if same_institution else "SAME_NAME_OTHER_INSTITUTION"})
        return result

    def scan_historical_duplicates(self) -> int:
        """Queue exact URL/name-institution pairs for review, never merge records."""
        with connect(self.db_path) as db:
            rows = [dict(r) for r in db.execute("""SELECT id,name,institution,official_profile_url
                FROM faculty_profiles WHERE legacy_professor_id IS NOT NULL ORDER BY id""")]
        created = 0
        now = utc_now()
        with transaction(self.db_path) as db:
            for index, first in enumerate(rows):
                for second in rows[index + 1:]:
                    first_url, second_url = _safe_canonical_url(first["official_profile_url"]), _safe_canonical_url(second["official_profile_url"])
                    same_url = bool(first_url and first_url == second_url)
                    same_identity = (_name(first["name"]) == _name(second["name"]) and
                                     _name(first["institution"]) == _name(second["institution"]))
                    if not (same_url or same_identity):
                        continue
                    cursor = db.execute("""INSERT OR IGNORE INTO faculty_duplicate_reviews
                        (faculty_profile_a_id,faculty_profile_b_id,reason,created_at)
                        VALUES(?,?,?,?)""", (
                        first["id"], second["id"], "EXACT_URL" if same_url else "SAME_NAME_INSTITUTION", now,
                    ))
                    created += cursor.rowcount
        return created

    def list_duplicate_reviews(self, *, pending_only: bool = False) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute("""SELECT r.*, a.name AS name_a,a.institution AS institution_a,
                b.name AS name_b,b.institution AS institution_b
                FROM faculty_duplicate_reviews r
                JOIN faculty_profiles a ON a.id=r.faculty_profile_a_id
                JOIN faculty_profiles b ON b.id=r.faculty_profile_b_id
                WHERE (?=0 OR r.review_state='PENDING') ORDER BY r.id""", (int(pending_only),))]

    def resolve_duplicate_review(self, review_id: int, decision: str) -> None:
        if decision not in {"DISTINCT", "DUPLICATE"}:
            raise ValueError("Choose distinct or duplicate")
        with transaction(self.db_path) as db:
            if db.execute("""UPDATE faculty_duplicate_reviews SET review_state=?,resolved_at=?
                WHERE id=? AND review_state='PENDING'""", (decision, utc_now(), review_id)).rowcount != 1:
                raise ValueError("Duplicate review is missing or already decided")

    def create_faculty(self, name: str, institution: str, *, profile_url: str | None = None,
                       department: str | None = None, evidence_id: int | None = None,
                       candidate_id: int | None = None) -> int:
        """Create a faculty profile and carry forward any reviewed candidate state."""
        if not name.strip() or not institution.strip():
            raise ValueError("Name and institution are required")
        signals = self.duplicate_signals(name, institution, profile_url)
        if any(s["signal"] in {"EXACT_URL", "SAME_NAME_INSTITUTION"} for s in signals):
            raise ValueError("Possible existing faculty identity: review the matching record")
        now = utc_now()
        with transaction(self.db_path) as db:
            faculty_id = db.execute("""INSERT INTO faculty_profiles
                (name,institution,department,official_profile_url,created_at,updated_at)
                VALUES(?,?,?,?,?,?)""", (
                name.strip(), institution.strip(), department,
                canonical_url(profile_url) if profile_url else None, now, now,
            )).lastrowid
            for match in signals:
                a, b = sorted((faculty_id, match["id"]))
                db.execute("""INSERT OR IGNORE INTO faculty_duplicate_reviews
                    (faculty_profile_a_id,faculty_profile_b_id,reason,created_at)
                    VALUES(?,?,?,?)""", (a, b, match["signal"], now))
            if evidence_id:
                db.execute("""INSERT INTO faculty_evidence_links
                    (faculty_profile_id,fact_type,source_evidence_id,linked_at)
                    VALUES(?,?,?,?)""", (faculty_id, "IDENTITY", evidence_id, now))
            if candidate_id:
                db.execute("""UPDATE faculty_candidates SET review_state='REVIEWED',faculty_profile_id=?
                    WHERE id=?""", (faculty_id, candidate_id))
                db.execute("UPDATE faculty_research_snapshots SET faculty_profile_id=? WHERE candidate_id=?",
                           (faculty_id, candidate_id))
                for decision in db.execute("SELECT * FROM faculty_decisions WHERE candidate_id=?",
                                           (candidate_id,)).fetchall():
                    if not db.execute("""SELECT 1 FROM faculty_decisions WHERE application_id=?
                        AND faculty_profile_id=?""", (decision["application_id"], faculty_id)).fetchone():
                        db.execute("UPDATE faculty_decisions SET faculty_profile_id=? WHERE id=?",
                                   (faculty_id, decision["id"]))
                    if decision["state"] == "PURSUE":
                        db.execute("""INSERT OR IGNORE INTO application_faculty
                            (application_id,faculty_profile_id,linked_at) VALUES(?,?,?)""",
                            (decision["application_id"], faculty_id, now))
            return faculty_id

    def verify_faculty(self, faculty_id: int, state: str, *, evidence_by_fact: dict[str, int],
                       reviewer: str, reason: str, updates: dict | None = None) -> None:
        if state not in FACULTY_STATES or not reviewer.strip() or not reason.strip():
            raise ValueError("Valid state, reviewer, and reason are required")
        if set(evidence_by_fact) - FACULTY_FACTS:
            raise ValueError("Unsupported faculty fact type")
        updates = updates or {}
        allowed = {"name", "institution", "department", "official_title", "lab", "official_profile_url",
                   "lab_url", "email", "email_state", "research_topics", "affiliation_state", "supervision_state", "notes"}
        if set(updates) - allowed:
            raise ValueError("Unsupported faculty update")
        if state == "VERIFIED" and not {"IDENTITY", "AFFILIATION", "TOPICS"}.issubset(evidence_by_fact):
            raise ValueError("Full verification requires identity, affiliation, and topics evidence")
        if updates.get("email_state") == "VERIFIED" and "EMAIL" not in evidence_by_fact:
            raise ValueError("Verified email needs its own evidence")
        if updates.get("supervision_state") in {"OPEN", "CLOSED"} and "SUPERVISION" not in evidence_by_fact:
            raise ValueError("Supervision state needs explicit source evidence")
        if updates.get("affiliation_state") == "CURRENT" and "AFFILIATION" not in evidence_by_fact:
            raise ValueError("Current affiliation needs evidence")
        for field in ("official_profile_url", "lab_url"):
            if updates.get(field):
                updates[field] = canonical_url(updates[field])
        evidence = {fact: self.get_evidence(eid) for fact, eid in evidence_by_fact.items()}
        if any(item["source_type"] not in {"FACULTY", "LAB", "PROGRAMME", "OPPORTUNITY"}
               for item in evidence.values()):
            raise ValueError("Faculty verification needs an institutional source snapshot")
        if state in {"VERIFIED", "PARTIALLY_VERIFIED"} and not evidence:
            raise ValueError("Verification needs source evidence")
        if state == "VERIFIED" and any(r["verification_state"] != "VERIFIED" for r in evidence.values()):
            raise ValueError("Full verification requires manually reviewed source snapshots")
        now = utc_now()
        with transaction(self.db_path) as db:
            old = db.execute("SELECT * FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
            if not old:
                raise ValueError("Faculty record not found")
            for fact, eid in evidence_by_fact.items():
                db.execute("""INSERT OR IGNORE INTO faculty_evidence_links
                    (faculty_profile_id,fact_type,source_evidence_id,linked_at,notes)
                    VALUES(?,?,?,?,?)""", (faculty_id, fact, eid, now, reviewer.strip()))
            updates = {**updates, "verification_state": state, "last_checked_at": now, "updated_at": now}
            db.execute("UPDATE faculty_profiles SET " + ",".join(f"{k}=?" for k in updates) + " WHERE id=?",
                       (*updates.values(), faculty_id))
            db.execute("""INSERT INTO faculty_verification_events
                (faculty_profile_id,previous_state,new_state,reason,evidence_ids_json,changed_at)
                VALUES(?,?,?,?,?,?)""", (
                faculty_id, old["verification_state"], state,
                f"{reviewer.strip()}: {reason.strip()}", _json(sorted(set(evidence_by_fact.values()))), now,
            ))

    def list_faculty(self, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM faculty_profiles ORDER BY verification_state,name"
        if limit:
            sql += " LIMIT ?"
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute(sql, (limit,) if limit else ())]

    def faculty_detail(self, faculty_id: int) -> dict:
        with connect(self.db_path) as db:
            faculty = db.execute("SELECT * FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
            if not faculty:
                raise ValueError("Faculty record not found")
            links = [dict(r) for r in db.execute("""SELECT l.fact_type,l.notes,e.*
                FROM faculty_evidence_links l JOIN source_evidence e ON e.id=l.source_evidence_id
                WHERE l.faculty_profile_id=? ORDER BY e.retrieved_at DESC""", (faculty_id,))]
            works = [dict(r) for r in db.execute("SELECT * FROM publications WHERE faculty_profile_id=? ORDER BY year DESC", (faculty_id,))]
            events = [dict(r) for r in db.execute("SELECT * FROM faculty_verification_events WHERE faculty_profile_id=? ORDER BY id DESC", (faculty_id,))]
            assessments = [dict(r) for r in db.execute("""SELECT * FROM match_assessments
                WHERE faculty_profile_id=? ORDER BY id DESC""", (faculty_id,))]
        result = dict(faculty)
        result["evidence"] = [{**link, "freshness": freshness(link["retrieved_at"],
            "FACULTY_EMAIL" if link["fact_type"] == "EMAIL" else "OPPORTUNITY_OPENING"
            if link["fact_type"] == "SUPERVISION" else "FACULTY_AFFILIATION")} for link in links]
        result["publications"] = works
        result["history"] = events
        result["assessments"] = assessments
        return result

    def add_opportunity(self, opportunity_type: str, title: str, institution: str,
                        evidence_id: int, *, opening_status: str = "UNKNOWN", **details) -> int:
        if opportunity_type not in OPPORTUNITY_TYPES or opening_status not in {"OPEN", "CLOSED", "UNKNOWN"}:
            raise ValueError("Invalid opportunity type or state")
        evidence = self.get_evidence(evidence_id)
        if evidence["source_type"] not in {"PROGRAMME", "OPPORTUNITY", "FACULTY", "LAB", "FUNDING"}:
            raise ValueError("Opportunity must use an institutional source")
        if opening_status != "UNKNOWN" and evidence["verification_state"] != "VERIFIED":
            raise ValueError("An open/closed state requires a reviewed source snapshot")
        if opportunity_type == "ADVERTISED_POSITION" and opening_status == "OPEN" and not details.get("application_route"):
            raise ValueError("An advertised open position needs an application route")
        return self.ledger.create_opportunity(
            opportunity_type, title, institution, source_evidence_id=evidence_id,
            canonical_url=evidence["canonical_url"], opening_status=opening_status,
            verification_state="VERIFIED" if evidence["verification_state"] == "VERIFIED" else "UNVERIFIED",
            last_checked_at=evidence["retrieved_at"], **details,
        )

    def link_to_application(self, application_id: int, *, programme_id: int | None = None,
                            opportunity_id: int | None = None, faculty_id: int | None = None,
                            evidence_id: int | None = None) -> str:
        """Only fill empty links. Create a review task for conflicting entered values."""
        if not any((programme_id, opportunity_id, faculty_id)):
            raise ValueError("Select a programme, opportunity, or faculty record")
        if evidence_id:
            self.get_evidence(evidence_id)
        now = utc_now()
        conflicts = []
        with transaction(self.db_path) as db:
            app = db.execute("SELECT * FROM applications WHERE id=?", (application_id,)).fetchone()
            if not app:
                raise ValueError("Application not found")
            if programme_id and not db.execute("SELECT 1 FROM programmes WHERE id=?", (programme_id,)).fetchone():
                raise ValueError("Programme not found")
            opportunity = db.execute("SELECT * FROM opportunities WHERE id=?", (opportunity_id,)).fetchone() if opportunity_id else None
            if opportunity_id and not opportunity:
                raise ValueError("Opportunity not found")
            if faculty_id and not db.execute("SELECT 1 FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone():
                raise ValueError("Faculty not found")
            if opportunity and programme_id and opportunity["programme_id"] and opportunity["programme_id"] != programme_id:
                conflicts.append("opportunity programme differs from selected programme")
            if opportunity and app["programme_id"] and opportunity["programme_id"] and app["programme_id"] != opportunity["programme_id"]:
                conflicts.append("opportunity programme differs from application programme")
            current_programme = db.execute("SELECT university FROM programmes WHERE id=?", (app["programme_id"],)).fetchone() if app["programme_id"] else None
            current_opportunity = db.execute("SELECT institution FROM opportunities WHERE id=?", (app["opportunity_id"],)).fetchone() if app["opportunity_id"] else None
            current_institution = current_programme["university"] if current_programme else current_opportunity["institution"] if current_opportunity else None
            new_programme = db.execute("SELECT university FROM programmes WHERE id=?", (programme_id,)).fetchone() if programme_id else None
            new_institution = new_programme["university"] if new_programme else opportunity["institution"] if opportunity else None
            if current_institution and new_institution and _name(current_institution) != _name(new_institution):
                conflicts.append(f"institution differs: current {current_institution}, discovered {new_institution}")
            for field, new_id in (("programme_id", programme_id), ("opportunity_id", opportunity_id)):
                if new_id and app[field] and app[field] != new_id:
                    conflicts.append(f"{field} differs: current #{app[field]}, discovered #{new_id}")
            if conflicts:
                db.execute("""INSERT INTO application_tasks
                    (application_id,task_type,description,status,priority,source_context,created_at)
                    VALUES(?,?,?,?,?,?,?)""", (
                    application_id, "DISCOVERY_CONFLICT", "; ".join(conflicts), "TODO", "HIGH",
                    f"Evidence #{evidence_id}" if evidence_id else "Discovery review", now,
                ))
                return "CONFLICT_TASK_CREATED"
            if programme_id and not app["programme_id"]:
                db.execute("UPDATE applications SET programme_id=?,updated_at=? WHERE id=?", (programme_id, now, application_id))
            if opportunity_id and not app["opportunity_id"]:
                db.execute("UPDATE applications SET opportunity_id=?,updated_at=? WHERE id=?", (opportunity_id, now, application_id))
            if faculty_id:
                db.execute("""INSERT OR IGNORE INTO application_faculty
                    (application_id,faculty_profile_id,source_evidence_id,linked_at)
                    VALUES(?,?,?,?)""", (application_id, faculty_id, evidence_id, now))
        return "LINKED"

    def create_application_from_opportunity(self, opportunity_id: int, cycle: str) -> int:
        opportunity = self.ledger.get("opportunities", opportunity_id)
        if not opportunity or opportunity["verification_state"] != "VERIFIED":
            raise ValueError("A reviewed opportunity is required")
        return self.ledger.create_application(cycle, programme_id=opportunity["programme_id"],
                                              opportunity_id=opportunity_id,
                                              next_action="Review route, eligibility, deadline and requirements")

    def propose_deadline(self, application_id: int, due_at: str, evidence_id: int,
                         *, deadline_type: str = "APPLICATION", timezone_name: str | None = None) -> str:
        self.get_evidence(evidence_id)
        with connect(self.db_path) as db:
            existing = [dict(r) for r in db.execute(
                "SELECT * FROM deadlines WHERE application_id=? AND deadline_type=?", (application_id, deadline_type))]
        if any(row["due_at"] == due_at and row["timezone"] == timezone_name for row in existing):
            return "ALREADY_RECORDED"
        if existing:
            self.ledger.create_task(application_id, "DISCOVERY_CONFLICT",
                f"Discovered {deadline_type} deadline {due_at} differs from entered deadline(s): " +
                ", ".join(row["due_at"] for row in existing), priority="HIGH",
                source_context=f"Evidence #{evidence_id}")
            return "CONFLICT_TASK_CREATED"
        self.ledger.create_deadline(application_id, deadline_type, due_at, evidence_id,
                                    timezone=timezone_name)
        return "ADDED"

    def propose_requirement(self, application_id: int, context: str, label: str,
                            state: str, evidence_id: int, **details) -> str:
        self.get_evidence(evidence_id)
        with connect(self.db_path) as db:
            existing = [dict(r) for r in db.execute(
                "SELECT * FROM requirements WHERE application_id=? AND context=?", (application_id, context))]
        comparable = [r for r in existing if _name(r["original_label"]) == _name(label)]
        if comparable:
            row = comparable[0]
            if row["requirement_state"] == state and all(row.get(k) == v for k, v in details.items()):
                return "ALREADY_RECORDED"
            self.ledger.create_task(application_id, "DISCOVERY_CONFLICT",
                f"Discovered requirement '{label}' differs from entered requirement #{row['id']}",
                priority="HIGH", source_context=f"Evidence #{evidence_id}")
            return "CONFLICT_TASK_CREATED"
        self.ledger.create_requirement(application_id, context, state, label, evidence_id, **details)
        return "ADDED"

    def assess(self, faculty_id: int, *, application_id: int | None = None,
               track_version_id: int | None = None, research_fit: float | None = None,
               application_readiness: float | None = None, research_components: dict | None = None,
               readiness_components: dict | None = None, unknowns: list[str] | None = None,
               evidence_ids: list[int] | None = None, notes: str = "") -> int:
        if research_fit is None and application_readiness is None and not unknowns:
            raise ValueError("Record an assessment or explicit unknowns")
        for score in (research_fit, application_readiness):
            if score is not None and not 0 <= score <= 10:
                raise ValueError("Scores must be 0–10 or unknown")
        for eid in evidence_ids or []:
            self.get_evidence(eid)
        with transaction(self.db_path) as db:
            return db.execute("""INSERT INTO match_assessments
                (faculty_profile_id,application_id,research_track_version_id,research_fit,
                 application_readiness,research_fit_components_json,
                 application_readiness_components_json,unknowns_json,evidence_ids_json,assessed_at,notes)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (
                faculty_id, application_id, track_version_id, research_fit,
                application_readiness, _json(research_components or {}),
                _json(readiness_components or {}), _json(unknowns or []),
                _json(evidence_ids or []), utc_now(), notes,
            )).lastrowid

    def track_overlap(self, faculty_id: int, profile_id: int) -> list[dict]:
        """Explainable lexical hints only; no admission or fit score is inferred."""
        with connect(self.db_path) as db:
            faculty = db.execute("SELECT research_topics FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
            tracks = [dict(r) for r in db.execute("""SELECT t.id,t.title,v.id AS version_id,
                v.approval_state,v.research_problem,v.proposed_methodology,v.supporting_claim_revision_ids_json
                FROM research_tracks t JOIN research_track_versions v ON v.track_id=t.id
                WHERE t.profile_id=? AND t.status='ACTIVE'
                AND v.version_number=(SELECT MAX(x.version_number) FROM research_track_versions x WHERE x.track_id=t.id)""", (profile_id,))]
        if not faculty:
            raise ValueError("Faculty not found")
        topics = set(re.findall(r"[a-z]{4,}", (faculty["research_topics"] or "").casefold()))
        common_words = {"research", "systems", "learning", "model", "models", "study", "methods"}
        return [{"track_id": t["id"], "track_version_id": t["version_id"], "title": t["title"],
                 "approval_state": t["approval_state"],
                 "shared_terms": sorted((topics & set(re.findall(r"[a-z]{4,}",
                     (t["title"] + " " + t["research_problem"] + " " + t["proposed_methodology"]).casefold()))) - common_words),
                 "supporting_claim_revision_ids": json.loads(t["supporting_claim_revision_ids_json"])}
                for t in tracks]

    def publication_relevance(self, publication_id: int, profile_id: int) -> list[dict]:
        """Transparent term overlaps for review; no semantic or admission score."""
        with connect(self.db_path) as db:
            work = db.execute("SELECT * FROM publications WHERE id=?", (publication_id,)).fetchone()
            tracks = [dict(r) for r in db.execute("""SELECT t.title,v.* FROM research_tracks t
                JOIN research_track_versions v ON v.track_id=t.id
                WHERE t.profile_id=? AND t.status='ACTIVE' AND v.version_number=
                    (SELECT MAX(x.version_number) FROM research_track_versions x WHERE x.track_id=t.id)""", (profile_id,))]
            claims = [dict(r) for r in db.execute("""SELECT cr.id,cr.claim_text FROM claim_revisions cr
                JOIN claims c ON c.id=cr.claim_id WHERE c.profile_id=? AND cr.review_status='APPROVED'
                AND cr.version_number=(SELECT MAX(x.version_number) FROM claim_revisions x WHERE x.claim_id=c.id)""", (profile_id,))]
        if not work:
            raise ValueError("Publication not found")
        stop = {"research", "learning", "model", "models", "method", "methods",
                "study", "using", "based", "systems", "tasks"}
        terms = lambda value: set(re.findall(r"[a-z]{4,}", (value or "").casefold())) - stop
        content = terms(work["title"] + " " + " ".join(json.loads(work["topics_json"])))
        result = []
        for track in tracks:
            topic = sorted(content & terms(track["title"] + " " + track["research_problem"]))
            method = sorted(content & terms(track["proposed_methodology"]))
            claim_ids = [c["id"] for c in claims if content & terms(c["claim_text"])]
            result.append({"publication_id": publication_id, "track_version_id": track["id"],
                           "track": track["title"], "track_state": track["approval_state"],
                           "topic_overlap": topic, "method_overlap": method,
                           "possible_claim_revision_ids": claim_ids})
        return result
