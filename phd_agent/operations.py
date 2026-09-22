"""SQLite-backed, cooperative background discovery for the single-service deployment."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.discovery import canonical_url
from phd_agent.faculty_research import FacultyResearch, extract_official_research
from phd_agent.official_search import OfficialSearchResult
from phd_agent.orchestration import ProgrammeOrchestrator
from phd_agent.university_enrichment import COUNTRY_NAMES


MAX_ACTIVE_OPERATIONS = 2
MAX_SEARCH_PAGES = 12
SCAN_TYPES = {"APPLICATION_DETAIL_SCAN", "APPLICATION_FULL_REFRESH"}


def _query_hint(criteria: dict) -> str:
    countries = {code: canonical for name, (canonical, code) in COUNTRY_NAMES.items()
                 if name == canonical.casefold()}
    parts = []
    if criteria.get("country_codes"):
        parts.append("Countries: " + ", ".join(countries.get(code, code)
                                                    for code in criteria["country_codes"]))
    if criteria.get("programme_types"):
        parts.append("Programme types: " + ", ".join(value.replace("_", " ")
                                                         for value in criteria["programme_types"]))
    if criteria.get("funding") == "Funded":
        parts.append("Funding: funded positions preferred")
    if criteria.get("qs_max"):
        parts.append(f"QS rank preference: top {criteria['qs_max']}")
    return "; ".join(parts)


class OperationService:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path).resolve()
        migrate(self.db_path)

    def _event(self, db, operation_id: int, message: str) -> None:
        db.execute("INSERT INTO operation_events(operation_id,event_text,created_at) VALUES(?,?,?)",
                   (operation_id, message[:500], utc_now()))

    def create_search(self, title: str, intent: str, context_id: int, criteria: dict) -> dict:
        if not title.strip():
            raise ValueError("Name this search")
        allowed = {"country_codes", "programme_types", "funding", "qs_max", "min_research_fit", "max_pages"}
        if set(criteria) - allowed:
            raise ValueError("Unsupported search criteria")
        budget = int(criteria.get("max_pages") or MAX_SEARCH_PAGES)
        if budget < 1 or budget > MAX_SEARCH_PAGES:
            raise ValueError("Search page budget must be between 1 and 12")
        criteria = {**criteria, "max_pages": budget}
        intent_row = ProgrammeOrchestrator(self.db_path).create_intent(intent, context_id)
        now = utc_now()
        with transaction(self.db_path) as db:
            search_id = db.execute("""INSERT INTO search_sessions
                (title,intent_id,context_id,criteria_json,created_at,updated_at) VALUES(?,?,?,?,?,?)""",
                (title.strip(), intent_row["id"], context_id, json.dumps(criteria), now, now)).lastrowid
        return self.get_search(search_id)

    def get_search(self, search_id: int) -> dict:
        with connect(self.db_path) as db:
            row = db.execute("""SELECT s.*,i.intent_text FROM search_sessions s
                JOIN discovery_intents i ON i.id=s.intent_id WHERE s.id=?""", (search_id,)).fetchone()
        if not row:
            raise ValueError("Search does not exist")
        result = dict(row)
        result["criteria"] = json.loads(result.pop("criteria_json"))
        return result

    def list_searches(self, include_archived: bool = False) -> list[dict]:
        with connect(self.db_path) as db:
            ids = [row[0] for row in db.execute(
                "SELECT id FROM search_sessions" + ("" if include_archived else " WHERE status='ACTIVE'") +
                " ORDER BY id DESC")]
        return [self.get_search(search_id) for search_id in ids]

    def update_search(self, search_id: int, *, title: str, criteria: dict) -> None:
        if not title.strip():
            raise ValueError("Name this search")
        if set(criteria) - {"country_codes", "programme_types", "funding", "qs_max", "min_research_fit", "max_pages"}:
            raise ValueError("Unsupported search criteria")
        if not 1 <= int(criteria.get("max_pages") or MAX_SEARCH_PAGES) <= MAX_SEARCH_PAGES:
            raise ValueError("Search page budget must be between 1 and 12")
        updated_title = title.strip()
        updated_criteria = json.dumps(criteria)
        with transaction(self.db_path) as db:
            current = db.execute("SELECT title,criteria_json FROM search_sessions WHERE id=?",
                                 (search_id,)).fetchone()
            if not current:
                raise ValueError("Search does not exist")
            db.execute("UPDATE search_sessions SET title=?,criteria_json=?,updated_at=? WHERE id=?",
                       (updated_title, updated_criteria, utc_now(), search_id))
            db.execute("""INSERT INTO record_change_events(entity_type,entity_id,action,changes_json,created_at)
                VALUES('SEARCH',?,'EDIT',?,?)""", (search_id, json.dumps({
                    "title": {"before": current["title"], "after": updated_title},
                    "criteria_json": {"before": current["criteria_json"], "after": updated_criteria},
                }), utc_now()))

    def archive_search(self, search_id: int, archived: bool = True) -> None:
        self.get_search(search_id)
        with transaction(self.db_path) as db:
            db.execute("UPDATE search_sessions SET status=?,updated_at=? WHERE id=?",
                       ("ARCHIVED" if archived else "ACTIVE", utc_now(), search_id))

    def queue(self, operation_type: str, *, search_id: int | None = None,
              application_id: int | None = None, context_id: int | None = None,
              faculty_id: int | None = None) -> int:
        """Validate and enqueue a search, scan, or faculty research operation."""
        if operation_type not in {"PROGRAMME_SEARCH", "FACULTY_DISCOVERY", *SCAN_TYPES}:
            raise ValueError("Unsupported background operation")
        if faculty_id and operation_type != "FACULTY_DISCOVERY":
            raise ValueError("Professor research requires a faculty discovery operation")
        if operation_type == "PROGRAMME_SEARCH":
            search = self.get_search(search_id)
            title = "Programme search · " + search["title"]
            context_id = search["context_id"]
            if search["status"] != "ACTIVE":
                raise ValueError("Restore the search before running it")
        elif operation_type in SCAN_TYPES:
            if not application_id:
                raise ValueError("Choose an application")
            with connect(self.db_path) as db:
                app = db.execute("""SELECT COALESCE(p.university,o.institution) AS institution
                    FROM applications a
                    LEFT JOIN programmes p ON p.id=a.programme_id
                    LEFT JOIN opportunities o ON o.id=a.opportunity_id
                    WHERE a.id=? AND a.archived_at IS NULL""", (application_id,)).fetchone()
            if not app:
                raise ValueError("Application is unavailable")
            title = ("Full refresh · " if operation_type == "APPLICATION_FULL_REFRESH" else "Scan missing details · ") + app["institution"]
            context_id = None
        else:
            if not application_id or not context_id:
                raise ValueError("Choose an application and applicant profile")
            with connect(self.db_path) as db:
                app = db.execute("""SELECT COALESCE(p.university,o.institution) AS institution
                    FROM applications a LEFT JOIN programmes p ON p.id=a.programme_id
                    LEFT JOIN opportunities o ON o.id=a.opportunity_id
                    WHERE a.id=? AND a.archived_at IS NULL""",
                                 (application_id,)).fetchone()
            if not app:
                raise ValueError("Application is unavailable")
            if faculty_id:
                with connect(self.db_path) as db:
                    faculty = db.execute("SELECT name,institution FROM faculty_profiles WHERE id=?",
                                         (faculty_id,)).fetchone()
                if not faculty or faculty["institution"].casefold() != app["institution"].casefold():
                    raise ValueError("Professor does not belong to this application institution")
                title = "Research deeper · " + faculty["name"]
            else:
                title = "Find professors · application " + str(application_id)
        now = utc_now()
        with transaction(self.db_path) as db:
            active = db.execute("SELECT COUNT(*) FROM operations WHERE status IN ('QUEUED','RUNNING','CANCEL_REQUESTED')").fetchone()[0]
            if active >= MAX_ACTIVE_OPERATIONS:
                raise ValueError("Two operations are already active. Stop or finish one before starting another")
            if operation_type in SCAN_TYPES and db.execute(
                    """SELECT 1 FROM operations WHERE application_id=?
                    AND operation_type IN ('APPLICATION_DETAIL_SCAN','APPLICATION_FULL_REFRESH')
                    AND status IN ('QUEUED','RUNNING','CANCEL_REQUESTED','PAUSED')""",
                    (application_id,)).fetchone():
                raise ValueError("Scan already running")
            if faculty_id and db.execute("""SELECT 1 FROM operations WHERE faculty_profile_id=?
                AND status IN ('QUEUED','RUNNING','CANCEL_REQUESTED')""", (faculty_id,)).fetchone():
                raise ValueError("Research is already running for this professor")
            operation_id = db.execute("""INSERT INTO operations
                (operation_type,title,status,search_id,application_id,context_id,faculty_profile_id,created_at,updated_at)
                VALUES(?,?,'QUEUED',?,?,?,?,?,?)""",
                (operation_type, title, search_id, application_id, context_id, faculty_id, now, now)).lastrowid
            self._event(db, operation_id, "Queued by applicant")
        return operation_id

    def launch(self, operation_id: int) -> None:
        operation = self.get(operation_id)
        if operation["status"] != "QUEUED":
            raise ValueError("Operation is not queued")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            subprocess.Popen(
                [sys.executable, "-m", "phd_agent.run_operation", str(self.db_path), str(operation_id)],
                cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=True, creationflags=flags, start_new_session=os.name != "nt",
            )
        except OSError as error:
            self._finish(operation_id, "FAILED", error="Background worker could not start")
            raise RuntimeError("Background worker could not start") from error

    def get(self, operation_id: int) -> dict:
        with connect(self.db_path) as db:
            row = db.execute("SELECT * FROM operations WHERE id=?", (operation_id,)).fetchone()
        if not row:
            raise ValueError("Operation does not exist")
        result = dict(row)
        result["checkpoint"] = json.loads(result.pop("checkpoint_json"))
        return result

    def list(self, limit: int = 30, *, search_id: int | None = None, application_id: int | None = None,
             operation_types: tuple[str, ...] | None = None) -> list[dict]:
        clauses, values = [], []
        if search_id is not None:
            clauses.append("search_id=?")
            values.append(search_id)
        if application_id is not None:
            clauses.append("application_id=?")
            values.append(application_id)
        if operation_types:
            clauses.append("operation_type IN (" + ",".join("?" for _ in operation_types) + ")")
            values.extend(operation_types)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with connect(self.db_path) as db:
            ids = [r[0] for r in db.execute(
                f"SELECT id FROM operations{where} ORDER BY id DESC LIMIT ?", (*values, limit))]
        return [self.get(item) for item in ids]

    def events(self, operation_id: int) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM operation_events WHERE operation_id=? ORDER BY id", (operation_id,))]

    def stop(self, operation_id: int) -> None:
        with transaction(self.db_path) as db:
            row = db.execute("SELECT status,operation_type,application_id FROM operations WHERE id=?", (operation_id,)).fetchone()
            if not row or row["status"] not in {"QUEUED", "RUNNING"}:
                raise ValueError("Only queued or running operations can be stopped")
            status = "CANCELLED" if row["status"] == "QUEUED" else "CANCEL_REQUESTED"
            db.execute("UPDATE operations SET status=?,cancellation_at=?,updated_at=?,finished_at=? WHERE id=?",
                       (status, utc_now(), utc_now(), utc_now() if status == "CANCELLED" else None, operation_id))
            self._event(db, operation_id, "Stop requested by applicant")
            if row["operation_type"] in SCAN_TYPES and row["application_id"]:
                action = "SCAN_STOPPED" if status == "CANCELLED" else "SCAN_STOP_REQUESTED"
                db.execute("""INSERT INTO record_change_events(entity_type,entity_id,action,changes_json,created_at)
                    VALUES('APPLICATION',?,?,?,?)""",
                    (row["application_id"], action, json.dumps({"operation_id": operation_id}), utc_now()))

    def resume(self, operation_id: int, *, retry: bool = False) -> None:
        with transaction(self.db_path) as db:
            row = db.execute("SELECT status,operation_type,application_id FROM operations WHERE id=?", (operation_id,)).fetchone()
            if not row or row["status"] not in {"CANCELLED", "INTERRUPTED", "FAILED", "PAUSED"}:
                raise ValueError("Only stopped, interrupted, or failed operations can resume")
            active = db.execute("SELECT COUNT(*) FROM operations WHERE status IN ('QUEUED','RUNNING','CANCEL_REQUESTED')").fetchone()[0]
            if active >= MAX_ACTIVE_OPERATIONS:
                raise ValueError("Two operations are already active")
            db.execute("""UPDATE operations SET status='QUEUED',stage='Queued',current_item=NULL,error_summary=NULL,
                checkpoint_json=CASE WHEN ? THEN '{}' ELSE checkpoint_json END,
                completed_units=CASE WHEN ? THEN 0 ELSE completed_units END,
                total_units=CASE WHEN ? THEN NULL ELSE total_units END,
                updated_at=?,finished_at=NULL,cancellation_at=NULL WHERE id=?""",
                (int(retry), int(retry), int(retry), utc_now(), operation_id))
            self._event(db, operation_id, "Restarted from the beginning" if retry else "Resumed from saved progress")
            if row["operation_type"] in SCAN_TYPES and row["application_id"]:
                db.execute("""INSERT INTO record_change_events(entity_type,entity_id,action,changes_json,created_at)
                    VALUES('APPLICATION',?,'SCAN_QUEUED',?,?)""",
                    (row["application_id"], json.dumps({"operation_id": operation_id, "retry": retry}), utc_now()))

    def recover_interrupted(self) -> int:
        with transaction(self.db_path) as db:
            rows = [r[0] for r in db.execute("SELECT id FROM operations WHERE status IN ('QUEUED','RUNNING','CANCEL_REQUESTED')")]
            for operation_id in rows:
                db.execute("UPDATE operations SET status='INTERRUPTED',updated_at=? WHERE id=?",
                           (utc_now(), operation_id))
                self._event(db, operation_id, "Interrupted by service restart; saved progress is available")
        return len(rows)

    def _claim(self, operation_id: int) -> bool:
        with transaction(self.db_path) as db:
            row = db.execute("SELECT status,operation_type,application_id FROM operations WHERE id=?",
                             (operation_id,)).fetchone()
            if not row or row["status"] != "QUEUED":
                return False
            db.execute("UPDATE operations SET status='RUNNING',started_at=?,updated_at=?,stage='Preparing' WHERE id=?",
                       (utc_now(), utc_now(), operation_id))
            self._event(db, operation_id, "Started")
            if row["operation_type"] in SCAN_TYPES and row["application_id"]:
                resumed = db.execute("""SELECT 1 FROM operation_events WHERE operation_id=? AND event_text IN
                    ('Resumed from saved progress','Restarted from the beginning')""", (operation_id,)).fetchone()
                if resumed:
                    db.execute("""INSERT INTO record_change_events(entity_type,entity_id,action,changes_json,created_at)
                        VALUES('APPLICATION',?,'SCAN_RESUMED',?,?)""",
                        (row["application_id"], json.dumps({"operation_id": operation_id}), utc_now()))
        return True

    def _progress(self, operation_id: int, *, stage: str, item: str = "", completed: int | None = None,
                  total: int | None = None, checkpoint: dict | None = None, model: str | None = None,
                  event: str | None = None) -> bool:
        """Persist progress for a running operation and report whether it may continue."""
        with transaction(self.db_path) as db:
            row = db.execute("SELECT status,search_id,application_id,faculty_profile_id FROM operations WHERE id=?", (operation_id,)).fetchone()
            if not row or row["status"] != "RUNNING":
                return False
            results = None
            if row["search_id"]:
                results = db.execute("""SELECT COUNT(*) FROM programme_candidates c
                    JOIN search_sessions s ON s.intent_id=c.intent_id
                    WHERE s.id=? AND c.archived_at IS NULL""", (row["search_id"],)).fetchone()[0]
            elif row["faculty_profile_id"]:
                results = db.execute("SELECT COUNT(*) FROM faculty_research_snapshots WHERE faculty_profile_id=?",
                                     (row["faculty_profile_id"],)).fetchone()[0]
            db.execute("""UPDATE operations SET stage=?,current_item=?,
                completed_units=COALESCE(?,completed_units),total_units=COALESCE(?,total_units),
                checkpoint_json=COALESCE(?,checkpoint_json),model_route=COALESCE(?,model_route),
                results_found=COALESCE(?,results_found),updated_at=? WHERE id=?""",
                (stage, item[:300], completed, total, json.dumps(checkpoint) if checkpoint is not None else None,
                 model, results, utc_now(), operation_id))
            if event:
                self._event(db, operation_id, event)
        return True

    def _finish(self, operation_id: int, status: str, *, error: str | None = None, results: int | None = None) -> None:
        """Finalize an operation with its status, result count, and optional error."""
        with transaction(self.db_path) as db:
            row = db.execute("SELECT status,search_id,application_id,operation_type,faculty_profile_id FROM operations WHERE id=?", (operation_id,)).fetchone()
            if not row:
                return
            if row["status"] == "INTERRUPTED":
                return
            if row["status"] == "CANCEL_REQUESTED":
                status = "CANCELLED"
                if row["operation_type"] in SCAN_TYPES and row["application_id"]:
                    db.execute("""INSERT INTO record_change_events(entity_type,entity_id,action,changes_json,created_at)
                        VALUES('APPLICATION',?,'SCAN_STOPPED',?,?)""",
                        (row["application_id"], json.dumps({"operation_id": operation_id}), utc_now()))
            if results is None and row["operation_type"] in SCAN_TYPES:
                results = db.execute("""SELECT COUNT(*) FROM application_scan_findings
                    WHERE operation_id=? AND state='EXTRACTED'""", (operation_id,)).fetchone()[0]
            elif results is None and row["search_id"]:
                results = db.execute("""SELECT COUNT(*) FROM programme_candidates c
                    JOIN search_sessions s ON s.intent_id=c.intent_id
                    WHERE s.id=? AND c.archived_at IS NULL""", (row["search_id"],)).fetchone()[0]
            elif results is None and row["faculty_profile_id"]:
                results = db.execute("SELECT COUNT(*) FROM faculty_research_snapshots WHERE faculty_profile_id=?",
                                     (row["faculty_profile_id"],)).fetchone()[0]
            elif results is None and row["application_id"]:
                results = db.execute("""SELECT COUNT(*) FROM faculty_candidates fc
                    JOIN source_catalogue sc ON sc.id=fc.source_catalogue_id
                    WHERE fc.review_state='NEW' AND sc.university=(
                        SELECT COALESCE(p.university,o.institution) FROM applications a
                        LEFT JOIN programmes p ON p.id=a.programme_id
                        LEFT JOIN opportunities o ON o.id=a.opportunity_id WHERE a.id=?)""",
                    (row["application_id"],)).fetchone()[0]
            results = results or 0
            db.execute("""UPDATE operations SET status=?,stage=?,current_item=NULL,results_found=?,error_summary=?,
                updated_at=?,finished_at=? WHERE id=?""",
                (status, status.title(), results, error, utc_now(), utc_now(), operation_id))
            label = "fields resolved" if row["operation_type"] in SCAN_TYPES else "results saved"
            self._event(db, operation_id, f"{status.title()}: {results} {label}")

    def run(self, operation_id: int, *, provider=None, api_key: str | None = None, fetcher=None) -> None:
        """Run a queued operation and persist its terminal outcome."""
        if not self._claim(operation_id):
            return
        operation = self.get(operation_id)
        key = os.getenv("OPENAI_API_KEY", "").strip() if api_key is None else api_key
        checkpoint = operation["checkpoint"]
        try:
            if operation["operation_type"] in SCAN_TYPES:
                from phd_agent.application_rescan import ApplicationRescan
                ApplicationRescan(self.db_path).execute(operation_id, self, fetcher=fetcher)
                return
            orchestrator = ProgrammeOrchestrator(self.db_path)
            if operation["faculty_profile_id"]:
                self._research(operation_id, operation, orchestrator, fetcher=fetcher)
            elif operation["operation_type"] == "PROGRAMME_SEARCH":
                search = self.get_search(operation["search_id"])
                if not checkpoint.get("ledger_done"):
                    if not self._progress(operation_id, stage="Checking saved programmes", event="Checking existing programme records"):
                        self._finish(operation_id, "CANCELLED")
                        return
                    orchestrator.discover_from_ledger(search["intent_id"])
                    checkpoint["ledger_done"] = True
                    self._progress(operation_id, stage="Checking official sources", checkpoint=checkpoint,
                                   event="Saved programme records checked")
                if key or provider or checkpoint.get("hits") is not None:
                    budget = int(search["criteria"].get("max_pages") or MAX_SEARCH_PAGES)
                    self._discovery(operation_id, checkpoint, budget, lambda **hooks:
                        orchestrator.discover_official_web(search["intent_id"], api_key=key,
                                                           provider=provider,
                                                           query_hint=_query_hint(search["criteria"]), **hooks))
            else:
                if not (key or provider or checkpoint.get("hits") is not None):
                    self._finish(operation_id, "FAILED", error="Configure OpenAI to search official faculty pages")
                    return
                self._discovery(operation_id, checkpoint, MAX_SEARCH_PAGES, lambda **hooks:
                    orchestrator.discover_faculty_official_web(
                        operation["application_id"], operation["context_id"], api_key=key,
                        provider=provider, **hooks))
            self._finish(operation_id, "COMPLETED")
        except Exception as error:
            # Only the error class reaches the UI; external responses and credentials are not logged.
            self._finish(operation_id, "FAILED", error=f"{type(error).__name__} while {self.get(operation_id)['stage']}")

    def _discovery(self, operation_id: int, checkpoint: dict, budget: int, discover) -> None:
        saved_hits = [OfficialSearchResult(**hit) for hit in checkpoint["hits"]] if "hits" in checkpoint else None
        if saved_hits is None:
            self._progress(operation_id, stage="Finding official pages", event="Finding official source URLs")

        def on_hits(hits, route):
            if saved_hits is not None:
                return
            checkpoint["hits"] = [vars(hit) for hit in hits[:budget]]
            checkpoint["index"] = 0
            self._progress(operation_id, stage="Checking official pages", completed=0,
                           total=len(checkpoint["hits"]), checkpoint=checkpoint,
                           model=f"{route.provider}/{route.model}",
                           event=f"Found {len(checkpoint['hits'])} official pages to check")

        def before_hit(index, total, hit):
            if index < checkpoint.get("index", 0):
                return True
            return self._progress(operation_id, stage="Checking official pages",
                                  item=hit.institution + " · " + hit.title,
                                  completed=index, total=min(total, budget))

        def after_hit(index, total, hit):
            checkpoint["index"] = index
            self._progress(operation_id, stage="Checking official pages", completed=index,
                           total=min(total, budget), checkpoint=checkpoint,
                           event=f"Checked {hit.institution}: {hit.title}")

        discover(saved_hits=saved_hits, on_hits=on_hits, before_hit=before_hit,
                 after_hit=after_hit, start_index=checkpoint.get("index", 0), max_hits=budget)

    def _research(self, operation_id: int, operation: dict, orchestrator: ProgrammeOrchestrator,
                  *, fetcher=None) -> None:
        """Collect reviewable research from identity-matched official faculty pages."""
        from urllib.parse import urljoin, urlparse
        from bs4 import BeautifulSoup

        with connect(self.db_path) as db:
            professor = db.execute("SELECT * FROM faculty_profiles WHERE id=?",
                                   (operation["faculty_profile_id"],)).fetchone()
        if not professor or not professor["official_profile_url"]:
            raise ValueError("Professor needs an official profile URL")
        checkpoint = operation["checkpoint"]
        pages = checkpoint.get("pages") or list(dict.fromkeys(
            url for url in (professor["official_profile_url"], professor["lab_url"]) if url))
        checkpoint["pages"] = pages
        research = FacultyResearch(self.db_path)
        index = checkpoint.get("index", 0)
        while index < min(len(pages), 5):
            url = pages[index]
            if not self._progress(operation_id, stage="Checking official research", item=url,
                                  completed=index, total=len(pages), checkpoint=checkpoint):
                return
            acquisition = fetcher(url) if fetcher else orchestrator.acquire_page(url)
            if (acquisition.status == "ACQUIRED" and len(acquisition.text or "") >= 300
                    and professor["name"].casefold() in acquisition.text.casefold()
                    and urlparse(acquisition.url).hostname == urlparse(url).hostname):
                evidence_id = orchestrator.ledger.create_evidence(
                    url, "FACULTY", " ".join(acquisition.text.split())[:12000], "UNVERIFIED")
                research.save_snapshot(url, evidence_id,
                                       extract_official_research(acquisition.text, acquisition.html),
                                       faculty_id=professor["id"])
                if index == 0 and acquisition.html:
                    base_host = urlparse(url).hostname
                    soup = BeautifulSoup(acquisition.html, "html.parser")
                    for link in soup.find_all("a", href=True):
                        if not re.search(r"\b(projects?|research|lab)\b", link.get_text(" ", strip=True), re.I):
                            continue
                        related = urljoin(url, link["href"])
                        if (urlparse(related).scheme != "https" or
                                urlparse(related).hostname != base_host):
                            continue
                        related = canonical_url(related)
                        if related not in pages and len(pages) < 5:
                            pages.append(related)
            else:
                self._progress(operation_id, stage="Checking official research", item=url,
                               event="Page needs professor identity review; no research attributed")
            checkpoint["index"] = index + 1
            if not self._progress(operation_id, stage="Checking official research", item=url,
                                  completed=index + 1, total=len(pages), checkpoint=checkpoint,
                                  event=f"Checked official research page {index + 1}/{len(pages)}"):
                return
            index += 1
        if not self._progress(operation_id, stage="Checking reviewed publications", checkpoint=checkpoint):
            return
        with connect(self.db_path) as db:
            verified = db.execute("""SELECT COUNT(*) FROM publications p JOIN source_evidence e
                ON e.id=p.source_evidence_id WHERE p.faculty_profile_id=?
                AND e.verification_state='VERIFIED'""", (professor["id"],)).fetchone()[0]
        if not self._progress(operation_id, stage="Checking applicant alignment", checkpoint=checkpoint,
                              event=f"{verified} previously verified publications available; external author identity remains manual"):
            return
        orchestrator.professor_cards(operation["application_id"], operation["context_id"])
