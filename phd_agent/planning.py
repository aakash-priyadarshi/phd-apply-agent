"""Deterministic daily priorities and a calendar over existing application records."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.ledger import Ledger


EVENT_TYPES = ("PERSONAL", "INTERVIEW", "FUNDING", "DOCUMENT", "FOLLOW_UP", "OTHER")
PRIORITY_ORDER = {"URGENT": 0, "NEXT": 1, "LATER": 2}


def _calendar_date(value: str) -> str:
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError("Use an ISO date or date and time") from error
    if len(text) < 10 or text[4] != "-" or text[7] != "-":
        raise ValueError("Use an ISO date or date and time")
    return parsed.isoformat() if len(text) > 10 else parsed.date().isoformat()


class DailyPlanner:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.ledger = Ledger(self.db_path)

    def _audit(self, db, entity: str, entity_id: int, action: str, changes: dict) -> None:
        db.execute("""INSERT INTO record_change_events(entity_type,entity_id,action,changes_json,created_at)
            VALUES(?,?,?,?,?)""", (entity, entity_id, action, json.dumps(changes, sort_keys=True), utc_now()))

    def create_event(self, title: str, event_type: str, starts_at: str, *,
                     application_id: int | None = None, notes: str = "") -> int:
        if not title.strip() or event_type not in EVENT_TYPES:
            raise ValueError("Add a title and choose an event type")
        when = _calendar_date(starts_at)
        now = utc_now()
        with transaction(self.db_path) as db:
            if application_id is not None and not db.execute(
                    "SELECT 1 FROM applications WHERE id=? AND archived_at IS NULL", (application_id,)).fetchone():
                raise ValueError("Choose an active application")
            event_id = db.execute("""INSERT INTO calendar_events
                (application_id,title,event_type,starts_at,notes,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?)""", (application_id, title.strip(), event_type, when,
                                           notes.strip(), now, now)).lastrowid
            self._audit(db, "CALENDAR_EVENT", event_id, "CREATE",
                        {"title": title.strip(), "starts_at": when, "application_id": application_id})
        return event_id

    def edit_event(self, event_id: int, *, title: str, event_type: str, starts_at: str,
                   notes: str) -> None:
        if not title.strip() or event_type not in EVENT_TYPES:
            raise ValueError("Add a title and choose an event type")
        when = _calendar_date(starts_at)
        changes = {"title": title.strip(), "event_type": event_type,
                   "starts_at": when, "notes": notes.strip()}
        with transaction(self.db_path) as db:
            before = db.execute("SELECT * FROM calendar_events WHERE id=? AND status='ACTIVE'",
                                (event_id,)).fetchone()
            if not before:
                raise ValueError("Calendar event is unavailable")
            db.execute("""UPDATE calendar_events SET title=?,event_type=?,starts_at=?,notes=?,updated_at=?
                WHERE id=?""", (*changes.values(), utc_now(), event_id))
            self._audit(db, "CALENDAR_EVENT", event_id, "EDIT",
                        {key: {"before": before[key], "after": value} for key, value in changes.items()
                         if before[key] != value})

    def set_event_active(self, event_id: int, active: bool) -> None:
        with transaction(self.db_path) as db:
            row = db.execute("SELECT status FROM calendar_events WHERE id=?", (event_id,)).fetchone()
            if not row:
                raise ValueError("Calendar event does not exist")
            new_status = "ACTIVE" if active else "CANCELLED"
            db.execute("UPDATE calendar_events SET status=?,updated_at=? WHERE id=?",
                       (new_status, utc_now(), event_id))
            self._audit(db, "CALENDAR_EVENT", event_id, "RESTORE" if active else "CANCEL",
                        {"status": {"before": row["status"], "after": new_status}})

    def complete_task(self, task_id: int) -> None:
        with transaction(self.db_path) as db:
            row = db.execute("""SELECT t.status FROM application_tasks t
                JOIN applications a ON a.id=t.application_id
                WHERE t.id=? AND a.archived_at IS NULL""", (task_id,)).fetchone()
            if not row or row["status"] in {"DONE", "CANCELLED"}:
                raise ValueError("Task is unavailable")
            now = utc_now()
            db.execute("UPDATE application_tasks SET status='DONE',completed_at=? WHERE id=?", (now, task_id))
            self._audit(db, "APPLICATION_TASK", task_id, "COMPLETE",
                        {"status": {"before": row["status"], "after": "DONE"}})

    def calendar(self, start: date, end: date, *, include_cancelled: bool = False) -> list[dict]:
        if start > end or (end - start).days > 730:
            raise ValueError("Choose a date range of at most two years")
        first, last = start.isoformat(), end.isoformat()
        items = []
        with connect(self.db_path) as db:
            deadlines = db.execute("""SELECT d.id,d.application_id,d.deadline_type,d.due_at,
                d.verification_state,e.canonical_url AS source_url,
                COALESCE(p.university,o.institution) AS institution
                FROM deadlines d JOIN applications a ON a.id=d.application_id
                LEFT JOIN programmes p ON p.id=a.programme_id
                LEFT JOIN opportunities o ON o.id=a.opportunity_id
                JOIN source_evidence e ON e.id=d.source_evidence_id
                WHERE a.archived_at IS NULL AND substr(d.due_at,1,10) BETWEEN ? AND ?""",
                (first, last)).fetchall()
            tasks = db.execute("""SELECT t.id,t.application_id,t.description,t.due_at,t.priority,
                COALESCE(p.university,o.institution) AS institution
                FROM application_tasks t JOIN applications a ON a.id=t.application_id
                LEFT JOIN programmes p ON p.id=a.programme_id
                LEFT JOIN opportunities o ON o.id=a.opportunity_id
                WHERE a.archived_at IS NULL AND t.status NOT IN ('DONE','CANCELLED')
                  AND substr(t.due_at,1,10) BETWEEN ? AND ?""", (first, last)).fetchall()
            referees = db.execute("""SELECT r.id,r.application_id,r.referee_name,r.deadline_at,
                COALESCE(p.university,o.institution) AS institution
                FROM application_referees r JOIN applications a ON a.id=r.application_id
                LEFT JOIN programmes p ON p.id=a.programme_id
                LEFT JOIN opportunities o ON o.id=a.opportunity_id
                WHERE a.archived_at IS NULL AND r.submission_state!='SUBMITTED'
                  AND substr(r.deadline_at,1,10) BETWEEN ? AND ?""", (first, last)).fetchall()
            manual = db.execute("""SELECT c.*,COALESCE(p.university,o.institution) AS institution
                FROM calendar_events c LEFT JOIN applications a ON a.id=c.application_id
                LEFT JOIN programmes p ON p.id=a.programme_id
                LEFT JOIN opportunities o ON o.id=a.opportunity_id
                WHERE (c.application_id IS NULL OR a.archived_at IS NULL)
                  AND (? OR c.status='ACTIVE') AND substr(c.starts_at,1,10) BETWEEN ? AND ?""",
                (int(include_cancelled), first, last)).fetchall()
        for row in deadlines:
            items.append({"kind": "DEADLINE", "id": row["id"], "application_id": row["application_id"],
                          "title": f"{row['deadline_type'].replace('_', ' ').title()} deadline",
                          "institution": row["institution"], "when": row["due_at"],
                          "verification_state": row["verification_state"], "source_url": row["source_url"]})
        for row in tasks:
            items.append({"kind": "TASK", "id": row["id"], "application_id": row["application_id"],
                          "title": row["description"], "institution": row["institution"],
                          "when": row["due_at"], "priority": row["priority"]})
        for row in referees:
            items.append({"kind": "REFEREE", "id": row["id"], "application_id": row["application_id"],
                          "title": f"Referee: {row['referee_name']}", "institution": row["institution"],
                          "when": row["deadline_at"]})
        for row in manual:
            items.append({"kind": "MANUAL", "id": row["id"], "application_id": row["application_id"],
                          "title": row["title"], "institution": row["institution"],
                          "when": row["starts_at"], "event_type": row["event_type"],
                          "notes": row["notes"], "status": row["status"]})
        return sorted(items, key=lambda item: (item["when"], item["kind"], item["id"]))

    def dashboard(self, *, today: date | None = None) -> dict:
        now = today or date.today()
        daily = self.ledger.today(as_of=now)
        with connect(self.db_path) as db:
            candidates = db.execute("""SELECT COUNT(*) FROM programme_candidates
                WHERE archived_at IS NULL AND review_state IN ('NEW','SHORTLISTED')""").fetchone()[0]
            operations = db.execute("""SELECT COUNT(*) FROM operations
                WHERE status IN ('QUEUED','RUNNING','CANCEL_REQUESTED')""").fetchone()[0]
            replies = [dict(row) for row in db.execute("""SELECT r.id,r.application_id,r.subject_raw,
                r.message_at,COALESCE(p.university,o.institution) AS institution
                FROM reply_events r LEFT JOIN applications a ON a.id=r.application_id
                LEFT JOIN programmes p ON p.id=a.programme_id
                LEFT JOIN opportunities o ON o.id=a.opportunity_id
                WHERE r.direction='INBOUND' AND r.detected_state='NEW'
                  AND (r.application_id IS NULL OR a.archived_at IS NULL)
                ORDER BY r.message_at DESC LIMIT 20""")]
            tasks = [dict(row) for row in db.execute("""SELECT t.id,t.application_id,t.description,
                t.due_at,t.priority,COALESCE(p.university,o.institution) AS institution
                FROM application_tasks t JOIN applications a ON a.id=t.application_id
                LEFT JOIN programmes p ON p.id=a.programme_id
                LEFT JOIN opportunities o ON o.id=a.opportunity_id
                WHERE a.archived_at IS NULL AND t.status NOT IN ('DONE','CANCELLED')
                ORDER BY t.due_at IS NULL,t.due_at,t.id LIMIT 100""")]
            application_status = {row["id"]: row["status"] for row in db.execute(
                "SELECT id,status FROM applications WHERE archived_at IS NULL")}
            recent_overdue = [dict(row) for row in db.execute("""SELECT d.id,d.application_id,
                d.deadline_type,d.due_at,d.verification_state,
                COALESCE(p.university,o.institution) AS institution
                FROM deadlines d JOIN applications a ON a.id=d.application_id
                LEFT JOIN programmes p ON p.id=a.programme_id
                LEFT JOIN opportunities o ON o.id=a.opportunity_id
                WHERE a.archived_at IS NULL AND substr(d.due_at,1,10) BETWEEN ? AND ?
                ORDER BY d.due_at DESC LIMIT 50""",
                ((now - timedelta(days=30)).isoformat(), (now - timedelta(days=1)).isoformat()))]
        actions = []
        upcoming_14 = []
        for deadline in daily["upcoming_deadlines"]:
            if (deadline["deadline_type"] == "APPLICATION" and application_status.get(deadline["application_id"])
                    in {"SUBMITTED", "INTERVIEW", "OFFER", "REJECTED", "WITHDRAWN"}):
                continue
            due = date.fromisoformat(deadline["due_at"][:10])
            days = (due - now).days
            if days > 14:
                continue
            upcoming_14.append(deadline)
            priority = "URGENT" if days <= 7 else "NEXT"
            when_text = "today" if days == 0 else f"in {days} days"
            label = f"{deadline['institution']} · {deadline['deadline_type'].replace('_', ' ').title()} deadline {when_text}"
            if deadline["verification_state"] != "VERIFIED":
                label = "Check: " + label
            actions.append({"priority": priority, "title": label,
                            "detail": deadline["due_at"], "application_id": deadline["application_id"],
                            "page": "Applications", "due": deadline["due_at"]})
        for deadline in recent_overdue:
            if (deadline["deadline_type"] == "APPLICATION" and application_status.get(deadline["application_id"])
                    in {"SUBMITTED", "INTERVIEW", "OFFER", "REJECTED", "WITHDRAWN"}):
                continue
            title = (f"{deadline['institution']} · {deadline['deadline_type'].replace('_', ' ').title()} "
                     "deadline passed")
            if deadline["verification_state"] != "VERIFIED":
                title = "Check: " + title
            actions.append({"priority": "URGENT", "title": title, "detail": deadline["due_at"],
                            "application_id": deadline["application_id"], "page": "Applications",
                            "due": deadline["due_at"]})
        for task in tasks:
            if not task["due_at"] and task["priority"] != "HIGH":
                continue
            days = (date.fromisoformat(task["due_at"][:10]) - now).days if task["due_at"] else None
            if days is not None and days > 30:
                continue
            priority = "URGENT" if days is not None and days < 0 else "NEXT" if (
                task["priority"] == "HIGH" or days is not None and days <= 7) else "LATER"
            actions.append({"priority": priority, "title": task["description"],
                            "detail": f"{task['institution']} · due {task['due_at'] or 'date open'}",
                            "application_id": task["application_id"], "task_id": task["id"],
                            "page": "Applications", "due": task["due_at"]})
        for reply in replies:
            actions.append({"priority": "NEXT", "title": "Professor replied: " + reply["subject_raw"],
                            "detail": reply["institution"] or "Professor correspondence",
                            "application_id": reply["application_id"], "page": "People", "due": reply["message_at"]})
        missing_by_app = {}
        for requirement in daily["missing_required"]:
            missing_by_app.setdefault(requirement["application_id"], []).append(requirement)
        near_deadline_apps = {item["application_id"] for item in upcoming_14}
        for app_id, requirements in missing_by_app.items():
            actions.append({"priority": "NEXT" if app_id in near_deadline_apps else "LATER",
                            "title": f"{requirements[0]['institution']} · {len(requirements)} required item(s) missing",
                            "detail": ", ".join(item["original_label"] for item in requirements[:3]),
                            "application_id": app_id, "page": "Applications", "due": None})
        for event in self.calendar(now, now + timedelta(days=14)):
            if event["kind"] != "MANUAL":
                continue
            actions.append({"priority": "NEXT", "title": event["title"],
                            "detail": f"{event['institution'] or 'Personal'} · {event['when']}",
                            "application_id": event["application_id"], "page": "Calendar",
                            "due": event["when"]})
        if candidates:
            actions.append({"priority": "LATER", "title": f"Review {candidates} programme results",
                            "detail": "New and saved candidates", "page": "Find programmes", "due": None})
        from phd_agent.application_rescan import ApplicationRescan
        scanner = ApplicationRescan(self.db_path)
        with connect(self.db_path) as db:
            active_apps = db.execute("""SELECT a.id, COALESCE(p.university,o.institution) AS institution
                FROM applications a
                LEFT JOIN programmes p ON p.id=a.programme_id
                LEFT JOIN opportunities o ON o.id=a.opportunity_id
                WHERE a.archived_at IS NULL ORDER BY a.id""").fetchall()
        for app in active_apps:
            summary = scanner.completeness(app["id"])
            if not summary["missing"] or scanner.active_operation(app["id"]):
                continue
            actions.append({"priority": "NEXT",
                            "title": f"{app['institution']} · {len(summary['missing'])} application details are still missing",
                            "detail": ", ".join(item["label"] for item in summary["missing"][:5]),
                            "application_id": app["id"], "page": "Applications",
                            "scan_application_id": app["id"], "due": None})
        actions.sort(key=lambda item: (PRIORITY_ORDER[item["priority"]], item["due"] or "9999", item["title"]))
        return {"counts": {"deadlines_14_days": len(upcoming_14), "new_replies": len(replies),
                           "programme_results": candidates, "missing_required": len(daily["missing_required"]),
                           "unknown_requirements": len(daily["unknown_requirements"]),
                           "document_alerts": len(daily["document_alerts"]),
                           "active_operations": operations},
                "actions": actions[:20]}
