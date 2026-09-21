"""Reply classification, reply-triggered tasks, and reviewed follow-up drafts."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.materials import MaterialStudio
from phd_agent.outreach import OutreachService


CATEGORIES = (
    "POSITIVE_INTEREST", "APPLICATION_REQUESTED", "PROPOSAL_REQUESTED", "CV_REQUESTED",
    "COVER_LETTER_REQUESTED", "ADDITIONAL_INFO_REQUESTED", "MEETING_REQUESTED",
    "REFERRAL", "NOT_ACCEPTING", "NO_FUNDING", "REJECTION", "OUT_OF_OFFICE", "BOUNCE", "OTHER",
)
DEFAULT_FOLLOW_UP_DAYS = 14
TASK_FOR_CATEGORY = {
    "PROPOSAL_REQUESTED": ("DOCUMENT", "Prepare a reviewed proposal variant for the reply"),
    "CV_REQUESTED": ("DOCUMENT", "Prepare an updated CV for the reply"),
    "COVER_LETTER_REQUESTED": ("DOCUMENT", "Prepare a reviewed cover letter for the reply"),
    "ADDITIONAL_INFO_REQUESTED": ("DOCUMENT", "Prepare additional information requested in the reply"),
    "MEETING_REQUESTED": ("MEETING", "Schedule a meeting requested by the professor"),
    "APPLICATION_REQUESTED": ("PORTAL", "Complete the formal application portal checklist"),
    "POSITIVE_INTEREST": ("FOLLOW_UP", "Draft a reviewed follow-up after positive interest"),
    "REFERRAL": ("FOLLOW_UP", "Record the referral and decide the next contact"),
    "NOT_ACCEPTING": ("CONTACT_REVIEW", "Review contact restrictions after a not-accepting reply"),
    "NO_FUNDING": ("CONTACT_REVIEW", "Review funding and contact state after a no-funding reply"),
    "REJECTION": ("CONTACT_REVIEW", "Review contact restrictions after a rejection"),
    "BOUNCE": ("CONTACT_REVIEW", "Reconcile the bounced outreach message"),
}
_HEURISTICS = (
    ("BOUNCE", ("delivery status", "undeliverable", "mailer-daemon", "returned mail", "bounce")),
    ("OUT_OF_OFFICE", ("out of office", "automatic reply", "away from the office", "on leave")),
    ("NOT_ACCEPTING", ("not accepting students", "not taking students", "no new students")),
    ("NO_FUNDING", ("no funding", "unfunded", "without funding")),
    ("REJECTION", ("position has been filled", "not a fit", "unable to supervise")),
    ("PROPOSAL_REQUESTED", ("send a proposal", "research proposal", "research statement")),
    ("CV_REQUESTED", ("send your cv", "updated cv", "curriculum vitae")),
    ("COVER_LETTER_REQUESTED", ("cover letter",)),
    ("APPLICATION_REQUESTED", ("apply through", "application portal", "formal application")),
    ("MEETING_REQUESTED", ("happy to meet", "schedule a call", "zoom", "teams meeting")),
    ("REFERRAL", ("refer you", "colleague of mine", "another faculty")),
    ("POSITIVE_INTEREST", ("happy to discuss", "interested in your", "please send materials")),
)


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ReplyIntelligence:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.outreach = OutreachService(self.db_path)
        self.materials = MaterialStudio(self.db_path)

    @staticmethod
    def suggest_category(subject: str, excerpt: str = "") -> str:
        text = f"{subject}\n{excerpt}".casefold()
        for category, phrases in _HEURISTICS:
            if any(phrase in text for phrase in phrases):
                return category
        return "OTHER"

    def classify(self, reply_event_id: int, category: str, reviewer: str, *,
                 excerpt: str = "", requested_document_type: str | None = None,
                 requested_length: str | None = None, requested_deadline: str | None = None,
                 confidence: str = "OPERATOR") -> dict:
        if category not in CATEGORIES:
            raise ValueError("Unknown reply category")
        if confidence not in {"HEURISTIC", "OPERATOR"}:
            raise ValueError("Classification confidence must be HEURISTIC or OPERATOR")
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        with transaction(self.db_path) as db:
            event = db.execute("SELECT * FROM reply_events WHERE id=?", (reply_event_id,)).fetchone()
            if not event:
                raise ValueError("Reply event not found")
            existing = db.execute("SELECT id FROM reply_classifications WHERE reply_event_id=?",
                                  (reply_event_id,)).fetchone()
            if existing:
                raise ValueError("Reply already classified")
            task_id = None
            spec = TASK_FOR_CATEGORY.get(category)
            if spec and event["application_id"]:
                task_id = db.execute("""INSERT INTO application_tasks
                    (application_id,task_type,description,status,due_at,priority,source_context,created_at,notes)
                    VALUES(?,?,?,'TODO',?,?,?,?,?)""",
                    (event["application_id"], spec[0], spec[1], requested_deadline,
                     "HIGH" if category.endswith("REQUESTED") else "MEDIUM",
                     _dump({"reply_event_id": reply_event_id, "category": category}), utc_now(), "")).lastrowid
            db.execute("""INSERT INTO reply_classifications
                (reply_event_id,category,confidence,excerpt,requested_document_type,requested_length,
                 requested_deadline,classified_by,classified_at,task_id)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (reply_event_id, category, confidence, excerpt or "", requested_document_type,
                 requested_length, requested_deadline, reviewer.strip(), utc_now(), task_id))
            db.execute("UPDATE reply_events SET detected_state='CLASSIFIED' WHERE id=?", (reply_event_id,))
            if category == "BOUNCE" and event["outreach_package_id"]:
                db.execute("""UPDATE outreach_packages SET status='BOUNCED'
                    WHERE id=? AND status IN ('SENT','FOLLOW_UP_DUE','REPLIED')""", (event["outreach_package_id"],))
                db.execute("""UPDATE outreach_messages SET status='BOUNCED',updated_at=?
                    WHERE package_id=? AND status IN ('SENT','REPLIED')""", (utc_now(), event["outreach_package_id"]))
            classification_id = db.execute("SELECT id FROM reply_classifications WHERE reply_event_id=?",
                                           (reply_event_id,)).fetchone()[0]
        return {"id": classification_id, "task_id": task_id, "category": category}

    def list_inbox(self) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute("""SELECT e.*, c.category, c.confidence, c.task_id, f.name AS faculty_name
                FROM reply_events e
                LEFT JOIN reply_classifications c ON c.reply_event_id=e.id
                LEFT JOIN faculty_profiles f ON f.id=e.faculty_profile_id
                ORDER BY e.id DESC""")]

    def counts(self) -> dict:
        with connect(self.db_path) as db:
            return {
                "new_replies": db.execute("SELECT COUNT(*) FROM reply_events WHERE detected_state='NEW'").fetchone()[0],
                "follow_ups_due": db.execute("SELECT COUNT(*) FROM outreach_packages WHERE status='FOLLOW_UP_DUE'").fetchone()[0],
            }

    def mark_follow_ups_due(self, *, now: datetime | None = None) -> list[int]:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("Follow-up time must include a timezone")
        marked = []
        with connect(self.db_path) as db:
            rows = [dict(r) for r in db.execute("""SELECT p.id, p.campaign_id, m.sent_at, c.policy_json
                FROM outreach_packages p
                JOIN outreach_messages m ON m.package_id=p.id
                LEFT JOIN campaigns c ON c.id=p.campaign_id
                WHERE p.status='SENT' AND m.status='SENT' AND m.sent_at IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM reply_events r
                    WHERE r.outreach_package_id=p.id AND r.direction='INBOUND')""")]
        for row in rows:
            delay = DEFAULT_FOLLOW_UP_DAYS
            if row["policy_json"]:
                delay = json.loads(row["policy_json"]).get("follow_up_delay_days", delay)
            sent = datetime.fromisoformat(row["sent_at"].replace("Z", "+00:00"))
            if sent.tzinfo is None:
                sent = sent.replace(tzinfo=timezone.utc)
            if now < sent + timedelta(days=delay):
                continue
            with transaction(self.db_path) as db:
                db.execute("UPDATE outreach_packages SET status='FOLLOW_UP_DUE' WHERE id=? AND status='SENT'",
                           (row["id"],))
                self.outreach._event(db, row["id"], None, "FOLLOW_UP_DUE", "SENT", "FOLLOW_UP_DUE", "SYSTEM")
            marked.append(row["id"])
        return marked

    def draft_follow_up(self, package_id: int, *, profile_version_id: int | None = None,
                        track_version_id: int | None = None) -> int:
        package = self.outreach.get_package(package_id)
        if package["status"] not in {"SENT", "FOLLOW_UP_DUE", "REPLIED"}:
            raise ValueError("Follow-up drafts start from a sent or replied package")
        context, _ = self.outreach._unpack(package)
        return self.outreach.prepare(
            context.professor_id, context.application_id, context.document_package_id,
            profile_version_id or context.profile_version_id,
            track_version_id or context.research_track_version_id,
            campaign_id=package["campaign_id"], stage="FOLLOW_UP")

    def generate_requested_document(self, classification_id: int, *,
                                    profile_version_id: int, track_version_id: int) -> int:
        with connect(self.db_path) as db:
            row = db.execute("""SELECT c.*, e.faculty_profile_id, e.application_id
                FROM reply_classifications c JOIN reply_events e ON e.id=c.reply_event_id
                WHERE c.id=?""", (classification_id,)).fetchone()
        if not row:
            raise ValueError("Classification not found")
        if row["category"] != "PROPOSAL_REQUESTED":
            raise ValueError("Only proposal requests generate a track-bound variant here")
        if not row["application_id"] or not row["faculty_profile_id"]:
            raise ValueError("Proposal requests need a linked application and professor")
        length = (row["requested_length"] or "").casefold()
        format_name = "CONCEPT_NOTE"
        if "one" in length or length.startswith("1"):
            format_name = "ONE_PAGE"
        elif "two" in length or length.startswith("2"):
            format_name = "TWO_PAGE"
        elif "full" in length:
            format_name = "FULL"
        return self.materials.create_proposal(
            profile_version_id, track_version_id,
            application_id=row["application_id"], faculty_id=row["faculty_profile_id"],
            format_name=format_name)
