"""Conservative page extraction and application-specific professor decisions."""

from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.discovery import canonical_url


def extract_official_research(text: str, html: str | None = None) -> dict:
    """Extract only explicitly labelled page material; everything awaits human review."""
    if html:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        lines = [(" ".join(node.get_text(" ", strip=True).split()), node.name.startswith("h"))
                 for node in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"])
                 if not node.find_parent("li") or node.name == "li"]
    else:
        lines = [(" ".join(line.split()), False) for line in text.splitlines()]
    lines = [(line[:500], heading) for line, heading in lines if 3 <= len(line) <= 500]
    summary = ""
    topics: list[str] = []
    recent: list[dict] = []
    projects: list[str] = []
    lab = ""
    excerpts: list[str] = []
    section = ""
    for line, heading in lines[:300]:
        header = re.match(r"^(research (?:interests?|areas?|topics?|focus)|(?:selected |recent )?publications?|"
                          r"(?:selected |recent )?(?:papers|research outputs?)|(?:current|ongoing|active) "
                          r"(?:projects?|research|work)|(?:lab|group))\s*:?\s*(.*)$", line, re.I)
        if header:
            label, rest = header.groups()
            section = ("interests" if label.lower().startswith("research") and
                       not re.search(r"outputs?", label, re.I) else
                       "recent" if re.search(r"publications?|papers|outputs?", label, re.I) else
                       "current" if re.search(r"current|ongoing|active", label, re.I) else "lab")
            if not rest:
                continue
            line = rest
        elif heading or re.match(r"^(about|teaching|education|contact|students|news|biography|awards)\b", line, re.I):
            section = ""
            continue
        if section == "interests" and not summary and len(line) >= 8:
            summary = line[:400]
            excerpts.append(line)
            parts = re.split(r"[,;|•]", line)
            topics = [part.strip(" .") for part in parts if 3 <= len(part.strip()) <= 65][:8]
        elif section == "recent" and len(recent) < 8 and len(line) >= 12:
            year = re.search(r"\b(?:19|20)\d{2}\b", line)
            title = re.sub(r"^(?:(?:19|20)\d{2}\s*[·:–-]\s*|\[?\d+\]?\.\s*)", "", line).strip()
            if title and (year or html):
                recent.append({"title": title[:250], "year": int(year.group()) if year else None,
                               "type": "OFFICIAL_PAGE_LISTING"})
                excerpts.append(line)
        elif section == "current" and len(projects) < 6 and len(line) >= 8:
            projects.append(line[:250])
            excerpts.append(line)
        elif section == "lab" and not lab and len(line) <= 100:
            lab = line
            excerpts.append(line)
    if not summary:
        match = re.search(r"\b(?:research interests? (?:include|are)|(?:his|her|their) research "
                          r"(?:focuses on|covers))\s+([^.;\n]{10,250})", text, re.I)
        if match:
            summary = match.group(1).strip()
            excerpts.append(match.group(0))
            topics = [part.strip(" .") for part in re.split(r"[,;]", summary)
                      if 3 <= len(part.strip()) <= 65][:8]
    return {"research_interest_summary": summary, "research_topics": topics,
            "recent_research_candidates": recent, "current_projects": projects,
            "lab": lab, "source_excerpt": " · ".join(excerpts)[:700]}


class FacultyResearch:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        migrate(self.db_path)

    def save_snapshot(self, source_url: str, evidence_id: int, metadata: dict, *,
                      candidate_id: int | None = None, faculty_id: int | None = None) -> int:
        if not (candidate_id or faculty_id):
            raise ValueError("A professor or pending lead is required")
        url = canonical_url(source_url)
        now = utc_now()
        with transaction(self.db_path) as db:
            source = db.execute("SELECT canonical_url,source_type,content_hash FROM source_evidence WHERE id=?",
                                (evidence_id,)).fetchone()
            if not source or source["canonical_url"] != url or source["source_type"] not in {"FACULTY", "LAB"}:
                raise ValueError("Research extraction requires matching source evidence")
            existing = db.execute("""SELECT s.id,s.metadata_json,s.extraction_state,e.content_hash
                FROM faculty_research_snapshots s JOIN source_evidence e ON e.id=s.source_evidence_id
                WHERE s.source_url=? AND (s.faculty_profile_id=? OR s.candidate_id=?)
                ORDER BY faculty_profile_id DESC LIMIT 1""", (url, faculty_id, candidate_id)).fetchone()
            if existing:
                unchanged = (existing["content_hash"] == source["content_hash"]
                             and json.loads(existing["metadata_json"]) == metadata)
                db.execute("""UPDATE faculty_research_snapshots SET source_evidence_id=?,
                    metadata_json=?,extraction_state=CASE WHEN ? THEN extraction_state ELSE 'NEEDS_REVIEW' END,
                    checked_at=?,reviewed_by=CASE WHEN ? THEN reviewed_by ELSE NULL END,
                    reviewed_at=CASE WHEN ? THEN reviewed_at ELSE NULL END,
                    candidate_id=COALESCE(candidate_id,?),faculty_profile_id=COALESCE(faculty_profile_id,?)
                    WHERE id=?""", (evidence_id, json.dumps(metadata), int(unchanged), now,
                                    int(unchanged), int(unchanged), candidate_id, faculty_id,
                                    existing["id"]))
                return existing["id"]
            return db.execute("""INSERT INTO faculty_research_snapshots
                (candidate_id,faculty_profile_id,source_url,source_evidence_id,metadata_json,checked_at)
                VALUES(?,?,?,?,?,?)""", (candidate_id, faculty_id, url, evidence_id,
                                           json.dumps(metadata), now)).lastrowid

    def snapshots(self, *, candidate_id: int | None = None, faculty_id: int | None = None) -> list[dict]:
        if (candidate_id is None) == (faculty_id is None):
            raise ValueError("Choose one research subject")
        field, value = ("candidate_id", candidate_id) if candidate_id else ("faculty_profile_id", faculty_id)
        with connect(self.db_path) as db:
            rows = [dict(r) for r in db.execute(f"""SELECT s.*,e.verification_state AS evidence_state
                FROM faculty_research_snapshots s JOIN source_evidence e ON e.id=s.source_evidence_id
                WHERE s.{field}=? ORDER BY s.checked_at DESC,s.id DESC""", (value,))]
        for row in rows:
            row["metadata"] = json.loads(row.pop("metadata_json"))
        return rows

    def review_snapshot(self, snapshot_id: int, reviewer: str) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        with transaction(self.db_path) as db:
            row = db.execute("SELECT id FROM faculty_research_snapshots WHERE id=?", (snapshot_id,)).fetchone()
            if not row:
                raise ValueError("Research snapshot missing")
            db.execute("""UPDATE faculty_research_snapshots SET extraction_state='VERIFIED',
                reviewed_by=?,reviewed_at=? WHERE id=?""", (reviewer.strip(), utc_now(), snapshot_id))

    def decision(self, application_id: int, *, candidate_id: int | None = None,
                 faculty_id: int | None = None) -> dict | None:
        if (candidate_id is None) == (faculty_id is None):
            raise ValueError("Choose one professor or lead")
        field, value = ("candidate_id", candidate_id) if candidate_id else ("faculty_profile_id", faculty_id)
        with connect(self.db_path) as db:
            row = db.execute(f"SELECT * FROM faculty_decisions WHERE application_id=? AND {field}=?",
                             (application_id, value)).fetchone()
        return dict(row) if row else None

    def decide(self, application_id: int, state: str, *, candidate_id: int | None = None,
               faculty_id: int | None = None) -> None:
        if state not in {"UNDECIDED", "PURSUE", "REJECTED", "ARCHIVED"}:
            raise ValueError("This decision must be made through its own contact workflow")
        if (candidate_id is None) == (faculty_id is None):
            raise ValueError("Choose one professor or lead")
        with transaction(self.db_path) as db:
            app = db.execute("""SELECT COALESCE(p.university,o.institution) AS institution
                FROM applications a LEFT JOIN programmes p ON p.id=a.programme_id
                LEFT JOIN opportunities o ON o.id=a.opportunity_id
                WHERE a.id=? AND a.archived_at IS NULL""", (application_id,)).fetchone()
            field, value, table = (("candidate_id", candidate_id, "faculty_candidates") if candidate_id else
                                   ("faculty_profile_id", faculty_id, "faculty_profiles"))
            professor = db.execute(f"SELECT institution FROM {table} WHERE id=?", (value,)).fetchone()
            if not app or not professor or app["institution"].casefold() != professor["institution"].casefold():
                raise ValueError("Professor and application must belong to the same institution")
            previous = db.execute(f"SELECT id,state FROM faculty_decisions WHERE application_id=? AND {field}=?",
                                  (application_id, value)).fetchone()
            old = previous["state"] if previous else "UNDECIDED"
            if old in {"CONTACTED", "REPLIED"}:
                raise ValueError("Contact history must be managed through outreach")
            now = utc_now()
            if previous:
                db.execute("UPDATE faculty_decisions SET state=?,decided_at=? WHERE id=?",
                           (state, now, previous["id"]))
                decision_id = previous["id"]
            else:
                decision_id = db.execute(f"""INSERT INTO faculty_decisions
                    (application_id,{field},state,decided_at) VALUES(?,?,?,?)""",
                    (application_id, value, state, now)).lastrowid
            db.execute("""INSERT INTO faculty_decision_events
                (decision_id,previous_state,new_state,changed_at) VALUES(?,?,?,?)""",
                (decision_id, old, state, now))
            if state == "PURSUE" and faculty_id:
                db.execute("""INSERT OR IGNORE INTO application_faculty
                    (application_id,faculty_profile_id,linked_at) VALUES(?,?,?)""",
                    (application_id, faculty_id, now))
