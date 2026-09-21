"""Versioned, CV-grounded applicant context and task-specific retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from phd_agent.db import connect, migrate, transaction, utc_now


CONTEXT_VERSION = "applicant-context-v1"
RETRIEVAL_VERSION = "context-retrieval-v1"
_STOP = {
    "about", "after", "against", "applicant", "application", "based", "from", "have",
    "into", "research", "study", "system", "systems", "that", "their", "these", "this",
    "through", "using", "with", "work", "working",
}
_TERM_GROUPS = (
    {"agent", "agentic", "autonomous", "multiagent", "planning", "workflow"},
    {"evaluat", "benchmark", "testing", "audit", "soundness", "reliab", "robust"},
    {"retrieval", "rag", "grounded", "grounding", "hallucination"},
    {"vision", "visual", "multimodal", "yolo", "image", "video"},
    {"robot", "robotic", "exoskeleton", "hardware", "control"},
    {"language", "llm", "nlp", "transformer", "generative"},
)


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_terms(text: str | None) -> set[str]:
    words = set(re.findall(r"[a-z][a-z0-9+-]{2,}", (text or "").casefold())) - _STOP
    result = set()
    for word in words:
        if word in {"evaluate", "evaluated", "evaluating", "evaluation", "evaluations"}:
            result.add("evaluat")
        elif word in {"reliable", "reliability"}:
            result.add("reliab")
        elif word in {"robots", "robotics"}:
            result.add("robot")
        elif word.endswith("s") and len(word) > 5 and not word.endswith("ss"):
            result.add(word[:-1])
        else:
            result.add(word)
    return result


def _expanded_overlap(query: set[str], item: set[str]) -> set[str]:
    shared = query & item
    for group in _TERM_GROUPS:
        if query & group and item & group:
            shared |= query & group
            shared |= item & group
    return shared


@dataclass(frozen=True)
class RetrievedItem:
    item_id: str
    kind: str
    text: str
    classification: str
    cv_section: str | None
    claim_revision_ids: tuple[int, ...]
    source_document_version_ids: tuple[int, ...]
    source_evidence_ids: tuple[int, ...]
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalResult:
    id: int
    applicant_context_id: int
    task_type: str
    query_text: str
    items: tuple[RetrievedItem, ...]

    @property
    def claim_revision_ids(self) -> tuple[int, ...]:
        return tuple(sorted({claim for item in self.items for claim in item.claim_revision_ids}))

    @property
    def evidence_ids(self) -> tuple[int, ...]:
        return tuple(sorted({evidence for item in self.items for evidence in item.source_evidence_ids}))


class ApplicantResearchContextService:
    """Build immutable approved context snapshots and record every retrieval that influences output."""

    OUTPUT_TABLES = {
        "MATCH_ASSESSMENT": "match_assessments",
        "GENERATED_ARTIFACT": "generated_artifacts",
        "OUTREACH_PACKAGE": "outreach_packages",
        "PORTAL_ANSWER": "answer_library",
        "PROGRAMME_CANDIDATE": "programme_candidates",
        "BROWSER_FILL_PLAN": "browser_fill_plans",
    }

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        migrate(self.db_path)

    def available_inputs(self) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(row) for row in db.execute("""SELECT m.id AS master_cv_version_id,
                m.profile_id,m.profile_version_id,m.version_number AS master_cv_version,
                p.owner_name,t.id AS research_track_id,t.title,t.priority,
                v.id AS research_track_version_id,v.version_number AS research_track_version
                FROM master_cv_versions m
                JOIN applicant_profiles p ON p.id=m.profile_id
                JOIN research_tracks t ON t.profile_id=m.profile_id AND t.status='ACTIVE'
                JOIN research_track_versions v ON v.track_id=t.id
                WHERE m.approval_state='APPROVED' AND v.approval_state='APPROVED'
                  AND v.version_number=(SELECT MAX(v2.version_number) FROM research_track_versions v2
                    WHERE v2.track_id=t.id AND v2.approval_state='APPROVED')
                ORDER BY m.id DESC,t.priority,t.id""")]

    def build_current(self, *, profile_id: int | None = None,
                      profile_version_id: int | None = None,
                      track_version_id: int | None = None) -> dict:
        inputs = self.available_inputs()
        if profile_id is not None:
            inputs = [row for row in inputs if row["profile_id"] == profile_id]
        if profile_version_id is not None:
            inputs = [row for row in inputs if row["profile_version_id"] == profile_version_id]
        if track_version_id is not None:
            inputs = [row for row in inputs if row["research_track_version_id"] == track_version_id]
        if not inputs:
            raise ValueError("Approve a Master CV, profile snapshot, and active research direction first")
        selected = inputs[0]
        return self.build(selected["profile_version_id"], selected["master_cv_version_id"],
                          selected["research_track_version_id"])

    def build(self, profile_version_id: int, master_cv_version_id: int,
              research_track_version_id: int) -> dict:
        with connect(self.db_path) as db:
            profile = db.execute("""SELECT v.*,p.owner_name,p.id AS applicant_profile_id
                FROM profile_versions v JOIN applicant_profiles p ON p.id=v.profile_id
                WHERE v.id=?""", (profile_version_id,)).fetchone()
            master = db.execute("SELECT * FROM master_cv_versions WHERE id=?", (master_cv_version_id,)).fetchone()
            track = db.execute("""SELECT v.*,t.profile_id,t.title,t.status AS track_status
                FROM research_track_versions v JOIN research_tracks t ON t.id=v.track_id
                WHERE v.id=?""", (research_track_version_id,)).fetchone()
            claims = [dict(row) for row in db.execute("""SELECT cr.*,c.category
                FROM profile_version_claims pvc
                JOIN claim_revisions cr ON cr.id=pvc.claim_revision_id
                JOIN claims c ON c.id=cr.claim_id
                WHERE pvc.profile_version_id=? AND cr.review_status='APPROVED'
                ORDER BY cr.id""", (profile_version_id,))]
        if not profile or profile["approval_state"] != "APPROVED":
            raise ValueError("An approved applicant profile snapshot is required")
        if not master or master["approval_state"] != "APPROVED" or master["profile_version_id"] != profile_version_id:
            raise ValueError("The approved Master CV must use the selected approved profile snapshot")
        if (not track or track["approval_state"] != "APPROVED" or track["track_status"] != "ACTIVE"
                or track["profile_id"] != profile["applicant_profile_id"]):
            raise ValueError("An active approved research direction for this applicant is required")
        claim_map = {row["id"]: row for row in claims}
        items: list[dict] = []
        cv_claim_ids: set[int] = set()
        source_document_ids: set[int] = set()
        source_evidence_ids: set[int] = set()
        sections = json.loads(master["sections_json"])
        for section_index, section in enumerate(sections):
            for bullet_index, bullet in enumerate(section.get("bullets", [])):
                claim_ids = sorted(set(bullet.get("claim_revision_ids", [])))
                if not claim_ids or not set(claim_ids) <= claim_map.keys():
                    raise ValueError("Every Master CV statement must resolve to approved profile claims")
                cv_claim_ids.update(claim_ids)
                linked = [claim_map[claim_id] for claim_id in claim_ids]
                document_ids = sorted({row["source_document_version_id"] for row in linked
                                       if row["source_document_version_id"]})
                evidence_ids = sorted({row["source_evidence_id"] for row in linked if row["source_evidence_id"]})
                source_document_ids.update(document_ids)
                source_evidence_ids.update(evidence_ids)
                items.append({
                    "item_id": f"cv:{section_index}:{bullet_index}",
                    "kind": "CV_EXPERIENCE",
                    "text": bullet["text"].strip(),
                    "classification": "DEMONSTRATED" if all(row["classification"] == "FACT" for row in linked)
                        else "SUPPORTED_INFERENCE",
                    "category": linked[0]["category"],
                    "normalized_claim_types": sorted({row["normalized_claim_type"] for row in linked}),
                    "cv_section": section["name"],
                    "claim_revision_ids": claim_ids,
                    "source_document_version_ids": document_ids,
                    "source_evidence_ids": evidence_ids,
                    "approved_for_application": all(row["approved_for_application"] for row in linked),
                    "approved_for_outreach": all(row["approved_for_outreach"] for row in linked),
                })
        for claim in claims:
            if claim["id"] in cv_claim_ids:
                continue
            document_ids = [claim["source_document_version_id"]] if claim["source_document_version_id"] else []
            evidence_ids = [claim["source_evidence_id"]] if claim["source_evidence_id"] else []
            source_document_ids.update(document_ids)
            source_evidence_ids.update(evidence_ids)
            classification = "PROPOSED" if claim["classification"] == "ASPIRATION" else (
                "DEMONSTRATED" if claim["classification"] == "FACT" else "SUPPORTED_INFERENCE")
            items.append({
                "item_id": f"claim:{claim['id']}", "kind": "PROFILE_CLAIM",
                "text": claim["claim_text"], "classification": classification,
                "category": claim["category"], "cv_section": None,
                "normalized_claim_types": [claim["normalized_claim_type"]],
                "claim_revision_ids": [claim["id"]],
                "source_document_version_ids": document_ids,
                "source_evidence_ids": evidence_ids,
                "approved_for_application": bool(claim["approved_for_application"]),
                "approved_for_outreach": bool(claim["approved_for_outreach"]),
            })
        track_text = " ".join(filter(None, (
            track["title"], track["research_problem"], track["research_questions"],
            track["proposed_methodology"], track["expected_contribution"],
        )))
        items.append({
            "item_id": f"track:{track['id']}", "kind": "PROPOSED_DIRECTION",
            "text": track_text, "classification": "PROPOSED", "category": "RESEARCH_DIRECTION",
            "normalized_claim_types": [],
            "cv_section": None,
            "claim_revision_ids": json.loads(track["supporting_claim_revision_ids_json"]),
            "source_document_version_ids": [], "source_evidence_ids": [],
            "approved_for_application": True, "approved_for_outreach": True,
        })
        context = {
            "schema_version": 1,
            "owner_name": profile["owner_name"],
            "profile_id": profile["applicant_profile_id"],
            "profile_version_id": profile_version_id,
            "master_cv_version_id": master_cv_version_id,
            "research_track_version_id": research_track_version_id,
            "research_track": {field: track[field] for field in (
                "title", "research_problem", "research_questions", "proposed_methodology",
                "evaluation_strategy", "expected_contribution",
            )},
            "items": items,
        }
        digest = hashlib.sha256(_dump(context).encode()).hexdigest()
        with connect(self.db_path) as db:
            existing = db.execute("SELECT * FROM applicant_research_contexts WHERE context_sha256=?", (digest,)).fetchone()
        if existing:
            return self.get(existing["id"])
        basis = {
            "profile_version_id": profile_version_id,
            "master_cv_version_id": master_cv_version_id,
            "research_track_version_id": research_track_version_id,
            "claim_revision_ids": sorted(claim_map),
            "source_document_version_ids": sorted(source_document_ids),
            "source_evidence_ids": sorted(source_evidence_ids),
        }
        with transaction(self.db_path) as db:
            version = db.execute("SELECT COALESCE(MAX(version_number),0)+1 FROM applicant_research_contexts WHERE profile_id=?",
                                 (profile["applicant_profile_id"],)).fetchone()[0]
            context_id = db.execute("""INSERT INTO applicant_research_contexts
                (profile_id,profile_version_id,master_cv_version_id,research_track_version_id,
                 version_number,context_sha256,context_json,approval_basis_json,provider,model,prompt_version,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (
                profile["applicant_profile_id"], profile_version_id, master_cv_version_id,
                research_track_version_id, version, digest, _dump(context), _dump(basis),
                "deterministic", CONTEXT_VERSION, "none", utc_now(),
            )).lastrowid
            previous_links = db.execute("""SELECT l.output_type,l.output_id,l.applicant_context_id
                FROM output_context_links l JOIN applicant_research_contexts c ON c.id=l.applicant_context_id
                WHERE c.profile_id=? AND c.id!=?""", (profile["applicant_profile_id"], context_id)).fetchall()
            for link in previous_links:
                db.execute("""INSERT OR IGNORE INTO context_staleness
                    (output_type,output_id,previous_context_id,current_context_id,reason,detected_at)
                    VALUES(?,?,?,?,?,?)""", (
                    link["output_type"], link["output_id"], link["applicant_context_id"], context_id,
                    "Applicant context changed — review recommended", utc_now(),
                ))
        return self.get(context_id)

    def get(self, context_id: int) -> dict:
        with connect(self.db_path) as db:
            row = db.execute("SELECT * FROM applicant_research_contexts WHERE id=?", (context_id,)).fetchone()
        if not row:
            raise ValueError("Applicant research context not found")
        result = dict(row)
        result["context"] = json.loads(result.pop("context_json"))
        result["approval_basis"] = json.loads(result.pop("approval_basis_json"))
        return result

    def latest(self, profile_id: int | None = None) -> dict | None:
        sql = "SELECT id FROM applicant_research_contexts"
        args: tuple = ()
        if profile_id is not None:
            sql += " WHERE profile_id=?"
            args = (profile_id,)
        sql += " ORDER BY id DESC LIMIT 1"
        with connect(self.db_path) as db:
            row = db.execute(sql, args).fetchone()
        return self.get(row["id"]) if row else None

    def retrieve(self, context_id: int, task_type: str, query_text: str, *,
                 top_k: int = 3, use: str = "application",
                 include_proposed: bool = True) -> RetrievalResult:
        if use not in {"application", "outreach"} or top_k < 1 or top_k > 12:
            raise ValueError("Invalid retrieval use or result limit")
        query = query_text.strip()
        if not query:
            raise ValueError("Retrieval needs a task-specific query")
        context = self.get(context_id)
        query_terms = normalized_terms(query)
        scored: list[tuple[float, dict, set[str]]] = []
        for item in context["context"]["items"]:
            if not item[f"approved_for_{use}"]:
                continue
            if not include_proposed and item["classification"] == "PROPOSED":
                continue
            item_terms = normalized_terms(item["text"])
            matched = _expanded_overlap(query_terms, item_terms)
            if not matched:
                continue
            score = len(matched) * 3 + len(query_terms & item_terms) * 2
            if item["classification"] == "DEMONSTRATED":
                score += 1.0
            if item["kind"] == "CV_EXPERIENCE":
                score += 0.5
            score += min(len(item_terms), 30) / 100
            scored.append((score, item, matched))
        scored.sort(key=lambda entry: (-entry[0], entry[1]["item_id"]))
        selected = scored[:top_k]
        items = tuple(RetrievedItem(
            item_id=item["item_id"], kind=item["kind"], text=item["text"],
            classification=item["classification"], cv_section=item["cv_section"],
            claim_revision_ids=tuple(item["claim_revision_ids"]),
            source_document_version_ids=tuple(item["source_document_version_ids"]),
            source_evidence_ids=tuple(item["source_evidence_ids"]),
            score=round(score, 2), matched_terms=tuple(sorted(matched)),
        ) for score, item, matched in selected)
        payload = [item.__dict__ for item in items]
        claim_ids = sorted({claim for item in items for claim in item.claim_revision_ids})
        sections = sorted({item.cv_section for item in items if item.cv_section})
        evidence_ids = sorted({evidence for item in items for evidence in item.source_evidence_ids})
        with transaction(self.db_path) as db:
            retrieval_id = db.execute("""INSERT INTO context_retrievals
                (applicant_context_id,task_type,query_text,selected_items_json,claim_revision_ids_json,
                 cv_sections_json,evidence_ids_json,provider,model,prompt_version,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (
                context_id, task_type.strip().upper(), query, _dump(payload), _dump(claim_ids),
                _dump(sections), _dump(evidence_ids), "deterministic", RETRIEVAL_VERSION, "none", utc_now(),
            )).lastrowid
        return RetrievalResult(retrieval_id, context_id, task_type.strip().upper(), query, items)

    def link_output(self, output_type: str, output_id: int, context_id: int,
                    retrieval_id: int | None = None, *, provider: str = "deterministic",
                    model: str = CONTEXT_VERSION, prompt_version: str = "none") -> int:
        output_type = output_type.strip().upper()
        table = self.OUTPUT_TABLES.get(output_type)
        if not table:
            raise ValueError("Unsupported output type")
        context = self.get(context_id)
        with connect(self.db_path) as db:
            if not db.execute(f"SELECT 1 FROM {table} WHERE id=?", (output_id,)).fetchone():
                raise ValueError("Output record not found")
            if retrieval_id and not db.execute("""SELECT 1 FROM context_retrievals
                    WHERE id=? AND applicant_context_id=?""", (retrieval_id, context_id)).fetchone():
                raise ValueError("Retrieval does not belong to this applicant context")
        with transaction(self.db_path) as db:
            cursor = db.execute("""INSERT OR IGNORE INTO output_context_links
                (output_type,output_id,applicant_context_id,retrieval_id,context_sha256,
                 provider,model,prompt_version,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)""", (
                output_type, output_id, context_id, retrieval_id, context["context_sha256"],
                provider, model, prompt_version, utc_now(),
            ))
            if cursor.rowcount == 1:
                return cursor.lastrowid
            row = db.execute("""SELECT id FROM output_context_links WHERE output_type=? AND output_id=?
                AND applicant_context_id=? AND retrieval_id IS ?""",
                (output_type, output_id, context_id, retrieval_id)).fetchone()
            return row["id"]

    def stale_outputs(self, *, unresolved_only: bool = True) -> list[dict]:
        sql = "SELECT * FROM context_staleness"
        if unresolved_only:
            sql += " WHERE reviewed_at IS NULL"
        sql += " ORDER BY detected_at DESC,id DESC"
        with connect(self.db_path) as db:
            return [dict(row) for row in db.execute(sql)]

    def acknowledge_stale(self, staleness_id: int, reviewer: str) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        with transaction(self.db_path) as db:
            changed = db.execute("""UPDATE context_staleness SET reviewed_at=?,reviewed_by=?
                WHERE id=? AND reviewed_at IS NULL""", (utc_now(), reviewer.strip(), staleness_id)).rowcount
        if not changed:
            raise ValueError("Staleness record not found or already reviewed")

    def profile_review(self, context_id: int) -> dict:
        context = self.get(context_id)
        items = context["context"]["items"]
        with connect(self.db_path) as db:
            answers = [dict(row) for row in db.execute("""SELECT * FROM answer_library
                WHERE approval_state='APPROVED' ORDER BY field_key,version_number DESC""")]
            documents = [dict(row) for row in db.execute("""SELECT d.document_type,d.title,v.id AS version_id,
                v.original_filename,v.approval_state,v.sensitivity
                FROM documents d JOIN document_versions v ON v.document_id=d.id
                WHERE v.approval_state='APPROVED' ORDER BY d.document_type,d.title""")]
        return {
            "context": context,
            "demonstrated": [item for item in items if item["classification"] == "DEMONSTRATED"],
            "supported_inferences": [item for item in items if item["classification"] == "SUPPORTED_INFERENCE"],
            "proposed": [item for item in items if item["classification"] == "PROPOSED"],
            "approved_answers": answers,
            "approved_documents": documents,
        }

    def bootstrap_profile_answers(self, context_id: int) -> list[int]:
        """Create compact draft reusable answers from approved context for operator review."""
        context = self.get(context_id)
        items = context["context"]["items"]
        groups: dict[str, list[dict]] = {"FULL_NAME": []}
        category_keys = {
            "EDUCATION": "EDUCATION", "DEGREE": "EDUCATION",
            "EMPLOYMENT": "OTHER", "RESEARCH_PROJECT": "RESEARCH_INTERESTS",
            "PROJECT": "RESEARCH_INTERESTS", "PUBLICATION": "PUBLICATIONS_SUMMARY",
            "PATENT": "PUBLICATIONS_SUMMARY", "AWARD": "AWARDS", "HONOUR": "AWARDS",
        }
        normalized_keys = {
            "email": "EMAIL", "email_address": "EMAIL",
            "phone": "PHONE", "phone_number": "PHONE", "mobile": "PHONE",
            "nationality": "NATIONALITY", "citizenship": "NATIONALITY",
            "degree_dates": "DEGREE_DATES", "education_dates": "DEGREE_DATES",
            "referee": "REFEREES", "referees": "REFEREES",
            "english_test": "ENGLISH_TEST", "ielts": "ENGLISH_TEST", "toefl": "ENGLISH_TEST",
        }
        for item in items:
            if item["classification"] not in {"DEMONSTRATED", "SUPPORTED_INFERENCE"}:
                continue
            key = next((normalized_keys.get(value.casefold())
                        for value in item.get("normalized_claim_types", [])
                        if normalized_keys.get(value.casefold())), None)
            key = key or category_keys.get((item.get("category") or "").upper())
            if key:
                groups.setdefault(key, []).append(item)
        candidates = {"FULL_NAME": ("Full legal name", context["context"]["owner_name"], None)}
        labels = {
            "EDUCATION": "Education summary", "RESEARCH_INTERESTS": "Research experience summary",
            "PUBLICATIONS_SUMMARY": "Publications and patents", "AWARDS": "Awards and honours",
            "OTHER": "Professional experience summary", "EMAIL": "Email address",
            "PHONE": "Phone number", "NATIONALITY": "Nationality or citizenship",
            "DEGREE_DATES": "Degree dates", "REFEREES": "Referee details",
            "ENGLISH_TEST": "English language qualification",
        }
        for key, selected in groups.items():
            if key == "FULL_NAME" or not selected:
                continue
            candidates[key] = (labels[key], " ".join(item["text"] for item in selected[:3]),
                               selected[0]["claim_revision_ids"][0] if selected[0]["claim_revision_ids"] else None)
        created = []
        with transaction(self.db_path) as db:
            for key, (label, value, claim_id) in candidates.items():
                duplicate = db.execute("""SELECT 1 FROM answer_library
                    WHERE field_key=? AND value_text=? AND approval_state IN ('DRAFT','APPROVED')""",
                    (key, value)).fetchone()
                if duplicate:
                    continue
                version = db.execute("SELECT COALESCE(MAX(version_number),0)+1 FROM answer_library WHERE field_key=?",
                                     (key,)).fetchone()[0]
                answer_id = db.execute("""INSERT INTO answer_library
                    (field_key,label,value_text,version_number,claim_revision_id,approval_state,created_at)
                    VALUES(?,?,?,?,?,'DRAFT',?)""", (key, label, value, version, claim_id, utc_now())).lastrowid
                created.append(answer_id)
        for answer_id in created:
            self.link_output("PORTAL_ANSWER", answer_id, context_id,
                             provider="deterministic", model="profile-bootstrap-v1",
                             prompt_version="approved-context-v1")
        self.record_workload("FIELDS_REUSED_FROM_PROFILE", len(created), entity_type="APPLICANT_CONTEXT",
                             entity_id=context_id)
        return created

    def record_workload(self, event_type: str, count: int, *, entity_type: str | None = None,
                        entity_id: int | None = None, metadata: dict | None = None) -> int:
        if count < 0:
            raise ValueError("Workload count cannot be negative")
        with transaction(self.db_path) as db:
            return db.execute("""INSERT INTO workload_events
                (event_type,count_value,entity_type,entity_id,metadata_json,created_at)
                VALUES(?,?,?,?,?,?)""", (
                event_type.strip().upper(), count, entity_type, entity_id, _dump(metadata or {}), utc_now(),
            )).lastrowid

    def workload_summary(self) -> dict[str, int]:
        with connect(self.db_path) as db:
            return {row["event_type"]: row["total"] for row in db.execute(
                "SELECT event_type,SUM(count_value) AS total FROM workload_events GROUP BY event_type")}


def flatten_retrieval_text(items: Iterable[RetrievedItem], limit: int = 2) -> str:
    """Return a short factual excerpt for reviewed generation paths."""
    return " ".join(item.text for item in list(items)[:limit])
