"""Evidence-backed, reviewable fit and independent application readiness."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from phd_agent.config import MATCH_WEIGHTS
from phd_agent.applicant_context import ApplicantResearchContextService
from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.discovery import freshness


STOP = {"about", "based", "from", "have", "into", "model", "models", "research", "study", "system", "systems", "that", "their", "this", "using", "with"}


def terms(text: str | None) -> set[str]:
    words = set(re.findall(r"[a-z]{4,}", (text or "").casefold())) - STOP
    normalized = set()
    for word in words:
        if word in {"evaluate", "evaluated", "evaluating", "evaluation", "evaluations"}:
            normalized.add("evaluat")
        elif word in {"reliable", "reliability"}:
            normalized.add("reliab")
        elif word.endswith("s") and len(word) > 5 and not word.endswith("ss"):
            normalized.add(word[:-1])
        else:
            normalized.add(word)
    return normalized


def overlap(left: str | None, right: str | None) -> tuple[float | None, list[str]]:
    a, b = terms(left), terms(right)
    if not a or not b:
        return None, []
    shared = sorted(a & b)
    return round(10 * len(shared) / max(1, len(a)), 2), shared


def contact_policy_state(value: str | None) -> str:
    """Only an explicit reviewed policy can authorize faculty contact."""
    policy = (value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if policy in {"ALLOWED", "CONTACT_ALLOWED", "CONTACT_ENCOURAGED", "CONTACT_REQUIRED"} or policy.startswith(("ALLOWED:", "CONTACT_ALLOWED:")):
        return "PASS"
    if policy in {"NOT_ALLOWED", "DO_NOT_CONTACT", "CONTACT_PROHIBITED", "NO_UNSOLICITED_CONTACT"}:
        return "BLOCK"
    return "UNKNOWN"


class MatchEngine:
    def __init__(self, db_path: Path | str, weights: dict[str, float] | None = None):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.weights = dict(weights or MATCH_WEIGHTS)
        if set(self.weights) != set(MATCH_WEIGHTS) or any(v < 0 for v in self.weights.values()) or abs(sum(self.weights.values()) - 1) > 1e-6:
            raise ValueError("Match weights must contain the five components and sum to 1")

    def assess(self, faculty_id: int, profile_version_id: int, track_version_id: int,
               application_id: int | None = None) -> dict:
        with connect(self.db_path) as db:
            profile = db.execute("SELECT * FROM profile_versions WHERE id=?", (profile_version_id,)).fetchone()
            track = db.execute("""SELECT v.*,t.profile_id,t.title,t.status AS track_status
                FROM research_track_versions v JOIN research_tracks t ON t.id=v.track_id WHERE v.id=?""", (track_version_id,)).fetchone()
            faculty = db.execute("SELECT * FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
            if not profile or profile["approval_state"] != "APPROVED" or not track or track["approval_state"] != "APPROVED" or track["profile_id"] != profile["profile_id"] or track["track_status"] != "ACTIVE":
                raise ValueError("An approved profile and active approved research track are required")
            if not faculty:
                raise ValueError("Faculty not found")
            claims = [dict(r) for r in db.execute("""SELECT cr.* FROM profile_version_claims pvc
                JOIN claim_revisions cr ON cr.id=pvc.claim_revision_id WHERE pvc.profile_version_id=?
                AND cr.review_status='APPROVED' AND cr.approved_for_application=1""", (profile_version_id,))]
            publications = [dict(r) for r in db.execute("""SELECT p.* FROM publications p
                JOIN source_evidence e ON e.id=p.source_evidence_id
                WHERE p.faculty_profile_id=? AND e.verification_state='VERIFIED' ORDER BY p.year DESC""", (faculty_id,))]
            links = [dict(r) for r in db.execute("""SELECT l.fact_type,e.id,e.retrieved_at,e.verification_state
                FROM faculty_evidence_links l JOIN source_evidence e ON e.id=l.source_evidence_id
                WHERE l.faculty_profile_id=?""", (faculty_id,))]
            app = db.execute("SELECT * FROM applications WHERE id=?", (application_id,)).fetchone() if application_id else None
            if application_id and not app:
                raise ValueError("Application not found")
            opportunity = db.execute("SELECT * FROM opportunities WHERE id=?", (app["opportunity_id"],)).fetchone() if app and app["opportunity_id"] else None
            programme = db.execute("SELECT * FROM programmes WHERE id=?", (app["programme_id"],)).fetchone() if app and app["programme_id"] else None
            deadlines = [dict(r) for r in db.execute("SELECT * FROM deadlines WHERE application_id=?", (application_id,))] if app else []
            requirements = [dict(r) for r in db.execute("SELECT * FROM requirements WHERE application_id=?", (application_id,))] if app else []
        if not claims:
            raise ValueError("Approved profile contains no application-approved claims")
        faculty_text = faculty["research_topics"] or ""
        topic_links = [x for x in links if x["fact_type"] == "TOPICS" and x["verification_state"] == "VERIFIED"]
        topic_evidence = [x["id"] for x in topic_links]
        work = [p for p in publications if p["year"] and p["year"] >= date.today().year - 5]
        work_text = " ".join(p["title"] + " " + (p["abstract_text"] or "") for p in work)
        fallback_claims = [c for c in claims if c["classification"] != "ASPIRATION"]
        claim_text = " ".join(c["claim_text"] for c in fallback_claims)
        experience_claim_ids = [c["id"] for c in fallback_claims]
        context_service = ApplicantResearchContextService(self.db_path)
        applicant_context = context_service.build_current(
            profile_id=track["profile_id"], profile_version_id=profile_version_id,
            track_version_id=track_version_id)
        context_retrieval = context_service.retrieve(
            applicant_context["id"], "FINAL_RESEARCH_FIT",
            faculty_text + " " + work_text + " " + track["research_problem"],
            top_k=6, use="application", include_proposed=True,
        )
        demonstrated = [item for item in context_retrieval.items if item.classification == "DEMONSTRATED"]
        if demonstrated:
            claim_text = " ".join(item.text for item in demonstrated)
            experience_claim_ids = sorted({claim_id for item in demonstrated
                                           for claim_id in item.claim_revision_ids})
        specs = {
            "topic": (track["title"] + " " + track["research_problem"], faculty_text, [], topic_evidence),
            "method": (track["proposed_methodology"], faculty_text + " " + work_text, [], topic_evidence + [p["source_evidence_id"] for p in work]),
            "recent_work": (track["research_problem"] + " " + track["proposed_methodology"], work_text, [], [p["source_evidence_id"] for p in work]),
            "experience": (claim_text, faculty_text + " " + work_text,
                experience_claim_ids,
                topic_evidence + [p["source_evidence_id"] for p in work]),
            "proposed_direction": (track["research_questions"] + " " + track["expected_contribution"], faculty_text + " " + work_text, json.loads(track["supporting_claim_revision_ids_json"]), topic_evidence + [p["source_evidence_id"] for p in work]),
        }
        components = {}
        for name, (left, right, ids, evidence) in specs.items():
            # No verified external evidence means unknown, even when faculty text exists.
            score, shared = overlap(left, right) if evidence else (None, [])
            components[name] = {"score": score, "weight": self.weights[name], "claim_revision_ids": ids,
                "research_track_version_id": track_version_id, "evidence_ids": sorted(set(evidence)),
                "explanation": "Shared source terms: " + ", ".join(shared) if shared else "No shared source terms" if score is not None else "Evidence or applicant text unavailable",
                "unknowns": [] if score is not None else ["Insufficient verified source or applicant text"],
                "confidence": round(min(1, len(evidence) / 2), 2) if score is not None else 0,
                "coverage": 1 if score is not None else 0}
        covered = sum(self.weights[k] for k, v in components.items() if v["score"] is not None)
        fit = round(sum(self.weights[k] * v["score"] for k, v in components.items() if v["score"] is not None) / covered, 2) if covered else None
        readiness = self._readiness(faculty, links, app, opportunity, programme, deadlines, requirements)
        unknowns = [name for name, value in components.items() if value["score"] is None] + [name for name, value in readiness.items() if value["state"] == "UNKNOWN"]
        evidence_ids = sorted({i for value in components.values() for i in value["evidence_ids"]} | {r["source_evidence_id"] for r in requirements} | {r["source_evidence_id"] for r in deadlines})
        with transaction(self.db_path) as db:
            assessment_id = db.execute("""INSERT INTO match_assessments
                (faculty_profile_id,application_id,research_track_version_id,research_fit,application_readiness,
                 research_fit_components_json,application_readiness_components_json,unknowns_json,evidence_ids_json,assessed_at,notes)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (faculty_id, application_id, track_version_id, fit, None,
                json.dumps(components), json.dumps(readiness), json.dumps(unknowns), json.dumps(evidence_ids), utc_now(),
                f"profile_version_id={profile_version_id}; coverage={covered:.2f}")).lastrowid
        context_service.link_output(
            "MATCH_ASSESSMENT", assessment_id, applicant_context["id"], context_retrieval.id,
            model="research-fit-v1", prompt_version="deterministic-components-v1",
        )
        return {"id": assessment_id, "research_fit": fit, "research_fit_coverage": round(covered, 2),
            "components": components, "application_readiness": readiness, "unknowns": unknowns,
            "publication_ids": [p["id"] for p in work], "evidence_ids": evidence_ids,
            "last_evidence_refresh": max((x["retrieved_at"] for x in links), default=None),
            "applicant_context_id": applicant_context["id"],
            "context_retrieval_id": context_retrieval.id,
            "demonstrated_overlap": [item.text for item in context_retrieval.items
                                     if item.classification == "DEMONSTRATED"],
            "proposed_overlap": [item.text for item in context_retrieval.items
                                  if item.classification == "PROPOSED"]}

    @staticmethod
    def _readiness(faculty, links, app, opportunity, programme, deadlines, requirements):
        def state(ok, known=True, evidence=None):
            return {"state": "PASS" if ok else "BLOCK" if known else "UNKNOWN", "evidence_ids": evidence or []}
        affiliation = [x for x in links if x["fact_type"] == "AFFILIATION" and x["verification_state"] == "VERIFIED"]
        current_affiliation = any(freshness(x["retrieved_at"], "FACULTY_AFFILIATION") == "CURRENT" for x in affiliation)
        route = app["portal_url"] if app else None
        route = route or (opportunity["application_route"] if opportunity else None) or (programme["portal_url"] if programme else None)
        main_deadline = next((x for x in deadlines if x["deadline_type"] == "APPLICATION"), None)
        return {
            "faculty_affiliation": state(current_affiliation, bool(affiliation), [x["id"] for x in affiliation]),
            "programme_or_opening": state(bool(programme or opportunity), bool(app)),
            "application_route": state(bool(route), bool(route)),
            "deadline": state(bool(main_deadline and main_deadline["due_at"][:10] >= date.today().isoformat() and main_deadline["verification_state"] == "VERIFIED"), bool(main_deadline), [main_deadline["source_evidence_id"]] if main_deadline else []),
            "eligibility": state(
                bool(app and app["eligibility_state"] == "ELIGIBLE"),
                bool(app and app["eligibility_state"] and app["eligibility_state"] not in {"UNKNOWN", "NEEDS_REVIEW"}),
            ),
            "contact_policy": {"state": contact_policy_state(opportunity["contact_policy"] if opportunity else None),
                               "evidence_ids": [opportunity["source_evidence_id"]] if opportunity and opportunity["source_evidence_id"] else []},
            "supervisor_opening": state(bool(opportunity and opportunity["opportunity_type"] == "ADVERTISED_POSITION" and opportunity["opening_status"] == "OPEN"), bool(opportunity and opportunity["opportunity_type"] == "ADVERTISED_POSITION" and opportunity["opening_status"] != "UNKNOWN"), [opportunity["source_evidence_id"]] if opportunity and opportunity["source_evidence_id"] else []),
            "critical_requirements": state(bool(requirements) and all(r["requirement_state"] != "UNKNOWN" for r in requirements), bool(requirements), [r["source_evidence_id"] for r in requirements]),
        }

    def review(self, assessment_id: int, reviewer: str, annotation: str, override: dict | None = None) -> int:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        override = override or {}
        if set(override) - {"research_fit", "application_readiness"}:
            raise ValueError("Only fit and readiness annotations may be overridden")
        if "research_fit" in override and override["research_fit"] is not None and not 0 <= override["research_fit"] <= 10:
            raise ValueError("Fit override must be 0–10")
        with transaction(self.db_path) as db:
            if not db.execute("SELECT 1 FROM match_assessments WHERE id=?", (assessment_id,)).fetchone():
                raise ValueError("Assessment not found")
            version = db.execute("SELECT COALESCE(MAX(version_number),0)+1 FROM match_reviews WHERE assessment_id=?", (assessment_id,)).fetchone()[0]
            return db.execute("""INSERT INTO match_reviews
                (assessment_id,version_number,reviewer,annotation,override_json,reviewed_at) VALUES(?,?,?,?,?,?)""",
                (assessment_id, version, reviewer.strip(), annotation, json.dumps(override), utc_now())).lastrowid
