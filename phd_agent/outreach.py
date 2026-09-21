"""Reviewed, evidence-bound faculty outreach. No background sender is started here."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import getaddresses, parseaddr
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.config import MATCH_WEIGHTS
from phd_agent.discovery import freshness
from phd_agent.documents import DocumentVault
from phd_agent.gmail_gateway import GmailGateway
from phd_agent.matching import contact_policy_state, terms
from phd_agent.materials import MaterialStudio
from phd_agent.packages import PackageBuilder


QUALITY_VERSION = "outreach-quality-v1"
POLICY_VERSION = "outreach-policy-v1"
STATES = {"DRAFT", "NEEDS_REVIEW", "BLOCKED", "APPROVED", "SCHEDULED", "SENDING",
          "SENT", "FAILED", "AMBIGUOUS_SEND", "BOUNCED", "REPLIED", "FOLLOW_UP_DUE",
          "CANCELLED", "DO_NOT_CONTACT"}
TRANSITIONS = {
    "DRAFT": {"NEEDS_REVIEW", "BLOCKED", "CANCELLED"},
    "NEEDS_REVIEW": {"APPROVED", "BLOCKED", "CANCELLED"},
    "BLOCKED": {"CANCELLED"},
    "APPROVED": {"SCHEDULED", "SENDING", "CANCELLED", "DO_NOT_CONTACT"},
    "SCHEDULED": {"APPROVED", "SENDING", "CANCELLED", "DO_NOT_CONTACT"},
    "SENDING": {"SENT", "FAILED", "AMBIGUOUS_SEND"},
    "SENT": {"REPLIED", "BOUNCED", "FOLLOW_UP_DUE"},
    "FAILED": {"CANCELLED"},
    "AMBIGUOUS_SEND": {"SENT", "CANCELLED"},
    "BOUNCED": set(), "REPLIED": set(), "FOLLOW_UP_DUE": set(),
    "CANCELLED": set(), "DO_NOT_CONTACT": set(),
}
PLACEHOLDER = re.compile(r"\[(?:insert|name|professor|university|paper|todo)[^]]*\]|\b(?:TBD|TODO|Lorem ipsum)\b", re.I)


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _email(value: str | None) -> str:
    return parseaddr(value or "")[1].strip().casefold()


@dataclass(frozen=True)
class Attachment:
    document_version_id: int
    requirement_id: int
    document_type: str
    filename: str
    sha256: str
    reason: str
    order: int


@dataclass(frozen=True)
class OutreachContext:
    applicant_name: str
    professor_id: int
    professor_name: str
    institution: str
    department: str | None
    lab: str | None
    title: str | None
    recipient: str
    research_topics: str
    professor_evidence_ids: tuple[int, ...]
    email_evidence_ids: tuple[int, ...]
    affiliation_evidence_ids: tuple[int, ...]
    publication_ids: tuple[int, ...]
    publication_titles: tuple[str, ...]
    profile_version_id: int
    claim_ids: tuple[int, ...]
    claim_texts: tuple[str, ...]
    research_track_version_id: int
    research_problem: str
    application_id: int
    programme: str | None
    opportunity_id: int | None
    opportunity_state: str | None
    contact_policy: str
    application_route: str | None
    funding_context: str | None
    deadline_at: str | None
    supervision_state: str
    application_readiness: dict
    document_package_id: int
    requirement_ids: tuple[int, ...]
    requirement_states: tuple[tuple[int, str, str | None], ...]
    attachments: tuple[Attachment, ...]
    previous_contact_count: int


@dataclass(frozen=True)
class OutreachEmailDraft:
    subject: str
    body: str
    candidate_claim_ids: tuple[int, ...]
    professor_evidence_ids: tuple[int, ...]
    publication_ids: tuple[int, ...]
    research_track_version_id: int
    requirement_ids: tuple[int, ...]
    attachment_document_version_ids: tuple[int, ...]
    attachments_mentioned: tuple[str, ...]
    risk_flags: tuple[str, ...]
    generated_at: str
    provider: str = "deterministic"
    model: str = "reviewed-template"
    prompt_version: str = "faculty-email-v1"


@dataclass(frozen=True)
class CampaignPolicy:
    included_universities: tuple[str, ...] = ()
    included_application_ids: tuple[int, ...] = ()
    included_opportunity_ids: tuple[int, ...] = ()
    minimum_research_fit: float = 0.0
    minimum_evidence_coverage: float = 0.0
    timezone_name: str | None = None
    allowed_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    local_start_hour: int = 9
    local_end_hour: int = 17
    daily_cap: int = 3
    minimum_spacing_minutes: int = 90
    maximum_randomized_spacing_minutes: int = 180
    retry_policy: str = "MANUAL_REVIEW_ONLY"
    follow_up_delay_days: int = 14

    def validate(self) -> None:
        if not 0 <= self.minimum_research_fit <= 10 or not 0 <= self.minimum_evidence_coverage <= 1:
            raise ValueError("Campaign fit and coverage thresholds are invalid")
        if not 0 <= self.local_start_hour < self.local_end_hour <= 24 or not all(0 <= d <= 6 for d in self.allowed_weekdays):
            raise ValueError("Campaign local send window is invalid")
        if self.daily_cap < 1 or self.minimum_spacing_minutes < 0 or self.maximum_randomized_spacing_minutes < self.minimum_spacing_minutes:
            raise ValueError("Campaign cap or spacing is invalid")
        if self.retry_policy != "MANUAL_REVIEW_ONLY" or self.follow_up_delay_days < 1:
            raise ValueError("Automatic retry or invalid follow-up timing is not supported")
        if self.timezone_name:
            try:
                ZoneInfo(self.timezone_name)
            except ZoneInfoNotFoundError as error:
                raise ValueError("Unknown campaign timezone") from error


class OutreachBlocked(ValueError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("Outreach blocked: " + "; ".join(reasons))


class OutreachService:
    def __init__(self, db_path: Path | str, vault: DocumentVault | None = None):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.vault = vault or DocumentVault(self.db_path)
        self.packages = PackageBuilder(self.db_path, self.vault)
        self.materials = MaterialStudio(self.db_path, self.vault)

    @staticmethod
    def key(faculty_id: int, stage: str = "INITIAL", campaign_id: int | None = None) -> str:
        if not stage or not re.fullmatch(r"[A-Z_]+", stage):
            raise ValueError("Use an explicit uppercase outreach stage")
        return f"{campaign_id or 'oneoff'}:{faculty_id}:{stage}"

    @staticmethod
    def greeting(context: OutreachContext) -> str:
        name = context.professor_name.strip()
        if name.casefold().startswith(("dr ", "professor ", "prof ")):
            return f"Dear {name},"
        if context.title and "professor" in context.title.casefold():
            return f"Dear Professor {name.split()[-1]},"
        return f"Dear {name},"

    def _contact_reasons(self, faculty_id: int, recipient: str, outreach_key: str,
                         *, allow_package_id: int | None = None) -> list[str]:
        with connect(self.db_path) as db:
            restriction = db.execute("SELECT state FROM contact_restrictions WHERE faculty_profile_id=?", (faculty_id,)).fetchone()
            history_rows = db.execute("""SELECT faculty_profile_id,recipient FROM gmail_threads
                WHERE direction='OUTBOUND' AND match_state!='REJECTED'""").fetchall()
            history = sum(r["faculty_profile_id"] == faculty_id or _email(recipient) in
                          {address.casefold() for _,address in getaddresses([r["recipient"]])}
                          for r in history_rows)
            sent = db.execute("SELECT status FROM outreach_messages WHERE outreach_key=?", (outreach_key,)).fetchone()
            active = db.execute("""SELECT id FROM outreach_packages WHERE outreach_key=?
                AND status IN ('DRAFT','NEEDS_REVIEW','APPROVED','SCHEDULED','SENDING','SENT','AMBIGUOUS_SEND')
                ORDER BY id DESC LIMIT 1""", (outreach_key,)).fetchone()
        reasons = []
        if restriction and restriction["state"] != "CLEAR":
            reasons.append("CONTACT_RESTRICTION:" + restriction["state"])
        if history:
            reasons.append("PREVIOUS_GMAIL_CONTACT")
        if sent:
            reasons.append("EXISTING_OUTREACH_MESSAGE:" + sent["status"])
        if active and active["id"] != allow_package_id:
            reasons.append("DUPLICATE_ACTIVE_OUTREACH")
        return reasons

    def build_context(self, faculty_id: int, application_id: int, document_package_id: int,
                      profile_version_id: int, track_version_id: int, *,
                      campaign_id: int | None = None, stage: str = "INITIAL",
                      allow_package_id: int | None = None) -> OutreachContext:
        reasons = []
        try:
            _, track, approved = self.materials._approved_context(profile_version_id, track_version_id, "outreach")
        except ValueError as error:
            raise OutreachBlocked(["APPLICANT_REVIEW:" + str(error)]) from error
        facts = [c for c in approved.values() if c["classification"] == "FACT"]
        if not facts:
            reasons.append("NO_APPROVED_OUTREACH_FACT")
        with connect(self.db_path) as db:
            faculty = db.execute("SELECT * FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
            app = db.execute("SELECT * FROM applications WHERE id=?", (application_id,)).fetchone()
            doc_package = db.execute("SELECT * FROM application_packages WHERE id=?", (document_package_id,)).fetchone()
            if not faculty or not app or not doc_package:
                raise OutreachBlocked(["MISSING_FACULTY_APPLICATION_OR_DOCUMENT_PACKAGE"])
            opportunity = db.execute("SELECT * FROM opportunities WHERE id=?", (app["opportunity_id"],)).fetchone() if app["opportunity_id"] else None
            opportunity_evidence = db.execute("SELECT verification_state,retrieved_at FROM source_evidence WHERE id=?",
                (opportunity["source_evidence_id"],)).fetchone() if opportunity and opportunity["source_evidence_id"] else None
            programme = db.execute("SELECT * FROM programmes WHERE id=?", (app["programme_id"],)).fetchone() if app["programme_id"] else None
            linked = db.execute("SELECT 1 FROM application_faculty WHERE application_id=? AND faculty_profile_id=?",
                                (application_id, faculty_id)).fetchone()
            links = [dict(r) for r in db.execute("""SELECT l.fact_type,e.id,e.retrieved_at,e.verification_state
                FROM faculty_evidence_links l JOIN source_evidence e ON e.id=l.source_evidence_id
                WHERE l.faculty_profile_id=?""", (faculty_id,))]
            publications = [dict(r) for r in db.execute("""SELECT p.*,e.verification_state AS evidence_state,
                e.retrieved_at AS evidence_retrieved_at
                FROM publications p JOIN source_evidence e ON e.id=p.source_evidence_id
                WHERE p.faculty_profile_id=? ORDER BY p.year DESC,p.id DESC""", (faculty_id,))]
            pdocs = [dict(r) for r in db.execute("SELECT * FROM package_documents WHERE package_id=? ORDER BY sort_order", (document_package_id,))]
            duplicate = db.execute("""SELECT 1 FROM faculty_duplicate_reviews WHERE review_state='PENDING'
                AND (faculty_profile_a_id=? OR faculty_profile_b_id=?) LIMIT 1""", (faculty_id, faculty_id)).fetchone()
            owner = db.execute("""SELECT a.owner_name FROM applicant_profiles a JOIN profile_versions v
                ON v.profile_id=a.id WHERE v.id=?""", (profile_version_id,)).fetchone()
            deadline = db.execute("""SELECT due_at FROM deadlines WHERE application_id=?
                AND deadline_type='APPLICATION' ORDER BY due_at LIMIT 1""", (application_id,)).fetchone()
            assessment = db.execute("""SELECT application_readiness_components_json FROM match_assessments
                WHERE faculty_profile_id=? AND application_id=? ORDER BY id DESC LIMIT 1""",
                (faculty_id,application_id)).fetchone()
        if faculty["verification_state"] != "VERIFIED" or faculty["affiliation_state"] != "CURRENT" or not linked:
            reasons.append("FACULTY_IDENTITY_OR_AFFILIATION_UNVERIFIED")
        if faculty["email_state"] != "VERIFIED" or not faculty["email"] or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", faculty["email"]):
            reasons.append("RECIPIENT_UNVERIFIED")
        if duplicate:
            reasons.append("UNRESOLVED_FACULTY_DUPLICATE")
        def current(fact_type: str, kind: str) -> list[int]:
            return [r["id"] for r in links if r["fact_type"] == fact_type and r["verification_state"] == "VERIFIED"
                    and freshness(r["retrieved_at"], kind) == "CURRENT"]
        affiliation_ids = current("AFFILIATION", "FACULTY_AFFILIATION")
        email_ids = current("EMAIL", "FACULTY_EMAIL")
        research_ids = current("TOPICS", "PUBLICATION")
        if not affiliation_ids:
            reasons.append("STALE_AFFILIATION_EVIDENCE")
        if not email_ids:
            reasons.append("STALE_EMAIL_EVIDENCE")
        if not research_ids:
            reasons.append("STALE_RESEARCH_EVIDENCE")
        if not faculty["research_topics"]:
            reasons.append("RESEARCH_TOPICS_MISSING")
        if not opportunity or opportunity["verification_state"] != "VERIFIED" or contact_policy_state(opportunity["contact_policy"]) != "PASS":
            reasons.append("CONTACT_POLICY_NOT_EXPLICITLY_ALLOWED")
        if not opportunity_evidence or opportunity_evidence["verification_state"] != "VERIFIED" or freshness(opportunity_evidence["retrieved_at"], "CONTACT_POLICY") != "CURRENT":
            reasons.append("CONTACT_POLICY_EVIDENCE_STALE_OR_UNVERIFIED")
        if opportunity and opportunity["opportunity_type"] == "ADVERTISED_POSITION" and opportunity["opening_status"] != "OPEN":
            reasons.append("OPENING_NOT_VERIFIED_OPEN")
        if opportunity and opportunity["opportunity_type"] == "ADVERTISED_POSITION" and (
            not opportunity_evidence or freshness(opportunity_evidence["retrieved_at"], "OPPORTUNITY_OPENING") != "CURRENT"):
            reasons.append("OPENING_EVIDENCE_STALE")
        if doc_package["application_id"] != application_id or doc_package["context"] != "FACULTY_OUTREACH" or doc_package["status"] != "READY":
            reasons.append("OUTREACH_DOCUMENT_PACKAGE_NOT_READY")
        if doc_package["profile_version_id"] != profile_version_id or doc_package["research_track_version_id"] != track_version_id:
            reasons.append("DOCUMENT_PACKAGE_APPLICANT_CONTEXT_MISMATCH")
        if not pdocs or not any(self.vault.get_version(d["document_version_id"])["document_type"] == "CV" for d in pdocs):
            reasons.append("APPROVED_CV_MISSING")
        with connect(self.db_path) as db:
            for selected in pdocs:
                artifact = db.execute("SELECT faculty_profile_id,application_id FROM generated_artifacts WHERE document_version_id=?",
                                      (selected["document_version_id"],)).fetchone()
                if artifact and ((artifact["faculty_profile_id"] and artifact["faculty_profile_id"] != faculty_id) or
                                 (artifact["application_id"] and artifact["application_id"] != application_id)):
                    reasons.append("PROFESSOR_SPECIFIC_DOCUMENT_MISMATCH")
        requirements = [r for r in self.packages.ledger.requirement_rows(application_id) if r["context"] == "FACULTY_OUTREACH"]
        if any(r["requirement_state"] == "UNKNOWN" for r in requirements):
            reasons.append("UNKNOWN_OUTREACH_REQUIREMENT")
        if doc_package["status"] == "READY" and self.packages.preflight(document_package_id)["status"] == "BLOCK":
            reasons.append("DOCUMENT_PREFLIGHT_BLOCKED")
        outreach_key = self.key(faculty_id, stage, campaign_id)
        reasons += self._contact_reasons(faculty_id, faculty["email"] or "", outreach_key,
                                         allow_package_id=allow_package_id)
        if reasons:
            raise OutreachBlocked(sorted(set(reasons)))
        attachments = []
        for d in pdocs:
            version = self.vault.get_version(d["document_version_id"])
            attachments.append(Attachment(version["id"], d["requirement_id"], version["document_type"],
                                          d["canonical_filename"], d["sha256"], d["inclusion_reason"], d["sort_order"]))
        pubs = [p for p in publications if p["evidence_state"] == "VERIFIED"
                and freshness(p["evidence_retrieved_at"], "PUBLICATION") == "CURRENT"]
        return OutreachContext(owner["owner_name"],faculty_id, faculty["name"], faculty["institution"], faculty["department"],
            faculty["lab"], faculty["official_title"], faculty["email"], faculty["research_topics"] or "",
            tuple(sorted(research_ids)), tuple(sorted(email_ids)), tuple(sorted(affiliation_ids)),
            tuple(p["id"] for p in pubs[:3]), tuple(p["title"] for p in pubs[:3]), profile_version_id,
            tuple(c["id"] for c in facts), tuple(c["claim_text"] for c in facts), track_version_id,
            track["research_problem"], application_id, programme["programme_name"] if programme else None,
            opportunity["id"] if opportunity else None, opportunity["opening_status"] if opportunity else None,
            opportunity["contact_policy"], app["portal_url"] or opportunity["application_route"] or (programme["portal_url"] if programme else None),
            opportunity["funding_text"], deadline["due_at"] if deadline else None, faculty["supervision_state"],
            json.loads(assessment[0]) if assessment else {}, document_package_id,
            tuple(r["id"] for r in requirements),
            tuple((r["id"],r["requirement_state"],r["normalized_document_type"]) for r in requirements),
            tuple(attachments), 0)

    @staticmethod
    def draft_text(context: OutreachContext) -> OutreachEmailDraft:
        research = terms(context.research_topics + " " + " ".join(context.publication_titles) + " " + context.research_problem)
        best = max(range(len(context.claim_texts)),
                   key=lambda i: len(terms(context.claim_texts[i]) & research))
        observation = (f'Your listed work includes "{context.publication_titles[0]}". This connects to '
                       f'{context.research_topics}.' if context.publication_titles else
                       f'Your verified research profile describes work on {context.research_topics}.')
        attached_types = [a.document_type for a in context.attachments]
        labels = {"CV": "CV", "RESEARCH_PROPOSAL": "research proposal", "COVER_LETTER": "cover letter"}
        attached = [labels.get(t, t.lower().replace("_", " ")) for t in attached_types]
        mention = ", ".join(attached[:-1]) + (" and " if len(attached) > 1 else "") + attached[-1]
        subject = f"PhD research enquiry - {(context.research_topics.split(',')[0].strip() or 'research fit')[:70]}"
        body = (f"{OutreachService.greeting(context)}\n\n"
                f"I am preparing a PhD application at {context.institution} and am writing to ask about the best route "
                f"for a research conversation. {observation}\n\n"
                f"{context.claim_texts[best]} My proposed research direction asks: {context.research_problem} "
                f"I see a possible connection to your group's work and would value your view on whether this question "
                f"fits its current research. I am particularly interested in evaluations that reveal when these systems fail.\n\n"
                f"Would you be open to discussing supervision or advising which application route is appropriate? "
                f"I have attached my {mention} for context. I would be grateful for any guidance you can share.\n\n"
                f"Best regards,\n{context.applicant_name}")
        return OutreachEmailDraft(subject, body, (context.claim_ids[best],), context.professor_evidence_ids,
            context.publication_ids[:1], context.research_track_version_id, context.requirement_ids,
            tuple(a.document_version_id for a in context.attachments), tuple(attached), (), utc_now())

    def quality_gate(self, context: OutreachContext, draft: OutreachEmailDraft) -> dict:
        rules = []
        def add(rule_id, severity, message, affected=None, remediation=""):
            rules.append({"rule_id": rule_id, "status": severity, "message": message,
                          "affected": affected, "remediation": remediation})
        with connect(self.db_path) as db:
            faculty = db.execute("SELECT * FROM faculty_profiles WHERE id=?", (context.professor_id,)).fetchone()
            profile = db.execute("SELECT approval_state FROM profile_versions WHERE id=?", (context.profile_version_id,)).fetchone()
            track = db.execute("SELECT approval_state FROM research_track_versions WHERE id=?", (context.research_track_version_id,)).fetchone()
            all_names = [r[0] for r in db.execute("SELECT name FROM faculty_profiles WHERE id!=? AND verification_state='VERIFIED'", (context.professor_id,))]
            all_institutions = [r[0] for r in db.execute("SELECT DISTINCT institution FROM faculty_profiles WHERE id!=?", (context.professor_id,))]
            all_programmes = [r[0] for r in db.execute("SELECT DISTINCT programme_name FROM programmes WHERE programme_name IS NOT NULL")]
        identity = bool(faculty and faculty["name"] == context.professor_name and faculty["institution"] == context.institution
                        and faculty["email_state"] == "VERIFIED" and _email(faculty["email"]) == _email(context.recipient))
        add("IDENTITY", "PASS" if identity else "BLOCK", "Verified professor and recipient" if identity else "Professor or recipient changed",
            context.professor_id, "Rebuild context from verified faculty")
        add("APPLICANT_PROFILE", "PASS" if profile and track and profile[0] == track[0] == "APPROVED" else "BLOCK",
            "Applicant profile and track approved" if profile and track and profile[0] == track[0] == "APPROVED" else "Applicant approval missing",
            context.profile_version_id, "Review applicant claims and track")
        claim_ids = set(draft.candidate_claim_ids)
        if not claim_ids or not claim_ids <= set(context.claim_ids):
            add("CLAIM_IDS", "BLOCK", "Claim IDs outside approved outreach context", sorted(claim_ids), "Regenerate from approved claims")
        else:
            with connect(self.db_path) as db:
                claims = [dict(r) for r in db.execute(f"SELECT id,claim_text,classification,review_status,approved_for_outreach FROM claim_revisions WHERE id IN ({','.join('?' for _ in claim_ids)})", tuple(claim_ids))]
            valid = len(claims) == len(claim_ids) and all(c["classification"] == "FACT" and c["review_status"] == "APPROVED"
                                                       and c["approved_for_outreach"] for c in claims)
            add("CLAIM_IDS", "PASS" if valid else "BLOCK", "Approved factual claim references" if valid else "Unapproved, rejected, or aspirational claim reference",
                sorted(claim_ids), "Use an approved factual claim")
            for claim in claims:
                add("CLAIM_TEXT", "PASS" if claim["claim_text"] in draft.body else "BLOCK",
                    "Exact approved claim appears in email" if claim["claim_text"] in draft.body else "Applicant fact was edited beyond approved wording",
                    claim["id"], "Restore approved wording or review a new claim revision")
        evidence_ids = set(draft.professor_evidence_ids)
        if not evidence_ids or not evidence_ids <= set(context.professor_evidence_ids):
            add("PROFESSOR_EVIDENCE", "BLOCK", "Research evidence IDs are missing or outside context", sorted(evidence_ids), "Regenerate from verified evidence")
        else:
            with connect(self.db_path) as db:
                evidence = [dict(r) for r in db.execute(f"SELECT id,verification_state,retrieved_at FROM source_evidence WHERE id IN ({','.join('?' for _ in evidence_ids)})", tuple(evidence_ids))]
            good = len(evidence) == len(evidence_ids) and all(e["verification_state"] == "VERIFIED" and freshness(e["retrieved_at"], "PUBLICATION") == "CURRENT" for e in evidence)
            add("PROFESSOR_EVIDENCE", "PASS" if good else "BLOCK", "Current verified research evidence" if good else "Research evidence stale or unverified",
                sorted(evidence_ids), "Refresh official faculty research evidence")
        if not draft.publication_ids or set(draft.publication_ids) <= set(context.publication_ids):
            with connect(self.db_path) as db:
                papers = [dict(r) for r in db.execute(f"SELECT p.id,p.title,p.faculty_profile_id,p.year,e.verification_state,e.retrieved_at FROM publications p JOIN source_evidence e ON e.id=p.source_evidence_id WHERE p.id IN ({','.join('?' for _ in draft.publication_ids)})", draft.publication_ids)] if draft.publication_ids else []
            good = len(papers) == len(draft.publication_ids) and all(p["faculty_profile_id"] == context.professor_id and p["title"] in draft.body and p["year"] and p["verification_state"] == "VERIFIED" and freshness(p["retrieved_at"], "PUBLICATION") == "CURRENT" for p in papers)
            add("PUBLICATIONS", "PASS" if good else "BLOCK", "Paper references resolve to this professor" if good else "Unsupported or mismatched paper reference",
                list(draft.publication_ids), "Use stored verified publications")
        else:
            add("PUBLICATIONS", "BLOCK", "Paper ID outside context", list(draft.publication_ids), "Regenerate from verified publications")
        docs_match = tuple(draft.attachment_document_version_ids) == tuple(a.document_version_id for a in context.attachments)
        add("ATTACHMENT_MANIFEST", "PASS" if docs_match else "BLOCK", "Exact selected attachment versions" if docs_match else "Attachment versions differ from document package",
            list(draft.attachment_document_version_ids), "Rebuild from approved document package")
        requirements_match = tuple(draft.requirement_ids) == context.requirement_ids
        add("REQUIREMENT_IDS", "PASS" if requirements_match else "BLOCK",
            "Sourced outreach requirements preserved" if requirements_match else "Requirement IDs differ from the frozen context",
            list(draft.requirement_ids), "Rebuild from current reviewed requirements")
        for a in context.attachments:
            version = self.vault.get_version(a.document_version_id)
            good = bool(version and version["approval_state"] == "APPROVED" and version["sha256"] == a.sha256
                        and self.vault.storage.verify_hash(version["storage_key"], a.sha256))
            add("ATTACHMENT_HASH", "PASS" if good else "BLOCK", "Approved file hash verified" if good else "Attachment missing, changed, or unapproved",
                a.document_version_id, "Restore or approve the exact Vault version")
        mentions = tuple(m.casefold() for m in draft.attachments_mentioned)
        good_mentions = bool(mentions and all(m in draft.body.casefold() for m in mentions))
        expected_types = {a.document_type for a in context.attachments}
        if "research proposal" in draft.body.casefold() and "RESEARCH_PROPOSAL" not in expected_types:
            good_mentions = False
        if "cover letter" in draft.body.casefold() and "COVER_LETTER" not in expected_types:
            good_mentions = False
        if "transcript" in draft.body.casefold() and "TRANSCRIPT" not in expected_types:
            good_mentions = False
        add("ATTACHMENT_MENTIONS", "PASS" if good_mentions else "BLOCK", "Email mentions only selected attachments" if good_mentions else "Email attachment wording differs from manifest",
            list(draft.attachments_mentioned), "Match email wording to exact selected files")
        wrong_names = [name for name in all_names if len(name.split()) >= 2 and re.search(r"\b" + re.escape(name) + r"\b", draft.body, re.I)]
        greeting_ok = draft.body.lstrip().startswith(self.greeting(context))
        add("WRONG_PROFESSOR", "BLOCK" if wrong_names or not greeting_ok else "PASS", "Other professor or incorrect greeting" if wrong_names or not greeting_ok else "Professor name check passed",
            wrong_names, "Remove other professor references")
        non_claim_body = draft.body
        for approved_text in context.claim_texts:
            non_claim_body = non_claim_body.replace(approved_text, "")
        non_claim_lines = [p.strip() for p in non_claim_body.splitlines()]
        wrong_institutions = [name for name in all_institutions if name and name.casefold() != context.institution.casefold()
                              and any(re.search(r"\b" + re.escape(name) + r"\b", line, re.I) for line in non_claim_lines)]
        add("WRONG_INSTITUTION", "BLOCK" if wrong_institutions else "PASS", "Other institution named" if wrong_institutions else "Institution check passed",
            wrong_institutions, "Remove references to another application")
        wrong_programmes = [name for name in all_programmes if name and name != context.programme
                            and any(re.search(r"\b" + re.escape(name) + r"\b", line, re.I) for line in non_claim_lines)]
        add("WRONG_PROGRAMME", "BLOCK" if wrong_programmes else "PASS", "Other programme named" if wrong_programmes else "Programme check passed",
            wrong_programmes, "Remove another programme's wording")
        placeholders = bool(PLACEHOLDER.search(draft.subject + " " + draft.body))
        add("PLACEHOLDERS", "BLOCK" if placeholders else "PASS", "Placeholder text found" if placeholders else "No placeholders",
            None, "Complete or remove placeholders")
        paragraphs = [p.strip() for p in draft.body.split("\n\n") if p.strip()]
        duplicate = len(paragraphs) != len(set(paragraphs))
        add("DUPLICATE_PARAGRAPHS", "BLOCK" if duplicate else "PASS", "Duplicate paragraph" if duplicate else "No duplicate paragraphs",
            None, "Edit repeated content")
        unsupported_supervision = bool(re.search(r"\b(?:you are|your group is) (?:accepting|recruiting) (?:new )?(?:students|PhD candidates)\b", draft.body, re.I))
        unsupported_funding = bool(re.search(r"\b(?:funded position|funding is available|fully funded)\b", draft.body, re.I))
        add("SUPERVISION_CLAIM", "BLOCK" if unsupported_supervision else "PASS", "Unsupported supervision assertion" if unsupported_supervision else "Supervision phrased as a question",
            None, "Ask without asserting current openings")
        add("FUNDING_CLAIM", "BLOCK" if unsupported_funding else "PASS", "Unsupported funding assertion" if unsupported_funding else "No unsupported funding assertion",
            None, "Remove or cite verified funding evidence")
        add("TRACK_LINEAGE", "PASS" if draft.research_track_version_id == context.research_track_version_id
            and context.research_problem in draft.body else "BLOCK",
            "Approved research direction preserved" if draft.research_track_version_id == context.research_track_version_id
            and context.research_problem in draft.body else "Research direction changed or omitted",
            context.research_track_version_id, "Restore approved direction wording")
        words = len(draft.body.split())
        add("LENGTH", "WARNING" if words < 120 or words > 170 else "PASS", f"{words} words; target 120–170", None, "Edit for a concise academic message")
        subject_ok = bool(draft.subject.strip()) and len(draft.subject) <= 120 and "\n" not in draft.subject and "\r" not in draft.subject
        add("SUBJECT", "PASS" if subject_ok else "BLOCK", "Concise safe subject" if subject_ok else "Subject missing, too long, or multiline",
            None, "Use a single-line subject under 120 characters")
        contact = self._contact_reasons(context.professor_id, context.recipient,
                                        self.key(context.professor_id), allow_package_id=None)
        # Package/queue duplication is checked before generation and again before send.
        contact = [r for r in contact if not r.startswith("DUPLICATE_ACTIVE_OUTREACH")]
        add("CONTACT_MEMORY", "BLOCK" if contact else "PASS", ", ".join(contact) if contact else "No known contact restriction or previous message",
            context.professor_id, "Reconcile Sent history or review contact state")
        status = "BLOCK" if any(r["status"] == "BLOCK" for r in rules) else "WARNING" if any(r["status"] == "WARNING" for r in rules) else "PASS"
        return {"version": QUALITY_VERSION, "status": status, "word_count": words, "rules": rules}

    def prepare(self, faculty_id: int, application_id: int, document_package_id: int,
                profile_version_id: int, track_version_id: int, *, campaign_id: int | None = None,
                stage: str = "INITIAL") -> int:
        context = self.build_context(faculty_id, application_id, document_package_id,
                                     profile_version_id, track_version_id, campaign_id=campaign_id, stage=stage)
        draft = self.draft_text(context)
        return self._store(context, draft, campaign_id=campaign_id, stage=stage)

    def _store(self, context: OutreachContext, draft: OutreachEmailDraft, *,
               campaign_id: int | None, stage: str) -> int:
        gate = self.quality_gate(context, draft)
        snapshot = {"context": asdict(context), "draft": asdict(draft)}
        content = _dump(snapshot)
        digest = hashlib.sha256(content.encode()).hexdigest()
        outreach_key = self.key(context.professor_id, stage, campaign_id)
        with transaction(self.db_path) as db:
            version = db.execute("SELECT COALESCE(MAX(version_number),0)+1 FROM outreach_packages WHERE outreach_key=?", (outreach_key,)).fetchone()[0]
            status = "BLOCKED" if gate["status"] == "BLOCK" else "NEEDS_REVIEW"
            package_id = db.execute("""INSERT INTO outreach_packages
                (outreach_key,version_number,faculty_profile_id,application_id,campaign_id,document_package_id,
                 stage,status,snapshot_json,snapshot_sha256,quality_json,quality_version,policy_version,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (outreach_key,version,context.professor_id,
                context.application_id,campaign_id,context.document_package_id,stage,status,content,digest,
                _dump(gate),QUALITY_VERSION,POLICY_VERSION,utc_now())).lastrowid
            for a in context.attachments:
                db.execute("""INSERT INTO outreach_package_documents
                    (package_id,document_version_id,requirement_id,sha256,filename,sort_order,inclusion_reason)
                    VALUES(?,?,?,?,?,?,?)""", (package_id,a.document_version_id,a.requirement_id,a.sha256,
                    a.filename,a.order,a.reason))
            self._event(db, package_id, None, "GENERATED", None, status, "SYSTEM")
        return package_id

    @staticmethod
    def _event(db, package_id, message_id, event, previous, current, actor,
               gmail_message_id=None, gmail_thread_id=None):
        db.execute("""INSERT INTO outreach_events
            (package_id,message_id,event,previous_status,new_status,policy_version,quality_version,
             actor,gmail_message_id,gmail_thread_id,event_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (package_id,message_id,event,previous,current,POLICY_VERSION,QUALITY_VERSION,actor,
             gmail_message_id,gmail_thread_id,utc_now()))

    def _transition(self, package_id: int, new_status: str, actor: str, *, message_id=None) -> None:
        with transaction(self.db_path) as db:
            row = db.execute("SELECT status FROM outreach_packages WHERE id=?", (package_id,)).fetchone()
            if not row or new_status not in TRANSITIONS[row["status"]]:
                raise ValueError("Invalid outreach state transition")
            db.execute("UPDATE outreach_packages SET status=? WHERE id=?", (new_status,package_id))
            self._event(db, package_id, message_id, "STATUS", row["status"], new_status, actor)

    def get_package(self, package_id: int) -> dict:
        with connect(self.db_path) as db:
            row = db.execute("SELECT * FROM outreach_packages WHERE id=?", (package_id,)).fetchone()
        if not row:
            raise ValueError("Outreach package not found")
        return dict(row)

    @staticmethod
    def _unpack(package: dict) -> tuple[OutreachContext, OutreachEmailDraft]:
        snapshot = json.loads(package["snapshot_json"])
        data = snapshot["context"]
        data["attachments"] = tuple(Attachment(**a) for a in data["attachments"])
        data["requirement_states"] = tuple(tuple(r) for r in data["requirement_states"])
        for field in ("professor_evidence_ids","email_evidence_ids","affiliation_evidence_ids",
                      "publication_ids","publication_titles","claim_ids","claim_texts","requirement_ids"):
            data[field] = tuple(data[field])
        context = OutreachContext(**data)
        draft_data = snapshot["draft"]
        for field in ("candidate_claim_ids","professor_evidence_ids","publication_ids","requirement_ids",
                      "attachment_document_version_ids","attachments_mentioned","risk_flags"):
            draft_data[field] = tuple(draft_data[field])
        return context, OutreachEmailDraft(**draft_data)

    def revise(self, package_id: int, subject: str, body: str, actor: str) -> int:
        if not actor.strip() or not subject.strip() or not body.strip():
            raise ValueError("Reviewer, subject, and body are required")
        old = self.get_package(package_id)
        if old["status"] in {"SENDING", "SENT", "AMBIGUOUS_SEND", "REPLIED", "BOUNCED"}:
            raise ValueError("Sent or uncertain packages cannot be revised")
        context, prior = self._unpack(old)
        current = self.build_context(context.professor_id, context.application_id,
            context.document_package_id, context.profile_version_id, context.research_track_version_id,
            campaign_id=old["campaign_id"], stage=old["stage"], allow_package_id=package_id)
        draft = OutreachEmailDraft(subject.strip(), body.strip(), prior.candidate_claim_ids,
            prior.professor_evidence_ids, prior.publication_ids, prior.research_track_version_id,
            prior.requirement_ids, prior.attachment_document_version_ids,
            prior.attachments_mentioned, prior.risk_flags, utc_now(), "manual", "operator-edit",
            "faculty-email-edit-v1")
        new_id = self._store(current, draft, campaign_id=old["campaign_id"], stage=old["stage"])
        with transaction(self.db_path) as db:
            if old["status"] in {"DRAFT", "NEEDS_REVIEW", "BLOCKED", "SCHEDULED"}:
                db.execute("UPDATE outreach_packages SET status='CANCELLED' WHERE id=?", (package_id,))
                self._event(db, package_id, None, "SUPERSEDED", old["status"], "CANCELLED", actor)
            else:
                db.execute("UPDATE outreach_packages SET stale_at=? WHERE id=?", (utc_now(),package_id))
                self._event(db, package_id, None, "SUPERSEDED", old["status"], old["status"], actor)
        return new_id

    def approve(self, package_id: int, reviewer: str) -> dict:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        package = self.get_package(package_id)
        if package["status"] != "NEEDS_REVIEW" or package["stale_at"]:
            raise ValueError("Only a current reviewed draft can be approved")
        context, draft = self._unpack(package)
        self.build_context(context.professor_id, context.application_id, context.document_package_id,
            context.profile_version_id, context.research_track_version_id,
            campaign_id=package["campaign_id"], stage=package["stage"], allow_package_id=package_id)
        if hashlib.sha256(package["snapshot_json"].encode()).hexdigest() != package["snapshot_sha256"]:
            raise ValueError("Outreach snapshot hash changed")
        gate = self.quality_gate(context, draft)
        if gate["status"] == "BLOCK":
            raise OutreachBlocked([r["rule_id"] for r in gate["rules"] if r["status"] == "BLOCK"])
        with transaction(self.db_path) as db:
            db.execute("UPDATE outreach_packages SET status='APPROVED',approved_at=?,approved_by=? WHERE id=? AND status='NEEDS_REVIEW'",
                       (utc_now(),reviewer.strip(),package_id))
            self._event(db, package_id, None, "APPROVED", "NEEDS_REVIEW", "APPROVED", reviewer.strip())
        return gate

    def reject(self, package_id: int, reviewer: str) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        package = self.get_package(package_id)
        if package["status"] not in {"DRAFT", "NEEDS_REVIEW", "BLOCKED", "APPROVED", "SCHEDULED"}:
            raise ValueError("Package cannot be cancelled now")
        self._transition(package_id, "CANCELLED", reviewer.strip())

    def schedule(self, package_id: int, when_utc: str, reviewer: str) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        datetime.fromisoformat(when_utc.replace("Z", "+00:00"))
        package = self.get_package(package_id)
        if package["status"] != "APPROVED":
            raise ValueError("Approve the package before scheduling")
        self._transition(package_id, "SCHEDULED", reviewer.strip())
        with transaction(self.db_path) as db:
            db.execute("UPDATE outreach_packages SET scheduled_at=? WHERE id=?", (when_utc,package_id))

    def refresh_staleness(self, package_id: int) -> list[str]:
        package = self.get_package(package_id)
        if package["status"] not in {"APPROVED", "SCHEDULED"}:
            return []
        context, draft = self._unpack(package)
        reasons = []
        with connect(self.db_path) as db:
            profile_row = db.execute("SELECT profile_id FROM profile_versions WHERE id=?", (context.profile_version_id,)).fetchone()
            track_row = db.execute("SELECT track_id FROM research_track_versions WHERE id=?", (context.research_track_version_id,)).fetchone()
            latest_profile = db.execute("SELECT MAX(id) FROM profile_versions WHERE profile_id=? AND approval_state='APPROVED'",
                                        (profile_row[0],)).fetchone()[0] if profile_row else None
            latest_track = db.execute("SELECT MAX(id) FROM research_track_versions WHERE track_id=? AND approval_state='APPROVED'",
                                      (track_row[0],)).fetchone()[0] if track_row else None
            current_evidence = {r[0] for r in db.execute("""SELECT source_evidence_id FROM faculty_evidence_links
                WHERE faculty_profile_id=? AND fact_type IN ('TOPICS','AFFILIATION','EMAIL')""", (context.professor_id,))}
        if latest_profile and latest_profile != context.profile_version_id:
            reasons.append("NEW_APPROVED_PROFILE_VERSION")
        if latest_track and latest_track != context.research_track_version_id:
            reasons.append("NEW_APPROVED_RESEARCH_DIRECTION")
        if current_evidence - set(context.professor_evidence_ids) - set(context.email_evidence_ids) - set(context.affiliation_evidence_ids):
            reasons.append("FACULTY_EVIDENCE_CHANGED")
        try:
            self.build_context(context.professor_id, context.application_id, context.document_package_id,
                context.profile_version_id, context.research_track_version_id,
                campaign_id=package["campaign_id"], stage=package["stage"], allow_package_id=package_id)
        except OutreachBlocked as error:
            reasons.extend(error.reasons)
        gate = self.quality_gate(context, draft)
        reasons += [r["rule_id"] for r in gate["rules"] if r["status"] == "BLOCK"]
        if reasons and not package["stale_at"]:
            with transaction(self.db_path) as db:
                db.execute("UPDATE outreach_packages SET stale_at=? WHERE id=?", (utc_now(),package_id))
                self._event(db, package_id, None, "STALE", package["status"], package["status"], "SYSTEM")
        return sorted(set(reasons))

    def reconcile_sent(self, records: list[dict], *, source: str = "GMAIL") -> dict:
        """Store metadata only. Exact-email matches stay pending until confirmed."""
        if source not in {"GMAIL", "TEST"}:
            raise ValueError("Invalid reconciliation source")
        matched = 0
        with transaction(self.db_path) as db:
            for item in records:
                addresses = [address.casefold() for _,address in getaddresses([item.get("recipient") or ""]) if address]
                recipient = addresses[0] if len(addresses) == 1 else (item.get("recipient") or "").strip().casefold()
                message_id = (item.get("message_id") or "").strip()
                thread_id = (item.get("thread_id") or "").strip()
                if not addresses or not recipient or not message_id or not thread_id or not item.get("message_at"):
                    raise ValueError("Sent record lacks recipient, identifiers, or timestamp")
                candidates = [r[0] for r in db.execute("SELECT id FROM faculty_profiles WHERE lower(email)=?", (recipient,))] if len(addresses) == 1 else []
                professor_id = candidates[0] if len(candidates) == 1 else None
                if professor_id:
                    matched += 1
                package = db.execute("SELECT id,application_id FROM outreach_packages WHERE snapshot_sha256=?",
                                     (item.get("package_identity") or "",)).fetchone() if item.get("package_identity") else None
                db.execute("""INSERT OR IGNORE INTO gmail_threads
                    (faculty_profile_id,recipient,direction,subject,gmail_message_id,gmail_thread_id,message_at,
                     outreach_package_id,application_id,contact_stage,match_state,match_confidence,source,created_at)
                    VALUES(?,?,'OUTBOUND',?,?,?,?,?,?,'INITIAL',?,?,?,?)""",
                    (professor_id,recipient,item.get("subject", ""),message_id,thread_id,item["message_at"],
                     package["id"] if package else None,package["application_id"] if package else None,
                     "PENDING", "EXACT_EMAIL" if professor_id else "UNKNOWN",source,utc_now()))
            db.execute("INSERT INTO gmail_reconciliation_runs(status,scanned_count,matched_count,run_at,source) VALUES('COMPLETE',?,?,?,?)",
                       (len(records),matched,utc_now(),source))
        return {"scanned": len(records), "exact_email_candidates": matched}

    def confirm_sent_link(self, gmail_message_id: str, faculty_id: int | None,
                          reviewer: str, *, accept: bool) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        with transaction(self.db_path) as db:
            row = db.execute("SELECT * FROM gmail_threads WHERE gmail_message_id=?", (gmail_message_id,)).fetchone()
            if not row or row["direction"] != "OUTBOUND":
                raise ValueError("Sent history item not found")
            if accept:
                faculty = db.execute("SELECT email FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
                if not faculty or _email(faculty["email"]) not in {a.casefold() for _,a in getaddresses([row["recipient"]])}:
                    raise ValueError("Professor email does not match Sent recipient")
                db.execute("UPDATE gmail_threads SET faculty_profile_id=?,match_state='CONFIRMED',match_confidence='OPERATOR_CONFIRMED' WHERE id=?",
                           (faculty_id,row["id"]))
            else:
                db.execute("UPDATE gmail_threads SET faculty_profile_id=NULL,match_state='REJECTED',match_confidence='OPERATOR_REJECTED' WHERE id=?",
                           (row["id"],))

    def record_manual_contact(self, faculty_id: int, recipient: str, subject: str,
                              sent_at: str, reviewer: str) -> str:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        with connect(self.db_path) as db:
            faculty = db.execute("SELECT email FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
        if not faculty or _email(faculty["email"]) != _email(recipient):
            raise ValueError("Manual contact recipient must match faculty email")
        message_id = "manual-" + uuid.uuid4().hex
        with transaction(self.db_path) as db:
            db.execute("""INSERT INTO gmail_threads
                (faculty_profile_id,recipient,direction,subject,gmail_message_id,gmail_thread_id,message_at,
                 contact_stage,match_state,match_confidence,source,created_at)
                VALUES(?,?,'OUTBOUND',?,?,?,?,'INITIAL','CONFIRMED','OPERATOR_CONFIRMED','MANUAL',?)""",
                (faculty_id,_email(recipient),subject,message_id,message_id,sent_at,utc_now()))
        return message_id

    def set_contact_restriction(self, faculty_id: int, state: str, reason: str, reviewer: str) -> None:
        if state not in {"CLEAR", "DO_NOT_CONTACT", "REJECTED"} or not reason.strip() or not reviewer.strip():
            raise ValueError("A reviewed contact state and reason are required")
        with transaction(self.db_path) as db:
            previous = db.execute("SELECT state FROM contact_restrictions WHERE faculty_profile_id=?", (faculty_id,)).fetchone()
            db.execute("""INSERT INTO contact_restrictions(faculty_profile_id,state,reason,reviewed_by,updated_at)
                VALUES(?,?,?,?,?) ON CONFLICT(faculty_profile_id) DO UPDATE SET
                state=excluded.state,reason=excluded.reason,reviewed_by=excluded.reviewed_by,updated_at=excluded.updated_at""",
                (faculty_id,state,reason.strip(),reviewer.strip(),utc_now()))
            self._event(db, None, None, "CONTACT_RESTRICTION:" + str(faculty_id),
                        previous[0] if previous else None, state, reviewer.strip())

    def send(self, package_id: int, transport, actor: str, *, acknowledge_emergency: bool = False,
             now: datetime | None = None) -> dict:
        """One reviewed manual attempt. Any uncertain network outcome is not retried."""
        if not actor.strip():
            raise ValueError("Sending operator required")
        package = self.get_package(package_id)
        if package["status"] not in {"APPROVED", "SCHEDULED"} or package["stale_at"]:
            raise OutreachBlocked(["PACKAGE_NOT_CURRENT_AND_APPROVED"])
        context, draft = self._unpack(package)
        with connect(self.db_path) as db:
            reconciled = db.execute("SELECT status FROM gmail_reconciliation_runs ORDER BY id DESC LIMIT 1").fetchone()
        if not reconciled or reconciled[0] != "COMPLETE":
            raise OutreachBlocked(["GMAIL_SENT_HISTORY_NOT_RECONCILED"])
        if hasattr(transport, "credentials_ready") and not transport.credentials_ready():
            raise OutreachBlocked(["ROTATED_GMAIL_CREDENTIALS_REQUIRED"])
        if isinstance(transport, GmailGateway):
            try:
                transport.service
                # A previous import can be stale if the mailbox was used elsewhere.
                # Reconcile all Sent metadata before the last contact-memory check.
                self.reconcile_sent(transport.list_sent(), source="GMAIL")
            except Exception as error:
                raise OutreachBlocked(["GMAIL_CONNECTION_NOT_READY"]) from error
        self.build_context(context.professor_id, context.application_id, context.document_package_id,
            context.profile_version_id, context.research_track_version_id,
            campaign_id=package["campaign_id"], stage=package["stage"], allow_package_id=package_id)
        gate = self.quality_gate(context, draft)
        if gate["status"] == "BLOCK":
            raise OutreachBlocked([r["rule_id"] for r in gate["rules"] if r["status"] == "BLOCK"])
        if hashlib.sha256(package["snapshot_json"].encode()).hexdigest() != package["snapshot_sha256"]:
            raise OutreachBlocked(["SNAPSHOT_HASH_MISMATCH"])
        if package["campaign_id"]:
            reasons = self.campaign_send_reasons(package["campaign_id"], now=now)
            if reasons:
                if reasons == ["EMERGENCY_STOP"] and acknowledge_emergency:
                    pass  # Explicit manual acknowledgement only; no automatic worker exists.
                else:
                    raise OutreachBlocked(reasons)
        attachments = []
        for a in context.attachments:
            version = self.vault.get_version(a.document_version_id)
            if not version or version["sha256"] != a.sha256 or not self.vault.storage.verify_hash(version["storage_key"], a.sha256):
                raise OutreachBlocked(["ATTACHMENT_HASH_MISMATCH:" + str(a.document_version_id)])
            attachments.append((a.filename,self.vault.storage.get(version["storage_key"])))
        with transaction(self.db_path) as db:
            if db.execute("SELECT 1 FROM outreach_messages WHERE outreach_key=?", (package["outreach_key"],)).fetchone():
                raise OutreachBlocked(["IDEMPOTENCY_KEY_ALREADY_USED"])
            now_text = utc_now()
            message_id = db.execute("""INSERT INTO outreach_messages
                (outreach_key,package_id,recipient,subject,status,created_at,updated_at)
                VALUES(?,?,?,?,'SENDING',?,?)""",
                (package["outreach_key"],package_id,context.recipient,draft.subject,now_text,now_text)).lastrowid
            attempt_id = db.execute("""INSERT INTO outreach_attempts
                (message_id,package_id,outcome,started_at) VALUES(?,?,'IN_FLIGHT',?)""",
                (message_id,package_id,now_text)).lastrowid
            db.execute("UPDATE outreach_packages SET status='SENDING' WHERE id=?", (package_id,))
            self._event(db, package_id, message_id, "SEND_STARTED", package["status"], "SENDING", actor)
        try:
            response = transport.send(context.recipient,draft.subject,draft.body,attachments,
                                      package["snapshot_sha256"])
            if not response.get("message_id") or not response.get("thread_id"):
                raise RuntimeError("Gmail send result missing identifiers")
        except Exception as error:
            # A transport exception may have happened after Gmail accepted the message.
            with transaction(self.db_path) as db:
                db.execute("UPDATE outreach_messages SET status='AMBIGUOUS_SEND',last_error_code=?,updated_at=? WHERE id=?",
                           (type(error).__name__,utc_now(),message_id))
                db.execute("UPDATE outreach_attempts SET outcome='AMBIGUOUS_SEND',completed_at=?,error_code=? WHERE id=?",
                           (utc_now(),type(error).__name__,attempt_id))
                db.execute("UPDATE outreach_packages SET status='AMBIGUOUS_SEND' WHERE id=?", (package_id,))
                self._event(db, package_id, message_id, "SEND_AMBIGUOUS", "SENDING", "AMBIGUOUS_SEND", actor)
            return {"status": "AMBIGUOUS_SEND", "message_id": message_id, "retry_allowed": False}
        with transaction(self.db_path) as db:
            db.execute("""UPDATE outreach_messages SET status='SENT',gmail_message_id=?,gmail_thread_id=?,
                sent_at=?,updated_at=? WHERE id=?""", (response["message_id"],response["thread_id"],utc_now(),utc_now(),message_id))
            db.execute("""UPDATE outreach_attempts SET outcome='SENT',completed_at=?,gmail_message_id=?,
                gmail_thread_id=? WHERE id=?""", (utc_now(),response["message_id"],response["thread_id"],attempt_id))
            db.execute("UPDATE outreach_packages SET status='SENT' WHERE id=?", (package_id,))
            db.execute("""INSERT OR IGNORE INTO gmail_threads
                (faculty_profile_id,recipient,direction,subject,gmail_message_id,gmail_thread_id,message_at,
                 outreach_package_id,application_id,contact_stage,match_state,match_confidence,source,created_at)
                VALUES(?,?,'OUTBOUND',?,?,?,?,?,?,'INITIAL','CONFIRMED','LOCAL_SEND','GMAIL',?)""",
                (context.professor_id,_email(context.recipient),draft.subject,response["message_id"],
                 response["thread_id"],utc_now(),package_id,context.application_id,utc_now()))
            self._event(db, package_id, message_id, "SENT", "SENDING", "SENT", actor,
                        response["message_id"],response["thread_id"])
        return {"status": "SENT", "message_id": message_id,
                "gmail_message_id": response["message_id"], "gmail_thread_id": response["thread_id"]}

    def recover_ambiguous(self, message_id: int, actor: str) -> dict:
        with connect(self.db_path) as db:
            message = db.execute("SELECT * FROM outreach_messages WHERE id=?", (message_id,)).fetchone()
            if not message or message["status"] not in {"AMBIGUOUS_SEND", "SENDING"}:
                raise ValueError("Message is not awaiting ambiguous-send reconciliation")
            rows = [dict(r) for r in db.execute("""SELECT * FROM gmail_threads WHERE direction='OUTBOUND'
                AND match_state!='REJECTED' AND subject=? AND message_at>=?""",
                (message["subject"],message["created_at"]))]
            matches = [r for r in rows if _email(message["recipient"]) in
                       {a.casefold() for _,a in getaddresses([r["recipient"]])}]
        if len(matches) != 1:
            if message["status"] == "SENDING":
                with transaction(self.db_path) as db:
                    db.execute("UPDATE outreach_messages SET status='AMBIGUOUS_SEND',updated_at=? WHERE id=?", (utc_now(),message_id))
                    db.execute("UPDATE outreach_packages SET status='AMBIGUOUS_SEND' WHERE id=?", (message["package_id"],))
                    self._event(db, message["package_id"], message_id, "INTERRUPTED_SEND",
                                "SENDING", "AMBIGUOUS_SEND", actor)
            return {"status": "AMBIGUOUS_SEND", "matches": len(matches), "retry_allowed": False}
        found = matches[0]
        with transaction(self.db_path) as db:
            db.execute("""UPDATE outreach_messages SET status='SENT',gmail_message_id=?,gmail_thread_id=?,
                sent_at=?,updated_at=? WHERE id=?""", (found["gmail_message_id"],found["gmail_thread_id"],
                found["message_at"],utc_now(),message_id))
            db.execute("UPDATE outreach_packages SET status='SENT' WHERE id=?", (message["package_id"],))
            self._event(db, message["package_id"], message_id, "AMBIGUOUS_RECOVERED", message["status"], "SENT",
                        actor,found["gmail_message_id"],found["gmail_thread_id"])
        return {"status": "SENT", "gmail_message_id": found["gmail_message_id"], "retry_allowed": False}

    def record_reply(self, item: dict, reviewer: str = "GMAIL_IMPORT") -> int:
        message_id = (item.get("message_id") or "").strip()
        thread_id = (item.get("thread_id") or "").strip()
        if not message_id or not thread_id or not item.get("message_at"):
            raise ValueError("Reply metadata is incomplete")
        subject = item.get("subject", "")
        normalized = re.sub(r"^(?:(?:re|fw|fwd):\s*)+", "", subject, flags=re.I).strip()
        with transaction(self.db_path) as db:
            prior = db.execute("""SELECT * FROM gmail_threads WHERE gmail_thread_id=?
                AND direction='OUTBOUND' ORDER BY id DESC LIMIT 1""", (thread_id,)).fetchone()
            professor_id = prior["faculty_profile_id"] if prior else None
            package_id = prior["outreach_package_id"] if prior else None
            app_id = prior["application_id"] if prior else None
            db.execute("""INSERT OR IGNORE INTO reply_events
                (gmail_message_id,gmail_thread_id,faculty_profile_id,direction,subject_raw,subject_normalized,
                 message_at,detected_state,outreach_package_id,application_id,created_at)
                VALUES(?,?,?,'INBOUND',?,?,?,'NEW',?,?,?)""",
                (message_id,thread_id,professor_id,subject,normalized,item["message_at"],package_id,app_id,utc_now()))
            if professor_id:
                db.execute("""INSERT OR IGNORE INTO gmail_threads
                    (faculty_profile_id,recipient,direction,subject,gmail_message_id,gmail_thread_id,message_at,
                     outreach_package_id,application_id,contact_stage,match_state,match_confidence,source,created_at)
                    VALUES(?,?,'INBOUND',?,?,?,?,?,?,'REPLY','CONFIRMED','THREAD_LINK','GMAIL',?)""",
                    (professor_id,prior["recipient"],subject,message_id,thread_id,item["message_at"],package_id,app_id,utc_now()))
            if package_id:
                current = db.execute("SELECT status FROM outreach_packages WHERE id=?", (package_id,)).fetchone()[0]
                if current == "SENT":
                    db.execute("UPDATE outreach_packages SET status='REPLIED' WHERE id=?", (package_id,))
                    self._event(db, package_id, None, "REPLY_DETECTED", "SENT", "REPLIED", reviewer)
            return db.execute("SELECT id FROM reply_events WHERE gmail_message_id=?", (message_id,)).fetchone()[0]

    def create_campaign(self, name: str, cycle: str, policy: CampaignPolicy) -> int:
        policy.validate()
        if not name.strip() or not cycle.strip():
            raise ValueError("Campaign name and cycle required")
        with transaction(self.db_path) as db:
            return db.execute("""INSERT INTO campaigns
                (name,cycle,policy_json,policy_version,auto_send_enabled,emergency_stop,created_at,updated_at)
                VALUES(?,?,?,?,0,0,?,?)""",
                (name.strip(),cycle.strip(),_dump(asdict(policy)),POLICY_VERSION,utc_now(),utc_now())).lastrowid

    def set_campaign_controls(self, campaign_id: int, actor: str, *, status: str | None = None,
                              emergency_stop: bool | None = None,
                              auto_send_enabled: bool | None = None) -> None:
        if not actor.strip() or status not in {None,"ACTIVE","PAUSED","STOPPED"}:
            raise ValueError("Invalid campaign control request")
        changes = {}
        if status is not None:
            changes["status"] = status
        if emergency_stop is not None:
            changes["emergency_stop"] = int(emergency_stop)
        if auto_send_enabled is not None:
            changes["auto_send_enabled"] = int(auto_send_enabled)
        if not changes:
            return
        with transaction(self.db_path) as db:
            if not db.execute("SELECT 1 FROM campaigns WHERE id=?", (campaign_id,)).fetchone():
                raise ValueError("Campaign not found")
            db.execute("UPDATE campaigns SET " + ",".join(f"{k}=?" for k in changes) + ",updated_at=? WHERE id=?",
                       (*changes.values(),utc_now(),campaign_id))

    def campaign_send_reasons(self, campaign_id: int, *, now: datetime | None = None) -> list[str]:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("Send time must include a timezone")
        with connect(self.db_path) as db:
            campaign = db.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if not campaign:
                return ["CAMPAIGN_MISSING"]
            messages = [r[0] for r in db.execute("""SELECT m.sent_at FROM outreach_messages m
                JOIN outreach_packages p ON p.id=m.package_id WHERE p.campaign_id=? AND m.status='SENT'
                AND m.sent_at IS NOT NULL""", (campaign_id,))]
        policy = CampaignPolicy(**json.loads(campaign["policy_json"]))
        reasons = []
        if campaign["emergency_stop"]:
            reasons.append("EMERGENCY_STOP")
        if campaign["status"] != "ACTIVE":
            reasons.append("CAMPAIGN_" + campaign["status"])
        if not policy.timezone_name:
            reasons.append("PROFESSOR_TIMEZONE_UNKNOWN")
            return reasons
        local = now.astimezone(ZoneInfo(policy.timezone_name))
        if local.weekday() not in policy.allowed_weekdays or not policy.local_start_hour <= local.hour < policy.local_end_hour:
            reasons.append("OUTSIDE_SEND_WINDOW")
        same_day = 0
        latest = None
        for timestamp in messages:
            sent = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if sent.astimezone(ZoneInfo(policy.timezone_name)).date() == local.date():
                same_day += 1
            latest = max(latest, sent) if latest else sent
        if same_day >= policy.daily_cap:
            reasons.append("DAILY_CAP_REACHED")
        if latest and (now - latest).total_seconds() < policy.minimum_spacing_minutes * 60:
            reasons.append("MINIMUM_SPACING")
        return reasons

    def _candidate_policy_reasons(self, faculty_id: int, application_id: int,
                                  campaign: dict, policy: CampaignPolicy) -> list[str]:
        reasons = []
        with connect(self.db_path) as db:
            faculty = db.execute("SELECT institution FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
            app = db.execute("SELECT opportunity_id FROM applications WHERE id=?", (application_id,)).fetchone()
            assessment = db.execute("""SELECT * FROM match_assessments WHERE faculty_profile_id=? AND application_id=?
                ORDER BY id DESC LIMIT 1""", (faculty_id,application_id)).fetchone()
        if policy.included_universities and faculty["institution"] not in policy.included_universities:
            reasons.append("UNIVERSITY_OUTSIDE_CAMPAIGN")
        if policy.included_application_ids and application_id not in policy.included_application_ids:
            reasons.append("APPLICATION_OUTSIDE_CAMPAIGN")
        if policy.included_opportunity_ids and app["opportunity_id"] not in policy.included_opportunity_ids:
            reasons.append("OPPORTUNITY_OUTSIDE_CAMPAIGN")
        if policy.minimum_research_fit or policy.minimum_evidence_coverage:
            if not assessment:
                reasons.append("MATCH_ASSESSMENT_MISSING")
            else:
                components = json.loads(assessment["research_fit_components_json"])
                coverage = sum(MATCH_WEIGHTS[k] for k,v in components.items() if v["score"] is not None)
                if assessment["research_fit"] is None or assessment["research_fit"] < policy.minimum_research_fit:
                    reasons.append("RESEARCH_FIT_BELOW_THRESHOLD")
                if coverage < policy.minimum_evidence_coverage:
                    reasons.append("EVIDENCE_COVERAGE_BELOW_THRESHOLD")
        return reasons

    def dry_run(self, campaign_id: int, profile_version_id: int, track_version_id: int,
                *, now: datetime | None = None) -> dict:
        with connect(self.db_path) as db:
            campaign = db.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            candidates = [dict(r) for r in db.execute("""SELECT af.faculty_profile_id,af.application_id
                FROM application_faculty af ORDER BY af.application_id,af.faculty_profile_id""")]
        if not campaign:
            raise ValueError("Campaign not found")
        policy = CampaignPolicy(**json.loads(campaign["policy_json"]))
        outcomes = []
        for candidate in candidates:
            faculty_id, app_id = candidate["faculty_profile_id"], candidate["application_id"]
            reasons = self._candidate_policy_reasons(faculty_id,app_id,dict(campaign),policy)
            with connect(self.db_path) as db:
                doc_package = db.execute("""SELECT id FROM application_packages WHERE application_id=?
                    AND context='FACULTY_OUTREACH' AND status='READY' ORDER BY id DESC LIMIT 1""", (app_id,)).fetchone()
                outreach = db.execute("""SELECT * FROM outreach_packages WHERE outreach_key=?
                    ORDER BY version_number DESC LIMIT 1""", (self.key(faculty_id,campaign_id=campaign_id),)).fetchone()
            if not doc_package:
                reasons.append("READY_OUTREACH_DOCUMENT_PACKAGE_MISSING")
            elif not reasons:
                try:
                    self.build_context(faculty_id,app_id,doc_package["id"],profile_version_id,track_version_id,
                        campaign_id=campaign_id,allow_package_id=outreach["id"] if outreach else None)
                except OutreachBlocked as error:
                    reasons += error.reasons
            if outreach and outreach["status"] in {"SENT", "SENDING", "AMBIGUOUS_SEND", "REPLIED"}:
                reasons.append("PRIOR_OR_UNCERTAIN_SEND")
            if outreach and outreach["status"] == "BLOCKED":
                reasons.append("OUTREACH_QUALITY_BLOCKED")
            if outreach and outreach["stale_at"]:
                reasons.append("APPROVED_PACKAGE_STALE")
            if outreach and outreach["status"] in {"APPROVED", "SCHEDULED"} and not reasons:
                reasons += self.refresh_staleness(outreach["id"])
            if outreach and outreach["status"] in {"APPROVED", "SCHEDULED"} and not reasons:
                reasons += self.campaign_send_reasons(campaign_id, now=now)
            outcome = "BLOCKED" if reasons else "WOULD_SEND" if outreach and outreach["status"] in {"APPROVED", "SCHEDULED"} else "WOULD_GENERATE"
            outcomes.append({"faculty_id":faculty_id,"application_id":app_id,
                             "outreach_package_id":outreach["id"] if outreach else None,
                             "outcome":outcome,"reasons":sorted(set(reasons))})
        report = {"campaign_id":campaign_id,"policy_version":campaign["policy_version"],
                  "auto_send_enabled":bool(campaign["auto_send_enabled"]),"emergency_stop":bool(campaign["emergency_stop"]),
                  "candidates":outcomes,"run_at":utc_now()}
        with transaction(self.db_path) as db:
            run_id = db.execute("""INSERT INTO campaign_dry_runs
                (campaign_id,policy_version,result_json,run_at) VALUES(?,?,?,?)""",
                (campaign_id,campaign["policy_version"],_dump(report),report["run_at"])).lastrowid
        return {"id":run_id,**report}
