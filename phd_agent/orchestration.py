"""Intent-first orchestration over the evidence ledger and reviewed services."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from phd_agent.applicant_context import ApplicantResearchContextService, normalized_terms
from phd_agent.browser_worker import BrowserWorker
from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.discovery import Discovery, canonical_url, freshness
from phd_agent.ledger import Ledger
from phd_agent.matching import contact_policy_state
from phd_agent.model_router import ModelRouter
from phd_agent.official_search import OfficialSearchProvider, OpenAIOfficialSearchProvider
from phd_agent.packages import PackageBuilder
from phd_agent.university_enrichment import (
    QS_SOURCE_URL, candidate_matches_filters, enrich_university, enrichment_payload,
    parse_preference_filters,
)


USER_AGENT = "PhD-Application-Agent/1.0 (+operator-supervised programme review)"
MAX_PAGE_BYTES = 2_000_000
MIN_USEFUL_PAGE_TEXT = 300
DOCUMENT_SIGNALS = {
    "CV": ("curriculum vitae", "cv", "resume"),
    "SOP": ("statement of purpose", "sop"),
    "PERSONAL_STATEMENT": ("personal statement",),
    "RESEARCH_PROPOSAL": ("research proposal", "project proposal"),
    "RESEARCH_STATEMENT": ("research statement",),
    "TRANSCRIPT": ("transcript", "academic record"),
    "DEGREE_CERTIFICATE": ("degree certificate", "degree award"),
    "ENGLISH_TEST": ("ielts", "toefl", "english language"),
    "COVER_LETTER": ("cover letter", "covering letter"),
}


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_reference_url(url: str) -> str:
    value = canonical_url(url)
    parsed = urlparse(value)
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not accepted")
    host = parsed.hostname or ""
    if host.casefold() == "localhost" or "." not in host:
        raise ValueError("Use an official public programme URL")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and not literal.is_global:
        raise ValueError("Programme retrieval is limited to public internet hosts")
    return value


def _public_url(url: str) -> str:
    value = _safe_reference_url(url)
    parsed = urlparse(value)
    host = parsed.hostname or ""
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)}
    except socket.gaierror as error:
        raise ValueError("The programme host could not be resolved") from error
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("Programme retrieval is limited to public internet hosts")
    return value


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _snippet(text: str, match: re.Match | None, radius: int = 130) -> str:
    if not match:
        return ""
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return _clean_text(text[start:end])


def _first(pattern: str, text: str, flags=re.I) -> tuple[str | None, str]:
    match = re.search(pattern, text, flags)
    return ((match.group(1).strip() if match else None), _snippet(text, match))


def _iso_deadline(text: str) -> tuple[str | None, str, str | None]:
    match = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b", text)
    if match:
        value = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
        return value, _snippet(text, match), None
    month = re.search(
        r"\b(?:deadline|closes?|apply by|applications? due)\D{0,50}"
        r"((?:0?[1-9]|[12]\d|3[01])\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2})",
        text, re.I,
    )
    if month:
        try:
            value = datetime.strptime(month.group(1), "%d %B %Y").date().isoformat()
            return value, _snippet(text, month), None
        except ValueError:
            pass
    raw, excerpt = _first(r"(?:deadline|closes?|apply by|applications? due)\s*[:\-]?\s*([^.;\n]{4,80})", text)
    return None, excerpt, raw


@dataclass(frozen=True)
class Acquisition:
    status: str
    method: str
    url: str
    text: str = ""
    html: str = ""
    reason: str | None = None
    browser_fallback_allowed: bool = True


class ProgrammeOrchestrator:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.ledger = Ledger(self.db_path)
        self.contexts = ApplicantResearchContextService(self.db_path)

    def create_intent(self, intent_text: str, context_id: int, *, filters: dict | None = None) -> dict:
        intent = intent_text.strip()
        if len(intent) < 12:
            raise ValueError("Describe the research area, cycle, funding need, or target region")
        self.contexts.get(context_id)
        retrieval = self.contexts.retrieve(context_id, "DISCOVERY_QUERY", intent, top_k=6,
                                           use="exploration", include_proposed=True)
        inferred = sorted({term for item in retrieval.items for term in item.matched_terms})
        parsed_filters = parse_preference_filters(intent, {
            "years": sorted(set(re.findall(r"\b20\d{2}\b", intent))),
            "funded_only": bool(re.search(r"\bfunded|funding|scholarship|studentship\b", intent, re.I)),
            "regions": [region for region in ("UK", "Europe", "US", "Canada", "Australia", "Singapore")
                        if re.search(rf"\b{re.escape(region)}\b", intent, re.I)],
            "context_terms": inferred,
            **(filters or {}),
        })
        with transaction(self.db_path) as db:
            intent_id = db.execute("""INSERT INTO discovery_intents
                (intent_text,filters_json,applicant_context_id,created_at) VALUES(?,?,?,?)""",
                (intent, _dump(parsed_filters), context_id, utc_now())).lastrowid
        return {"id": intent_id, "intent_text": intent, "filters": parsed_filters,
                "retrieval_id": retrieval.id,
                "relevant_experience": [item.text for item in retrieval.items]}

    def list_intents(self, *, active_only: bool = True) -> list[dict]:
        sql = "SELECT * FROM discovery_intents"
        if active_only:
            sql += " WHERE status='ACTIVE'"
        sql += " ORDER BY id DESC"
        with connect(self.db_path) as db:
            rows = [dict(row) for row in db.execute(sql)]
        for row in rows:
            row["filters"] = json.loads(row.pop("filters_json"))
        return rows

    def discover_from_ledger(self, intent_id: int) -> list[dict]:
        intent = self._intent(intent_id)
        query_terms = normalized_terms(intent["intent_text"] + " " + " ".join(intent["filters"].get("context_terms", [])))
        with connect(self.db_path) as db:
            rows = [dict(row) for row in db.execute("""SELECT p.*,
                o.id AS opportunity_id,o.title AS opportunity_title,o.research_area,o.funding_text,
                o.eligibility_text,o.contact_policy,o.opening_status,o.verification_state,
                o.source_evidence_id
                FROM programmes p LEFT JOIN opportunities o ON o.programme_id=p.id
                ORDER BY p.id""")]
        candidates = []
        for row in rows:
            url = row["programme_url"] or row["admissions_url"] or row["portal_url"]
            if not url:
                continue
            haystack = " ".join(str(row.get(field) or "") for field in (
                "university", "programme_name", "department", "degree_type", "notes",
                "opportunity_title", "research_area", "funding_text", "eligibility_text",
            ))
            shared = query_terms & normalized_terms(haystack)
            if not shared:
                continue
            payload = {
                "university": row["university"], "department": row["department"],
                "programme": row["programme_name"], "degree": row["degree_type"],
                "intake": row["cycle"], "deadline": None, "deadline_timezone": None,
                "funding": row["funding_text"], "eligibility": row["eligibility_text"],
                "fees": None, "english_requirements": None, "required_documents": [],
                "research_proposal_requirement": "UNKNOWN", "statement_requirement": "UNKNOWN",
                "cv_requirement": "UNKNOWN", "referee_count": None,
                "supervisor_contact_policy": row["contact_policy"],
                "application_route": row["portal_url"], "official_application_url": row["portal_url"],
                "research_area": row["research_area"], "opening_status": row["opening_status"],
                "existing_programme_id": row["id"], "existing_opportunity_id": row["opportunity_id"],
                "unknown_fields": ["deadline", "fees", "required_documents"],
                **enrichment_payload(enrich_university(row["university"])),
            }
            field_evidence = {key: {"state": "VERIFIED" if row.get("verification_state") == "VERIFIED" else "UNCERTAIN",
                                    "excerpt": "Existing reviewed ledger record"}
                              for key, value in payload.items() if value not in (None, "", [], "UNKNOWN")}
            candidate_id = self._save_candidate(intent, url, payload, field_evidence,
                                                min(0.95, 0.45 + len(shared) / max(4, len(query_terms))),
                                                source_evidence_id=row["source_evidence_id"])
            candidates.append(self.get_candidate(candidate_id))
        return candidates

    def discover_official_web(self, intent_id: int, *, api_key: str = "",
                              provider: OfficialSearchProvider | None = None) -> dict:
        """Propose official URLs with web search, then acquire and structure each page locally."""
        intent = self._intent(intent_id)
        retrieval = self.contexts.retrieve(
            intent["applicant_context_id"], "DISCOVERY_PLANNING", intent["intent_text"],
            top_k=6, use="exploration", include_proposed=True,
        )
        route = ModelRouter().route("DISCOVERY_PLANNING")
        search = provider or OpenAIOfficialSearchProvider(api_key)
        hits = search.search(intent["intent_text"], tuple(item.text for item in retrieval.items), route)
        created = []
        fallbacks = []
        for hit in hits:
            if hit.source_kind not in {"PROGRAMME", "ADMISSIONS", "VACANCY", "PROJECT", "FUNDING"}:
                continue
            result = self.analyse_url(hit.official_url, intent["applicant_context_id"], intent_id=intent_id)
            if result["status"] == "CANDIDATE_READY":
                candidate = result["candidate"]
                candidate["field_evidence"]["web_discovery"] = {
                    "state": "MODEL_PROPOSED_URL", "provider": route.provider, "model": route.model,
                    "policy_version": route.policy_version, "source_kind": hit.source_kind,
                    "reason": hit.relevance_reason, "title": hit.title, "institution": hit.institution,
                }
                with transaction(self.db_path) as db:
                    db.execute("UPDATE programme_candidates SET field_evidence_json=? WHERE id=? AND review_state='NEW'",
                               (_dump(candidate["field_evidence"]), candidate["id"]))
                self.contexts.link_output(
                    "PROGRAMME_CANDIDATE", candidate["id"], intent["applicant_context_id"], retrieval.id,
                    provider=route.provider, model=route.model, prompt_version=route.policy_version,
                )
                created.append(self.get_candidate(candidate["id"]))
            else:
                fallbacks.append({"url": hit.official_url, "reason": result.get("reason"),
                                  "ingestion_id": result.get("ingestion_id")})
        self.contexts.record_workload("OFFICIAL_URLS_DISCOVERED", len(hits),
                                      entity_type="DISCOVERY_INTENT", entity_id=intent_id,
                                      metadata={"provider": search.provider, "model": route.model})
        return {"route": route, "hits": len(hits), "candidates": created, "human_fallbacks": fallbacks}

    def acquire_static(self, url: str) -> Acquisition:
        safe_url = "invalid://public-url"
        try:
            safe_url = _public_url(url)
            current_url = safe_url
            response = None
            for _ in range(6):
                response = requests.get(current_url, timeout=(5, 20), allow_redirects=False,
                                        headers={"User-Agent": USER_AGENT}, stream=True)
                if response.status_code not in {301, 302, 303, 307, 308}:
                    break
                destination = response.headers.get("location")
                response.close()
                if not destination:
                    raise ValueError("Programme page returned an invalid redirect")
                current_url = _public_url(urljoin(current_url, destination))
            else:
                raise ValueError("Programme page exceeded the redirect limit")
            if response is None:
                raise ValueError("Programme page returned no response")
            final_url = current_url
            if response.status_code in {401, 403, 429}:
                response.close()
                return Acquisition("HUMAN_INPUT_REQUIRED", "STATIC_HTTP", final_url,
                                   reason="The site blocked automated reading or requires human verification")
            response.raise_for_status()
            declared = int(response.headers.get("content-length", "0") or 0)
            if declared > MAX_PAGE_BYTES:
                raise ValueError("Programme page exceeds the safe acquisition size")
            chunks = []
            size = 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > MAX_PAGE_BYTES:
                    raise ValueError("Programme page exceeds the safe acquisition size")
                chunks.append(chunk)
            raw = b"".join(chunks)
            response.close()
            content_type = response.headers.get("content-type", "").casefold()
            if "pdf" in content_type or final_url.casefold().endswith(".pdf"):
                from io import BytesIO
                text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(raw)).pages)
                return Acquisition("ACQUIRED", "STATIC_HTTP", final_url, text=text)
            html = raw.decode(response.encoding or "utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            for node in soup(["script", "style", "noscript", "svg"]):
                node.decompose()
            text = soup.get_text("\n", strip=True)
            if re.search(r"captcha|access denied|verify you are human|enable javascript", text, re.I):
                return Acquisition("HUMAN_INPUT_REQUIRED", "STATIC_HTTP", final_url,
                                   reason="The site blocked automated reading or requires human verification")
            return Acquisition("ACQUIRED", "STATIC_HTTP", final_url, text=text, html=html)
        except ValueError as error:
            return Acquisition("HUMAN_INPUT_REQUIRED", "STATIC_HTTP", safe_url,
                               reason=f"Static acquisition failed: {type(error).__name__}",
                               browser_fallback_allowed=False)
        except requests.RequestException as error:
            return Acquisition("HUMAN_INPUT_REQUIRED", "STATIC_HTTP", safe_url,
                               reason=f"Static acquisition failed: {type(error).__name__}")

    def analyse_url(self, url: str, context_id: int, *, intent_id: int | None = None,
                    browser_worker: BrowserWorker | None = None) -> dict:
        acquisition = self.acquire_static(url)
        if (browser_worker and acquisition.browser_fallback_allowed
                and (acquisition.status != "ACQUIRED" or len(acquisition.text) < MIN_USEFUL_PAGE_TEXT)):
            try:
                rendered = browser_worker.acquire(acquisition.url)
                rendered_url = _public_url(rendered.url)
                acquisition = Acquisition(rendered.status, rendered.method, rendered_url, rendered.text,
                                          rendered.html, rendered.human_action)
            except (RuntimeError, ValueError) as error:
                acquisition = Acquisition("HUMAN_INPUT_REQUIRED", "PLAYWRIGHT", acquisition.url,
                                          reason=str(error), browser_fallback_allowed=False)
        if acquisition.status != "ACQUIRED" or len(acquisition.text) < MIN_USEFUL_PAGE_TEXT:
            reason = acquisition.reason or "The retrieved page did not contain enough readable programme text"
            ingestion_id = self._save_ingestion(intent_id, acquisition.url, acquisition.method,
                                                "HUMAN_INPUT_REQUIRED", "", reason=reason)
            return {"status": "HUMAN_INPUT_REQUIRED", "ingestion_id": ingestion_id,
                    "message": "Automation could not reliably read this page.",
                    "actions": ["Paste page text", "Upload saved HTML/PDF"], "reason": reason}
        return self._analyse_acquired(acquisition.url, acquisition.text, acquisition.html,
                                     acquisition.method, context_id, intent_id)

    def analyse_supplied(self, url: str, content: str | bytes, context_id: int, *,
                         intent_id: int | None = None, method: str = "PASTED_TEXT",
                         filename: str | None = None) -> dict:
        safe_url = _safe_reference_url(url)
        if method not in {"PASTED_TEXT", "UPLOADED_HTML", "UPLOADED_PDF"}:
            raise ValueError("Unsupported human-assisted ingestion method")
        if method == "UPLOADED_PDF":
            from io import BytesIO
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
            raise ValueError("Supply enough page content to identify the programme and requirements")
        return self._analyse_acquired(safe_url, text, html, method, context_id, intent_id,
                                     source_note=f"Operator supplied {filename or method.lower().replace('_', ' ')}")

    def _analyse_acquired(self, url: str, text: str, html: str, method: str,
                          context_id: int, intent_id: int | None, source_note: str | None = None) -> dict:
        self.contexts.get(context_id)
        clean = _clean_text(text)
        evidence_state = "NEEDS_REVIEW" if method != "STATIC_HTTP" else "UNVERIFIED"
        excerpt = clean[:12000]
        evidence_id = self.ledger.create_evidence(url, "PROGRAMME", excerpt, evidence_state)
        ingestion_id = self._save_ingestion(intent_id, url, method, "ACQUIRED", clean,
                                            source_evidence_id=evidence_id)
        payload, field_evidence, confidence = self._extract(url, clean, html, context_id)
        if source_note:
            field_evidence["source_method"] = {"state": "MANUALLY_SUPPLIED", "excerpt": source_note}
        candidate_id = self._save_candidate(
            self._intent(intent_id) if intent_id else {"id": None, "applicant_context_id": context_id},
            url, payload, field_evidence, confidence, ingestion_id=ingestion_id,
            source_evidence_id=evidence_id,
        )
        with transaction(self.db_path) as db:
            db.execute("""INSERT INTO workload_events
                (event_type,count_value,entity_type,entity_id,metadata_json,created_at)
                VALUES('PAGES_INGESTED',1,'PROGRAMME_CANDIDATE',?, ?,?)""",
                (candidate_id, _dump({"method": method}), utc_now()))
            extracted = sum(value not in (None, "", [], "UNKNOWN") for key, value in payload.items()
                            if key not in {"unknown_fields", "relevant_applicant_experience"})
            db.execute("""INSERT INTO workload_events
                (event_type,count_value,entity_type,entity_id,metadata_json,created_at)
                VALUES('FIELDS_EXTRACTED_AUTOMATICALLY',?,'PROGRAMME_CANDIDATE',?,'{}',?)""",
                (extracted, candidate_id, utc_now()))
        return {"status": "CANDIDATE_READY", "candidate": self.get_candidate(candidate_id)}

    def _extract(self, url: str, text: str, html: str, context_id: int) -> tuple[dict, dict, float]:
        soup = BeautifulSoup(html, "html.parser") if html else None
        title = _clean_text((soup.title.get_text(" ") if soup and soup.title else ""))
        h1 = _clean_text((soup.find("h1").get_text(" ") if soup and soup.find("h1") else ""))
        programme = h1 or title.split("|")[0].split("–")[0].strip()
        degree, degree_excerpt = _first(r"\b((?:Doctor of Philosophy|PhD|DPhil)(?:\s+(?:in|programme in)\s+[^.;|]{2,90})?)", text)
        if not programme and degree:
            programme = degree
        host_label = (urlparse(url).hostname or "University").split(".")[-2].replace("-", " ").title()
        university = ""
        for candidate in ([part.strip() for part in re.split(r"[|–—]", title)] if title else []):
            if re.search(r"university|institute|college|school", candidate, re.I):
                university = candidate
                break
        if not university:
            match = re.search(r"([A-Z][A-Za-z&' .-]{2,80}(?:University|Institute|College|School))", text)
            university = _clean_text(match.group(1)) if match else host_label
        department, department_excerpt = _first(r"\b((?:Department|School|Faculty) of [A-Z][A-Za-z& ,'-]{2,90})", text)
        intake, intake_excerpt = _first(r"\b((?:September|October|January|Fall|Autumn|Spring)?\s*20\d{2}\s+(?:entry|intake|start)?)\b", text)
        deadline, deadline_excerpt, raw_deadline = _iso_deadline(text)
        funding, funding_excerpt = _first(r"((?:fully funded|funding|studentship|scholarship)[^.;]{0,220})", text)
        eligibility, eligibility_excerpt = _first(r"((?:eligibility|entry requirements?|minimum requirements?)[^.;]{0,260})", text)
        fees, fees_excerpt = _first(r"((?:tuition fees?|application fee)[^.;]{0,180})", text)
        english, english_excerpt = _first(r"((?:English language|IELTS|TOEFL)[^.;]{0,220})", text)
        docs = []
        doc_evidence = {}
        lowered = f" {text.casefold()} "
        for document_type, signals in DOCUMENT_SIGNALS.items():
            signal = next((value for value in signals if re.search(
                rf"\b{re.escape(value)}\b" if value == "cv" else re.escape(value), lowered, re.I)), None)
            if signal:
                match = re.search(re.escape(signal.strip()), text, re.I)
                state = "REQUIRED" if match and re.search(r"required|must|upload|submit", _snippet(text, match, 100), re.I) else "UNKNOWN"
                docs.append({"document_type": document_type, "state": state,
                             "label": document_type.replace("_", " ").title()})
                doc_evidence[document_type] = _snippet(text, match)
        referee_match = re.search(r"\b(?:two|three|2|3)\s+(?:academic\s+)?(?:references|referees|recommendation letters)\b", text, re.I)
        referee_count = None
        if referee_match:
            referee_count = {"two": 2, "2": 2, "three": 3, "3": 3}[referee_match.group(0).split()[0].casefold()]
        contact_match = re.search(r"(?:contact|identify|name) (?:a |the )?(?:potential )?supervisor[^.;]{0,180}", text, re.I)
        contact_text = _snippet(text, contact_match)
        contact_policy = "UNKNOWN"
        if contact_match:
            if re.search(r"must|required|before applying|expected", contact_text, re.I):
                contact_policy = "CONTACT_REQUIRED"
            elif re.search(r"not required|do not|should not|no need", contact_text, re.I):
                contact_policy = "DO_NOT_CONTACT"
            elif re.search(r"encouraged|may|can|welcome", contact_text, re.I):
                contact_policy = "CONTACT_ALLOWED"
        fit_query = " ".join(filter(None, (programme, degree, department, text[:4000])))
        retrieval = self.contexts.retrieve(context_id, "PROGRAMME_TRIAGE", fit_query,
                                           top_k=3, use="exploration", include_proposed=True)
        demonstrated = [item for item in retrieval.items if item.classification == "DEMONSTRATED"]
        proposed = [item for item in retrieval.items if item.classification == "PROPOSED"]
        fit_score = round(min(10, sum(item.score for item in retrieval.items) / max(1, len(retrieval.items)) / 2), 1)
        university_match = enrich_university(university or host_label, page_text=text)
        payload = {
            "university": university or None, "department": department,
            "programme": programme or degree, "degree": degree,
            "intake": intake, "deadline": deadline, "deadline_raw": raw_deadline,
            "deadline_timezone": None, "funding": funding, "eligibility": eligibility,
            "fees": fees, "english_requirements": english, "required_documents": docs,
            "research_proposal_requirement": next((d["state"] for d in docs if d["document_type"] == "RESEARCH_PROPOSAL"), "UNKNOWN"),
            "statement_requirement": next((d["state"] for d in docs if d["document_type"] in {"SOP", "PERSONAL_STATEMENT"}), "UNKNOWN"),
            "cv_requirement": next((d["state"] for d in docs if d["document_type"] == "CV"), "UNKNOWN"),
            "referee_count": referee_count, "supervisor_contact_policy": contact_policy,
            "application_route": url, "official_application_url": url,
            "research_fit": fit_score,
            "demonstrated_overlap": [item.text for item in demonstrated],
            "proposed_overlap": [item.text for item in proposed],
            "relevant_applicant_experience": [item.text for item in retrieval.items],
            "context_retrieval_id": retrieval.id,
            **enrichment_payload(university_match),
        }
        important = ("university", "programme", "deadline", "funding", "eligibility", "fees",
                     "english_requirements", "required_documents", "referee_count", "supervisor_contact_policy")
        payload["unknown_fields"] = [field for field in important if payload.get(field) in (None, "", [], "UNKNOWN")]
        field_evidence = {
            "university": {"state": "UNCERTAIN" if university == host_label else "EXTRACTED", "excerpt": title or host_label},
            "programme": {"state": "EXTRACTED" if programme else "UNKNOWN", "excerpt": h1 or title},
            "degree": {"state": "EXTRACTED" if degree else "UNKNOWN", "excerpt": degree_excerpt},
            "department": {"state": "EXTRACTED" if department else "UNKNOWN", "excerpt": department_excerpt},
            "intake": {"state": "EXTRACTED" if intake else "UNKNOWN", "excerpt": intake_excerpt},
            "deadline": {"state": "EXTRACTED" if deadline else "UNCERTAIN" if raw_deadline else "UNKNOWN", "excerpt": deadline_excerpt},
            "funding": {"state": "EXTRACTED" if funding else "UNKNOWN", "excerpt": funding_excerpt},
            "eligibility": {"state": "EXTRACTED" if eligibility else "UNKNOWN", "excerpt": eligibility_excerpt},
            "fees": {"state": "EXTRACTED" if fees else "UNKNOWN", "excerpt": fees_excerpt},
            "english_requirements": {"state": "EXTRACTED" if english else "UNKNOWN", "excerpt": english_excerpt},
            "required_documents": {"state": "EXTRACTED" if docs else "UNKNOWN", "excerpt": doc_evidence},
            "referee_count": {"state": "EXTRACTED" if referee_count else "UNKNOWN", "excerpt": _snippet(text, referee_match)},
            "supervisor_contact_policy": {"state": "EXTRACTED" if contact_policy != "UNKNOWN" else "UNKNOWN", "excerpt": contact_text},
            "applicant_context": {"state": "APPROVED", "retrieval_id": retrieval.id,
                                  "claim_revision_ids": list(retrieval.claim_revision_ids)},
            "country": {"state": "EXTRACTED" if university_match.country else "UNKNOWN",
                        "excerpt": university_match.country or "", "source": university_match.country_source,
                        "match_state": university_match.country_match_state},
            "qs_rank": {"state": "CATALOG" if university_match.qs_match_state in {"EXACT", "ALIAS_MATCH"} else "UNKNOWN",
                        "excerpt": university_match.qs_rank_display or "",
                        "match_state": university_match.qs_match_state,
                        "source_url": university_match.qs_source_url},
        }
        extracted_count = sum(value["state"] in {"EXTRACTED", "APPROVED"} for value in field_evidence.values())
        confidence = round(extracted_count / len(field_evidence), 2)
        return payload, field_evidence, confidence

    def _save_ingestion(self, intent_id: int | None, url: str, method: str, status: str,
                        text: str, *, reason: str | None = None,
                        source_evidence_id: int | None = None) -> int:
        digest = hashlib.sha256(text.encode()).hexdigest() if text else None
        with transaction(self.db_path) as db:
            return db.execute("""INSERT INTO page_ingestions
                (intent_id,canonical_url,acquisition_method,status,content_sha256,content_text,
                 failure_reason,source_evidence_id,created_at) VALUES(?,?,?,?,?,?,?,?,?)""", (
                intent_id, url, method, status, digest, text, reason, source_evidence_id, utc_now(),
            )).lastrowid

    def _save_candidate(self, intent: dict, url: str, payload: dict, field_evidence: dict,
                        confidence: float, *, ingestion_id: int | None = None,
                        source_evidence_id: int | None = None) -> int:
        context_id = intent["applicant_context_id"]
        with transaction(self.db_path) as db:
            existing = db.execute("""SELECT id FROM programme_candidates
                WHERE intent_id IS ? AND canonical_url=? AND review_state!='REJECTED'
                ORDER BY id DESC LIMIT 1""", (intent.get("id"), url)).fetchone()
            if existing and ingestion_id is None:
                return existing["id"]
            candidate_id = db.execute("""INSERT INTO programme_candidates
                (intent_id,applicant_context_id,ingestion_id,source_evidence_id,canonical_url,
                 payload_json,field_evidence_json,confidence,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)""", (
                intent.get("id"), context_id, ingestion_id, source_evidence_id, url,
                _dump(payload), _dump(field_evidence), max(0, min(1, confidence)), utc_now(),
            )).lastrowid
        self.contexts.link_output("PROGRAMME_CANDIDATE", candidate_id, context_id,
                                  payload.get("context_retrieval_id"), model="programme-orchestrator-v1")
        return candidate_id

    def _intent(self, intent_id: int | None) -> dict:
        if intent_id is None:
            raise ValueError("Discovery intent not found")
        with connect(self.db_path) as db:
            row = db.execute("SELECT * FROM discovery_intents WHERE id=?", (intent_id,)).fetchone()
        if not row:
            raise ValueError("Discovery intent not found")
        result = dict(row)
        result["filters"] = json.loads(result.pop("filters_json"))
        return result

    def get_candidate(self, candidate_id: int) -> dict:
        with connect(self.db_path) as db:
            row = db.execute("SELECT * FROM programme_candidates WHERE id=?", (candidate_id,)).fetchone()
        if not row:
            raise ValueError("Programme candidate not found")
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        result["field_evidence"] = json.loads(result.pop("field_evidence_json"))
        return result

    def list_candidates(self, *, intent_id: int | None = None,
                        states: tuple[str, ...] = ("NEW", "SHORTLISTED"),
                        extra_filters: dict | None = None) -> list[dict]:
        clauses, args = [], []
        if intent_id is not None:
            clauses.append("intent_id=?")
            args.append(intent_id)
        if states:
            clauses.append("review_state IN (" + ",".join("?" for _ in states) + ")")
            args.extend(states)
        sql = "SELECT id FROM programme_candidates" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY id DESC"
        with connect(self.db_path) as db:
            ids = [row["id"] for row in db.execute(sql, args)]
        items = [self.get_candidate(candidate_id) for candidate_id in ids]
        result = []
        for item in items:
            filters = {}
            if item.get("intent_id"):
                try:
                    filters.update(self._intent(item["intent_id"])["filters"])
                except ValueError:
                    pass
            if extra_filters:
                filters.update({key: value for key, value in extra_filters.items()
                                if value not in (None, "", [], "Any")})
            if candidate_matches_filters(item["payload"], filters):
                result.append(item)
        return result

    def review_candidate(self, candidate_id: int, decision: str, reviewer: str) -> None:
        decision = decision.strip().upper()
        if decision not in {"SHORTLISTED", "REJECTED"} or not reviewer.strip():
            raise ValueError("Choose shortlist or reject and name the reviewer")
        with transaction(self.db_path) as db:
            changed = db.execute("""UPDATE programme_candidates SET review_state=?,reviewed_at=?,reviewed_by=?
                WHERE id=? AND review_state IN ('NEW','SHORTLISTED')""",
                (decision, utc_now(), reviewer.strip(), candidate_id)).rowcount
        if not changed:
            raise ValueError("Candidate is already accepted or unavailable")

    def accept_candidate(self, candidate_id: int, reviewer: str, *, cycle: str | None = None,
                         field_overrides: dict | None = None) -> dict:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        candidate = self.get_candidate(candidate_id)
        if candidate["review_state"] == "ACCEPTED":
            with connect(self.db_path) as db:
                application = db.execute("SELECT opportunity_id FROM applications WHERE id=?",
                                         (candidate["accepted_application_id"],)).fetchone()
            return {"programme_id": candidate["accepted_programme_id"],
                    "opportunity_id": application["opportunity_id"] if application else None,
                    "application_id": candidate["accepted_application_id"],
                    "source_evidence_id": candidate["source_evidence_id"]}
        if candidate["review_state"] == "REJECTED":
            raise ValueError("Rejected candidate cannot be accepted")
        payload = candidate["payload"]
        overrides = field_overrides or {}
        allowed_overrides = {
            "university", "department", "programme", "degree", "intake", "deadline",
            "deadline_timezone", "funding", "eligibility", "fees", "english_requirements",
            "supervisor_contact_policy", "application_route", "official_application_url",
        }
        if set(overrides) - allowed_overrides:
            raise ValueError("Unsupported candidate field override")
        if overrides:
            payload.update({key: value.strip() if isinstance(value, str) else value
                            for key, value in overrides.items()})
            evidence = candidate["field_evidence"]
            for key in overrides:
                evidence[key] = {"state": "OPERATOR_CONFIRMED", "excerpt": f"Confirmed by {reviewer.strip()}"}
            payload["unknown_fields"] = [field for field in payload.get("unknown_fields", [])
                                         if payload.get(field) in (None, "", [], "UNKNOWN")]
            with transaction(self.db_path) as db:
                db.execute("""UPDATE programme_candidates SET payload_json=?,field_evidence_json=?
                    WHERE id=?""", (_dump(payload), _dump(evidence), candidate_id))
            candidate["field_evidence"] = evidence
        if not payload.get("university") or not payload.get("programme"):
            raise ValueError("Review and supply the university and programme name before accepting")
        if payload.get("deadline"):
            try:
                datetime.fromisoformat(payload["deadline"])
            except ValueError as error:
                raise ValueError("Confirm the deadline as an ISO date before accepting") from error
        original_source = (self.ledger.get("source_evidence", candidate["source_evidence_id"])
                           if candidate["source_evidence_id"] else None)
        if not original_source:
            raise ValueError("Analyse the official programme URL before accepting this candidate")
        source = original_source["id"] if original_source["verification_state"] == "VERIFIED" else self.ledger.create_evidence(
            original_source["canonical_url"], original_source["source_type"],
            original_source["relevant_excerpt"], "VERIFIED", last_manually_verified_at=utc_now())
        programme_id = payload.get("existing_programme_id")
        enrichment = {
            key: payload.get(key) for key in (
                "country", "country_code", "country_source", "country_match_state",
                "qs_ranking_system", "qs_ranking_year", "qs_rank_display", "qs_rank_numeric",
                "qs_rank_band_low", "qs_rank_band_high", "qs_source_url", "qs_match_state",
            ) if payload.get(key) not in (None, "")
        }
        if enrichment.get("qs_match_state") in {"EXACT", "ALIAS_MATCH"}:
            enrichment["qs_checked_at"] = utc_now()
            rank_excerpt = (
                f"QS World University Rankings {payload.get('qs_ranking_year')}: "
                f"{payload['university']} is listed as {payload.get('qs_rank_display')}."
            )
            enrichment["qs_source_evidence_id"] = self.ledger.create_evidence(
                payload.get("qs_source_url") or QS_SOURCE_URL, "QS_RANKING", rank_excerpt,
                "UNVERIFIED")
        if not programme_id:
            programme_id = self.ledger.create_programme(
                payload["university"], payload["programme"], department=payload.get("department"),
                degree_type=payload.get("degree"), cycle=cycle or payload.get("intake"),
                programme_url=candidate["canonical_url"], admissions_url=candidate["canonical_url"],
                portal_url=payload.get("official_application_url"),
                notes=f"Accepted from programme candidate #{candidate_id}; evidence #{source}",
                **enrichment,
            )
        elif enrichment:
            self.ledger.update_programme(programme_id, **enrichment)
        opportunity_id = payload.get("existing_opportunity_id")
        if not opportunity_id:
            contact_evidence = candidate["field_evidence"].get("supervisor_contact_policy", {})
            contact_policy = (payload.get("supervisor_contact_policy")
                              if contact_evidence.get("state") == "OPERATOR_CONFIRMED" else "UNKNOWN")
            contact_excerpt = contact_evidence.get("excerpt", "").strip()
            opportunity_id = self.ledger.create_opportunity(
                "PROGRAMME_APPLICATION", payload["programme"], payload["university"],
                programme_id=programme_id, canonical_url=candidate["canonical_url"],
                department_lab=payload.get("department"), research_area=payload.get("research_area"),
                funding_text=payload.get("funding"), eligibility_text=payload.get("eligibility"),
                deadline_at=payload.get("deadline"), deadline_timezone=payload.get("deadline_timezone"),
                application_route=payload.get("application_route"),
                contact_policy=contact_policy, opening_status="UNKNOWN",
                verification_state="VERIFIED", last_checked_at=utc_now(), source_evidence_id=source,
                notes=(f"Unverified supervisor-contact excerpt from evidence #{source}: {contact_excerpt}"
                       if contact_excerpt and contact_evidence.get("state") != "OPERATOR_CONFIRMED" else ""),
            )
        application_id = self.ledger.create_application(
            cycle or payload.get("intake") or "UNKNOWN", programme_id, opportunity_id,
            portal_url=payload.get("official_application_url"),
            next_action="Review extracted unknowns and complete application requirements",
            owner_notes=f"Created from candidate #{candidate_id}",
        )
        if payload.get("deadline"):
            deadline_state = ("VERIFIED" if candidate["field_evidence"].get("deadline", {}).get("state")
                              == "OPERATOR_CONFIRMED" else "NEEDS_REVIEW")
            self.ledger.create_deadline(application_id, "APPLICATION", payload["deadline"], source,
                                        timezone=payload.get("deadline_timezone"), verification_state=deadline_state,
                                        last_checked_at=utc_now())
        recorded_types = set()
        for document in payload.get("required_documents", []):
            self.ledger.create_requirement(
                application_id, "FORMAL_APPLICATION", document.get("state", "UNKNOWN"),
                document.get("label") or document["document_type"], source,
                normalized_document_type=document["document_type"], last_checked_at=utc_now(),
            )
            recorded_types.add(document["document_type"])
        unknown_groups = (
            ({"CV"}, "CV requirement unresolved", "CV"),
            ({"SOP", "PERSONAL_STATEMENT", "RESEARCH_STATEMENT"}, "Statement requirement unresolved", "SOP"),
            ({"RESEARCH_PROPOSAL"}, "Research proposal requirement unresolved", "RESEARCH_PROPOSAL"),
            ({"TRANSCRIPT", "MARKSHEET"}, "Transcript requirement unresolved", "TRANSCRIPT"),
            ({"ENGLISH_TEST"}, "English evidence applicability unresolved", "ENGLISH_TEST"),
        )
        unknown_count = 0
        for alternatives, label, normalized in unknown_groups:
            if recorded_types & alternatives:
                continue
            self.ledger.create_requirement(
                application_id, "FORMAL_APPLICATION", "UNKNOWN", label, source,
                normalized_document_type=normalized, last_checked_at=utc_now(),
            )
            unknown_count += 1
        referee_count = payload.get("referee_count")
        self.ledger.create_task(
            application_id, "REFEREES",
            f"Add and verify {referee_count} referee records" if referee_count else "Confirm referee count and requirements",
            priority="HIGH" if referee_count else "MEDIUM", source_context=f"Evidence #{source}",
        )
        with transaction(self.db_path) as db:
            db.execute("""UPDATE programme_candidates SET review_state='ACCEPTED',reviewed_at=?,reviewed_by=?,
                source_evidence_id=?,accepted_programme_id=?,accepted_application_id=? WHERE id=?""",
                (utc_now(), reviewer.strip(), source, programme_id, application_id, candidate_id))
            db.execute("""INSERT INTO workload_events
                (event_type,count_value,entity_type,entity_id,metadata_json,created_at)
                VALUES('FIELDS_REQUIRING_MANUAL_ENTRY',?,'APPLICATION',?,'{}',?)""",
                (unknown_count, application_id, utc_now()))
        return {"programme_id": programme_id, "opportunity_id": opportunity_id,
                "application_id": application_id, "source_evidence_id": source}

    def professor_cards(self, application_id: int, context_id: int) -> list[dict]:
        with connect(self.db_path) as db:
            app = db.execute("SELECT * FROM applications WHERE id=?", (application_id,)).fetchone()
            if not app:
                raise ValueError("Application not found")
            programme = db.execute("SELECT * FROM programmes WHERE id=?", (app["programme_id"],)).fetchone() if app["programme_id"] else None
            opportunity = db.execute("SELECT * FROM opportunities WHERE id=?", (app["opportunity_id"],)).fetchone() if app["opportunity_id"] else None
            institution = programme["university"] if programme else opportunity["institution"] if opportunity else ""
            faculty = [dict(row) for row in db.execute("""SELECT * FROM faculty_profiles
                WHERE lower(institution)=lower(?) AND verification_state IN ('VERIFIED','PARTIALLY_VERIFIED')
                ORDER BY verification_state,name""", (institution,))]
        cards = []
        for professor in faculty:
            detail_text = " ".join(filter(None, (professor["research_topics"], professor["lab"], professor["department"])))
            with connect(self.db_path) as db:
                publications = [dict(row) for row in db.execute("""SELECT p.*,e.retrieved_at,e.verification_state
                    FROM publications p JOIN source_evidence e ON e.id=p.source_evidence_id
                    WHERE p.faculty_profile_id=? ORDER BY p.year DESC,p.id DESC LIMIT 5""", (professor["id"],))]
                links = [dict(row) for row in db.execute("""SELECT l.fact_type,e.id,e.retrieved_at,e.verification_state
                    FROM faculty_evidence_links l JOIN source_evidence e ON e.id=l.source_evidence_id
                    WHERE l.faculty_profile_id=?""", (professor["id"],))]
            verified_publications = [paper for paper in publications if paper["verification_state"] == "VERIFIED"]
            query = detail_text + " " + " ".join((paper["title"] + " " + (paper["abstract_text"] or ""))
                                                    for paper in verified_publications)
            retrieval = self.contexts.retrieve(context_id, "FACULTY_ALIGNMENT", query,
                                               top_k=3, use="exploration", include_proposed=True)
            demonstrated = [item for item in retrieval.items if item.classification == "DEMONSTRATED"]
            proposed = [item for item in retrieval.items if item.classification == "PROPOSED"]
            evidence_ids = sorted({link["id"] for link in links if link["verification_state"] == "VERIFIED"})
            unknowns = []
            if professor["affiliation_state"] != "CURRENT": unknowns.append("Current affiliation")
            if professor["supervision_state"] == "UNKNOWN": unknowns.append("Current supervision availability")
            if not verified_publications: unknowns.append("Recent verified publications")
            if not professor["email"] or professor["email_state"] != "VERIFIED": unknowns.append("Verified email")
            score = round(min(10, sum(item.score for item in retrieval.items) / max(1, len(retrieval.items)) / 2), 1)
            cards.append({
                "faculty_id": professor["id"], "name": professor["name"],
                "institution": professor["institution"], "department": professor["department"],
                "lab": professor["lab"], "research_topics": professor["research_topics"],
                "recent_work": [paper["title"] for paper in verified_publications[:3]],
                "relevant_applicant_experience": [item.text for item in demonstrated],
                "proposed_overlap": [item.text for item in proposed],
                "overlapping_terms": sorted({term for item in retrieval.items for term in item.matched_terms}),
                "research_fit": score, "fit_is_admission_probability": False,
                "contact_policy": opportunity["contact_policy"] if opportunity else None,
                "contact_policy_state": contact_policy_state(opportunity["contact_policy"] if opportunity else None),
                "application_readiness": "REVIEW_REQUIRED" if unknowns else "READY_FOR_POLICY_CHECK",
                "unknowns": unknowns, "evidence_ids": evidence_ids,
                "context_retrieval_id": retrieval.id,
            })
        return sorted(cards, key=lambda card: (-card["research_fit"], card["name"]))

    def discover_faculty_official_web(self, application_id: int, context_id: int, *, api_key: str = "",
                                      provider: OfficialSearchProvider | None = None) -> dict:
        """Queue faculty candidates from acquired official pages; verification remains an operator decision."""
        with connect(self.db_path) as db:
            app = db.execute("SELECT * FROM applications WHERE id=?", (application_id,)).fetchone()
            if not app:
                raise ValueError("Application not found")
            programme = db.execute("SELECT * FROM programmes WHERE id=?", (app["programme_id"],)).fetchone() if app["programme_id"] else None
            opportunity = db.execute("SELECT * FROM opportunities WHERE id=?", (app["opportunity_id"],)).fetchone() if app["opportunity_id"] else None
        institution = programme["university"] if programme else opportunity["institution"] if opportunity else ""
        programme_name = programme["programme_name"] if programme else opportunity["title"] if opportunity else ""
        applicant_context = self.contexts.get(context_id)
        track = applicant_context["context"]["research_track"]
        query = " ".join(filter(None, (
            institution, programme_name, track.get("title"), track.get("research_problem"),
            track.get("proposed_methodology"), "relevant supervisors and faculty",
        )))
        retrieval = self.contexts.retrieve(context_id, "FACULTY_SYNTHESIS", query,
                                           top_k=6, use="exploration", include_proposed=True)
        route = ModelRouter().route("FACULTY_SYNTHESIS")
        search = provider or OpenAIOfficialSearchProvider(api_key)
        hits = search.search(query, tuple(item.text for item in retrieval.items), route, purpose="FACULTY")
        discovery = Discovery(self.db_path)
        queued = []
        fallbacks = []
        for hit in hits:
            if hit.source_kind not in {"FACULTY", "LAB"} or not hit.person_name:
                continue
            if institution and institution.casefold() not in hit.institution.casefold() and hit.institution.casefold() not in institution.casefold():
                continue
            acquisition = self.acquire_static(hit.official_url)
            if acquisition.status != "ACQUIRED" or len(acquisition.text) < MIN_USEFUL_PAGE_TEXT:
                fallbacks.append({"url": hit.official_url, "reason": acquisition.reason})
                continue
            with connect(self.db_path) as db:
                existing_source = db.execute("""SELECT id FROM source_catalogue
                    WHERE canonical_url=? AND source_type IN ('FACULTY','LAB') ORDER BY id DESC LIMIT 1""",
                    (canonical_url(hit.official_url),)).fetchone()
            source_id = existing_source["id"] if existing_source else discovery.add_source(
                institution, hit.source_kind, hit.official_url, department=hit.department,
                strategy="MANUAL", notes=f"Model-proposed official URL via {route.model}; requires operator verification",
            )
            with connect(self.db_path) as db:
                existing_candidate = db.execute("""SELECT id,source_evidence_id FROM faculty_candidates
                    WHERE source_catalogue_id=? AND lower(name)=lower(?) AND review_state='NEW'""",
                    (source_id, hit.person_name)).fetchone()
            if existing_candidate:
                candidate_id = existing_candidate["id"]
                evidence_id = existing_candidate["source_evidence_id"]
            else:
                evidence_id = self.ledger.create_evidence(
                    hit.official_url, hit.source_kind, _clean_text(acquisition.text)[:12000], "UNVERIFIED")
                candidate_id = discovery.add_faculty_candidate(
                    source_id, hit.person_name, evidence_id, profile_url=hit.official_url,
                    department=hit.department,
                )
            queued.append({
                "candidate_id": candidate_id, "name": hit.person_name, "institution": institution,
                "department": hit.department, "official_url": hit.official_url,
                "relevance_reason": hit.relevance_reason,
                "relevant_applicant_experience": [item.text for item in retrieval.items],
                "evidence_id": evidence_id, "verification_state": "NEEDS_REVIEW",
                "provider": route.provider, "model": route.model,
            })
        self.contexts.record_workload("FACULTY_PAGES_INGESTED", len(queued),
                                      entity_type="APPLICATION", entity_id=application_id,
                                      metadata={"provider": search.provider, "model": route.model})
        return {"queued": queued, "human_fallbacks": fallbacks, "route": route}

    def application_overview(self, application_id: int) -> dict:
        apps = {row["id"]: row for row in self.ledger.list_applications()}
        if application_id not in apps:
            raise ValueError("Application not found")
        app = apps[application_id]
        readiness = self.ledger.readiness(application_id)
        requirements = self.ledger.requirement_rows(application_id, verify=False)
        with connect(self.db_path) as db:
            tasks = [dict(row) for row in db.execute("""SELECT * FROM application_tasks
                WHERE application_id=? AND status!='DONE' ORDER BY priority,due_at""", (application_id,))]
            faculty = [dict(row) for row in db.execute("""SELECT f.id,f.name,f.verification_state,f.supervision_state
                FROM application_faculty af JOIN faculty_profiles f ON f.id=af.faculty_profile_id
                WHERE af.application_id=?""", (application_id,))]
            artifacts = [dict(row) for row in db.execute("""SELECT id,kind,approval_state,generated_at
                FROM generated_artifacts WHERE application_id=? ORDER BY id DESC""", (application_id,))]
            package = db.execute("""SELECT * FROM application_packages WHERE application_id=?
                AND context='FORMAL_APPLICATION' ORDER BY version_number DESC LIMIT 1""", (application_id,)).fetchone()
        return {"application": app, "readiness": readiness, "requirements": requirements,
                "open_tasks": tasks, "faculty": faculty, "generated_documents": artifacts,
                "latest_package": dict(package) if package else None,
                "next_action": app["next_action"] or (tasks[0]["description"] if tasks else "Review requirements")}

    def prepare_application(self, application_id: int, *, context_id: int | None = None) -> dict:
        overview = self.application_overview(application_id)
        package = overview["latest_package"]
        blockers = []
        build_error = None
        if context_id:
            context = self.contexts.get(context_id)
            if context.get("trust_level") != "TRUSTED":
                blockers.append("Confirm your profile with Use this profile before preparing application documents")
        readiness = overview["readiness"]
        if readiness["required_complete"] < readiness["required_total"]:
            blockers.append("Required documents are incomplete")
        if readiness["unknown_requirements"]:
            blockers.append("Unknown requirements need review")
        if not package and context_id and not blockers:
            context = self.contexts.get(context_id)
            try:
                package_id = PackageBuilder(self.db_path).build(
                    application_id, context["profile_version_id"], context["research_track_version_id"])
                with connect(self.db_path) as db:
                    package = dict(db.execute("SELECT * FROM application_packages WHERE id=?", (package_id,)).fetchone())
                overview["latest_package"] = package
            except ValueError as error:
                build_error = str(error)
                blockers.append("Package assembly needs review: " + build_error)
        preflight = PackageBuilder(self.db_path).preflight(package["id"]) if package else None
        if not package:
            blockers.append("No formal application package has been built")
        elif preflight and preflight["status"] == "BLOCK":
            blockers.append("Application preflight is blocked")
        elif package["status"] != "READY":
            blockers.append("The assembled package still needs reviewer approval")
        with connect(self.db_path) as db:
            stale = db.execute("""SELECT COUNT(*) FROM context_staleness
                WHERE reviewed_at IS NULL AND (
                    (output_type='GENERATED_ARTIFACT' AND output_id IN (
                        SELECT id FROM generated_artifacts WHERE application_id=?
                    )) OR
                    (output_type='OUTREACH_PACKAGE' AND output_id IN (
                        SELECT id FROM outreach_packages WHERE application_id=?
                    ))
                )""", (application_id, application_id)).fetchone()[0]
        if stale:
            blockers.append(f"{stale} applicant-context change needs document review")
        return {**overview, "preflight": preflight, "status": "READY" if not blockers else "NOT_READY",
                "blocking": blockers, "package_build_error": build_error}

    def workload_summary(self) -> dict[str, int]:
        return self.contexts.workload_summary()
