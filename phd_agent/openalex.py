"""Reviewed OpenAlex author resolution and sourced publication enrichment."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from phd_agent.db import connect, migrate, transaction, utc_now
from phd_agent.discovery import canonical_url


API_ROOT = "https://api.openalex.org"


def _norm(text: str) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", text.casefold()).split()
    return " ".join(word for word in words if len(word) > 1 and word not in {"dr", "prof", "professor"})


def _short_id(value: str, prefix: str) -> str:
    item = value.rstrip("/").split("/")[-1]
    if not re.fullmatch(prefix + r"\d+", item):
        raise ValueError("Invalid OpenAlex identifier")
    return item


def _abstract(index: dict | None) -> str | None:
    if not index:
        return None
    positions = [(position, word) for word, occurrences in index.items() for position in occurrences]
    if not positions or len(positions) > 2000:
        return None
    ordered = sorted(positions)
    if [p for p, _ in ordered] != list(range(len(ordered))):
        return None
    return " ".join(word for _, word in ordered)


class OpenAlexEnrichment:
    def __init__(self, db_path: Path | str, session=None, api_key: str | None = None):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.session = session or requests.Session()
        self.api_key = api_key or os.environ.get("OPENALEX_API_KEY")

    def _get(self, path: str, params: dict | None = None) -> dict:
        parameters = dict(params or {})
        if self.api_key:
            parameters["api_key"] = self.api_key
        response = self.session.get(API_ROOT + path, params=parameters, timeout=20,
                                    headers={"User-Agent": "PhD-Apply-Agent/2.0 (research review)"})
        response.raise_for_status()
        return response.json()

    def search_authors(self, faculty_id: int) -> list[dict]:
        with connect(self.db_path) as db:
            faculty = db.execute("SELECT * FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
        if not faculty:
            raise ValueError("Faculty not found")
        if faculty["verification_state"] not in {"VERIFIED", "PARTIALLY_VERIFIED"}:
            raise ValueError("Review the institutional faculty identity first")
        data = self._get("/authors", {"search": _norm(faculty["name"]), "per_page": 10})
        candidates = []
        for author in data.get("results", []):
            names = [author.get("display_name") or "", *(author.get("display_name_alternatives") or [])]
            name_match = _norm(faculty["name"]) in {_norm(name) for name in names}
            institutions = [item.get("display_name", "") for item in (author.get("last_known_institutions") or [])]
            institutions.extend(item.get("institution", {}).get("display_name", "")
                                for item in (author.get("affiliations") or []))
            institution_match = any(_norm(faculty["institution"]) == _norm(name) for name in institutions)
            topics = [item.get("display_name", "") for item in (author.get("topics") or [])]
            faculty_terms = set(re.findall(r"[a-z]{5,}", (faculty["research_topics"] or "").casefold()))
            topic_terms = set(re.findall(r"[a-z]{5,}", " ".join(topics).casefold()))
            topic_overlap = sorted(faculty_terms & topic_terms)
            candidates.append({
                "id": _short_id(author["id"], "A"), "display_name": author.get("display_name"),
                "name_match": name_match, "institution_match": institution_match,
                "institutions": institutions, "topics": topics, "topic_overlap": topic_overlap,
                "works_count": author.get("works_count", 0),
            })
        strong = [c for c in candidates if c["name_match"] and c["institution_match"]]
        if faculty["openalex_resolution_state"] != "RESOLVED":
            with transaction(self.db_path) as db:
                db.execute("UPDATE faculty_profiles SET openalex_resolution_state=?,updated_at=? WHERE id=?", (
                    "AMBIGUOUS" if len(strong) > 1 else "NO_MATCH" if not candidates else "UNRESOLVED",
                    utc_now(), faculty_id,
                ))
        return candidates

    def resolve_author(self, faculty_id: int, author_id: str, *, reviewer: str,
                       reason: str = "") -> dict:
        if not reviewer.strip():
            raise ValueError("Author resolution requires a reviewer")
        short_id = _short_id(author_id, "A")
        with connect(self.db_path) as db:
            faculty = db.execute("SELECT * FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
        if not faculty or faculty["verification_state"] not in {"VERIFIED", "PARTIALLY_VERIFIED"}:
            raise ValueError("Institutional faculty identity must be reviewed first")
        if faculty["openalex_resolution_state"] == "RESOLVED":
            if faculty["openalex_author_id"] != short_id:
                raise ValueError("A resolved author cannot be silently replaced")
            return {"author_id": short_id, "evidence_id": None,
                    "institution_match": None, "topic_overlap": []}
        author = self._get(f"/authors/{short_id}")
        names = [author.get("display_name") or "", *(author.get("display_name_alternatives") or [])]
        if _norm(faculty["name"]) not in {_norm(name) for name in names}:
            raise ValueError("OpenAlex author name conflicts with reviewed faculty name")
        institutions = [item.get("display_name", "") for item in (author.get("last_known_institutions") or [])]
        institutions.extend(item.get("institution", {}).get("display_name", "")
                            for item in (author.get("affiliations") or []))
        institution_match = any(_norm(faculty["institution"]) == _norm(name) for name in institutions)
        topics = [item.get("display_name", "") for item in (author.get("topics") or [])]
        faculty_terms = set(re.findall(r"[a-z]{5,}", (faculty["research_topics"] or "").casefold()))
        topic_terms = set(re.findall(r"[a-z]{5,}", " ".join(topics).casefold()))
        topic_overlap = sorted(faculty_terms & topic_terms)
        if not institution_match and not topic_overlap:
            raise ValueError("Author needs an affiliation or topic corroboration before resolution")
        now = utc_now()
        excerpt = f"OpenAlex author {short_id}: {author.get('display_name')}; institutions: {institutions}; topics: {topics}. Reviewer: {reviewer}. {reason}"
        with transaction(self.db_path) as db:
            evidence_id = db.execute("""INSERT INTO source_evidence
                (canonical_url,source_type,retrieved_at,relevant_excerpt,content_hash,
                 verification_state,last_manually_verified_at,created_at)
                VALUES(?,?,?,?,?,?,?,?)""", (
                f"{API_ROOT}/authors/{short_id}", "OPENALEX_AUTHOR", now, excerpt,
                hashlib.sha256(json.dumps(author, sort_keys=True).encode()).hexdigest(),
                "VERIFIED", now, now,
            )).lastrowid
            db.execute("""UPDATE faculty_profiles SET openalex_author_id=?,
                openalex_resolution_state='RESOLVED',updated_at=? WHERE id=?""", (short_id, now, faculty_id))
        return {"author_id": short_id, "evidence_id": evidence_id,
                "institution_match": institution_match, "topic_overlap": topic_overlap}

    def enrich_works(self, faculty_id: int, *, max_works: int = 10) -> int:
        if not 1 <= max_works <= 25:
            raise ValueError("Fetch 1–25 works at a time")
        with connect(self.db_path) as db:
            faculty = db.execute("SELECT * FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
        if not faculty or faculty["openalex_resolution_state"] != "RESOLVED":
            raise ValueError("Resolve OpenAlex author identity first")
        author_id = faculty["openalex_author_id"]
        data = self._get("/works", {"filter": f"author.id:{author_id}",
                                    "sort": "publication_date:desc", "per_page": max_works})
        inserted = 0
        for work in data.get("results", []):
            try:
                work_id = _short_id(work.get("id") or "", "W")
            except (TypeError, ValueError):
                continue
            author_ids = []
            for entry in work.get("authorships") or []:
                try:
                    author_ids.append(_short_id(((entry.get("author") or {}).get("id") or ""), "A"))
                except (TypeError, ValueError):
                    continue
            if author_id not in author_ids:
                continue
            title = (work.get("title") or work.get("display_name") or "").strip()
            if not title:
                continue
            with connect(self.db_path) as db:
                if db.execute("SELECT 1 FROM publications WHERE faculty_profile_id=? AND openalex_id=?",
                              (faculty_id, work_id)).fetchone():
                    continue
            venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
            authors = [(entry.get("author") or {}).get("display_name") or entry.get("raw_author_name")
                       for entry in work.get("authorships", [])]
            topics = [item.get("display_name") for item in work.get("topics", []) if item.get("display_name")]
            abstract = _abstract(work.get("abstract_inverted_index"))
            now = utc_now()
            with transaction(self.db_path) as db:
                evidence_id = db.execute("""INSERT INTO source_evidence
                    (canonical_url,source_type,retrieved_at,relevant_excerpt,content_hash,
                     verification_state,created_at)
                    VALUES(?,?,?,?,?,?,?)""", (
                    f"{API_ROOT}/works/{work_id}", "OPENALEX_WORK", now,
                    f"{title} ({work.get('publication_year')}); DOI: {work.get('doi') or 'unknown'}; topics: {topics}",
                    hashlib.sha256(json.dumps(work, sort_keys=True).encode()).hexdigest(),
                    "UNVERIFIED", now,
                )).lastrowid
                db.execute("""INSERT INTO publications
                    (faculty_profile_id,title,publication_date,year,doi,openalex_id,venue,
                     authors_json,abstract_text,topics_json,source_evidence_id,retrieved_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    faculty_id, title, work.get("publication_date"), work.get("publication_year"),
                    work.get("doi"), work_id, venue, json.dumps(authors), abstract,
                    json.dumps(topics), evidence_id, now,
                ))
            inserted += 1
        return inserted

    def corroborate_publication(self, faculty_id: int, publication_id: int,
                                source_url: str, *, reviewer: str) -> int:
        """Append a reviewed external source linking an author to a stored work."""
        if not reviewer.strip():
            raise ValueError("A reviewer is required")
        with connect(self.db_path) as db:
            faculty = db.execute("SELECT * FROM faculty_profiles WHERE id=?", (faculty_id,)).fetchone()
            work = db.execute("SELECT * FROM publications WHERE id=? AND faculty_profile_id=?",
                              (publication_id, faculty_id)).fetchone()
        if not faculty or not work or faculty["openalex_resolution_state"] != "RESOLVED":
            raise ValueError("A resolved faculty author and associated publication are required")
        url = canonical_url(source_url)
        response = self.session.get(url, timeout=20,
            headers={"User-Agent": "PhD-Apply-Agent/2.0 (research review)"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        full_text = " ".join(soup.get_text(" ", strip=True).split())
        normalized = full_text.casefold()
        if " ".join(work["title"].casefold().split()) not in normalized:
            raise ValueError("The source does not mention the stored publication title")
        if _norm(faculty["name"]) not in _norm(full_text):
            raise ValueError("The source does not name the reviewed faculty member")
        title_offset = normalized.index(" ".join(work["title"].casefold().split()))
        excerpt = full_text[max(0, title_offset - 120):title_offset + len(work["title"]) + 300]
        now = utc_now()
        with transaction(self.db_path) as db:
            evidence_id = db.execute("""INSERT INTO source_evidence
                (canonical_url,source_type,retrieved_at,relevant_excerpt,content_hash,
                 verification_state,last_manually_verified_at,created_at)
                VALUES(?,?,?,?,?,?,?,?)""", (
                url, "PUBLICATION_CORROBORATION", now, excerpt,
                hashlib.sha256(full_text.encode()).hexdigest(), "VERIFIED", now, now,
            )).lastrowid
            db.execute("""INSERT INTO faculty_evidence_links
                (faculty_profile_id,fact_type,source_evidence_id,linked_at,notes)
                VALUES(?,?,?,?,?)""", (
                faculty_id, "OPENALEX_CORROBORATION", evidence_id, now,
                f"Publication #{publication_id}; reviewer {reviewer.strip()}",
            ))
        return evidence_id
