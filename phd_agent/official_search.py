"""Optional official-source web discovery behind a small provider boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from openai import OpenAI
from pydantic import BaseModel, Field

from phd_agent.model_router import RouteDecision


BLOCKED_DISCOVERY_HOSTS = {
    "findaphd.com", "linkedin.com", "facebook.com", "x.com", "twitter.com",
    "researchgate.net", "reddit.com", "youtube.com",
}


class SearchHit(BaseModel):
    title: str
    institution: str
    official_url: str
    source_kind: str = Field(description="PROGRAMME, ADMISSIONS, VACANCY, PROJECT, or FUNDING")
    relevance_reason: str
    official_source: bool
    person_name: str | None = None
    department: str | None = None


class SearchBatch(BaseModel):
    results: list[SearchHit]


@dataclass(frozen=True)
class OfficialSearchResult:
    title: str
    institution: str
    official_url: str
    source_kind: str
    relevance_reason: str
    person_name: str | None = None
    department: str | None = None


class OfficialSearchProvider(Protocol):
    provider: str

    def search(self, intent: str, relevant_experience: tuple[str, ...],
               route: RouteDecision, *, purpose: str = "PROGRAMME") -> list[OfficialSearchResult]: ...


def _allowed_host(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or not host:
        return False
    return not any(host == blocked or host.endswith("." + blocked) for blocked in BLOCKED_DISCOVERY_HOSTS)


class OpenAIOfficialSearchProvider:
    """Use OpenAI web search only to propose official URLs; every page is acquired and reviewed later."""

    provider = "openai"

    def __init__(self, api_key: str, client=None):
        if not api_key and client is None:
            raise ValueError("An OpenAI API key is required for web discovery")
        self.client = client or OpenAI(api_key=api_key)

    def search(self, intent: str, relevant_experience: tuple[str, ...],
               route: RouteDecision, *, purpose: str = "PROGRAMME") -> list[OfficialSearchResult]:
        purpose = purpose.strip().upper()
        if purpose not in {"PROGRAMME", "FACULTY"}:
            raise ValueError("Official search purpose must be programme or faculty")
        programme_instructions = (
            "Find current PhD programme, university admissions, funded studentship, project, and funding pages "
            "that match the supplied applicant intent and demonstrated background. Return only official university, "
            "department, lab, government funding, or project pages. Exclude directories, rankings, social media, "
            "FindAPhD, reposts, and inferred openings. A publication page is not an opening. Mark official_source "
            "false when uncertain. Do not infer funding, deadlines, eligibility, or supervision from a title."
        )
        faculty_instructions = (
            "Find current faculty or lab profile pages at the named university whose verified research description "
            "could match the supplied applicant intent and demonstrated background. Return only official university, "
            "department, faculty, or lab HTTPS pages. Set source_kind to FACULTY or LAB and person_name to the academic's "
            "name when the page identifies one. Exclude aggregators, social media, publication-only profiles, and inferred "
            "student openings. Do not claim that a professor is accepting students unless the official page says so."
        )
        response = self.client.responses.parse(
            model=route.model,
            tools=[{"type": "web_search"}],
            instructions=faculty_instructions if purpose == "FACULTY" else programme_instructions,
            input=("Applicant intent:\n" + intent + "\n\nRelevant approved applicant experience:\n" +
                   "\n".join(f"- {item}" for item in relevant_experience)),
            text_format=SearchBatch,
            max_tool_calls=6,
            max_output_tokens=3000,
            store=False,
        )
        batch = response.output_parsed
        if not isinstance(batch, SearchBatch):
            raise ValueError("Official-source search returned no structured result")
        results = []
        seen = set()
        for hit in batch.results:
            if not hit.official_source or not _allowed_host(hit.official_url):
                continue
            key = hit.official_url.rstrip("/").casefold()
            if key in seen:
                continue
            seen.add(key)
            results.append(OfficialSearchResult(
                hit.title.strip(), hit.institution.strip(), hit.official_url.strip(),
                hit.source_kind.strip().upper(), hit.relevance_reason.strip(),
                hit.person_name.strip() if hit.person_name else None,
                hit.department.strip() if hit.department else None,
            ))
        return results[:12]
