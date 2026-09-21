"""Evidence-backed university country and QS World University Rankings 2027 lookup.

Ranks are stored as reference records with a source URL. Matching is exact or alias-based
only — similar names never inherit another university's rank. QS rank is never mixed into
research-fit scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

QS_RANKING_SYSTEM = "QS_WORLD_UNIVERSITY_RANKINGS"
QS_RANKING_YEAR = 2027
QS_SOURCE_URL = "https://www.topuniversities.com/world-university-rankings"

# Compact 2027 reference catalog (overall WUR only). Unlisted names stay UNKNOWN.
# Sources: QS 2027 results overview, QS/PR Newswire 18 June 2026 release, and
# university announcements that cite the same edition.
QS_WUR_2027 = (
    ("Massachusetts Institute of Technology", "1", 1, None, None, "United States", "US",
     ("MIT", "Massachusetts Institute of Technology (MIT)")),
    ("Imperial College London", "=2", 2, None, None, "United Kingdom", "GB",
     ("Imperial", "Imperial College")),
    ("Stanford University", "=2", 2, None, None, "United States", "US",
     ("Stanford",)),
    ("University of Oxford", "4", 4, None, None, "United Kingdom", "GB",
     ("Oxford", "Oxford University")),
    ("Harvard University", "5", 5, None, None, "United States", "US",
     ("Harvard",)),
    ("University of Cambridge", "6", 6, None, None, "United Kingdom", "GB",
     ("Cambridge", "Cambridge University")),
    ("California Institute of Technology", "7", 7, None, None, "United States", "US",
     ("Caltech", "California Institute of Technology (Caltech)")),
    ("ETH Zurich", "=8", 8, None, None, "Switzerland", "CH",
     ("ETH Zürich", "ETH Zurich – Swiss Federal Institute of Technology",
      "Swiss Federal Institute of Technology Zurich", "ETHZ")),
    ("University College London", "=8", 8, None, None, "United Kingdom", "GB",
     ("UCL",)),
    ("National University of Singapore", "10", 10, None, None, "Singapore", "SG",
     ("NUS",)),
    ("University of Hong Kong", "11", 11, None, None, "Hong Kong", "HK",
     ("The University of Hong Kong", "HKU")),
    ("Nanyang Technological University", "12", 12, None, None, "Singapore", "SG",
     ("NTU", "Nanyang Technological University, Singapore")),
    ("Peking University", "13", 13, None, None, "China", "CN", ()),
    ("Tsinghua University", "14", 14, None, None, "China", "CN", ()),
    ("University of Pennsylvania", "15", 15, None, None, "United States", "US",
     ("UPenn", "Penn")),
    ("Cornell University", "16", 16, None, None, "United States", "US", ()),
    ("Yale University", "16", 16, None, None, "United States", "US",
     ("Yale",)),
    ("Chinese University of Hong Kong", "18", 18, None, None, "Hong Kong", "HK",
     ("The Chinese University of Hong Kong", "CUHK")),
    ("University of New South Wales", "19", 19, None, None, "Australia", "AU",
     ("UNSW", "The University of New South Wales")),
    ("Johns Hopkins University", "20", 20, None, None, "United States", "US",
     ("JHU",)),
    ("University of California, Berkeley", "20", 20, None, None, "United States", "US",
     ("UC Berkeley", "UCB", "University of California Berkeley",
      "University of California, Berkeley (UCB)")),
    ("EPFL", "22", 22, None, None, "Switzerland", "CH",
     ("École polytechnique fédérale de Lausanne", "Ecole Polytechnique Federale de Lausanne",
      "Swiss Federal Institute of Technology Lausanne")),
    ("University of Melbourne", "23", 23, None, None, "Australia", "AU",
     ("The University of Melbourne",)),
    ("McGill University", "30", 30, None, None, "Canada", "CA",
     ("McGill",)),
    ("University of Toronto", "32", 32, None, None, "Canada", "CA",
     ("U of T", "UofT", "Toronto")),
    ("University of Edinburgh", "35", 35, None, None, "United Kingdom", "GB",
     ("The University of Edinburgh", "Edinburgh")),
    ("University of Manchester", "40", 40, None, None, "United Kingdom", "GB",
     ("The University of Manchester", "Manchester")),
    ("University of Liverpool", "139", 139, None, None, "United Kingdom", "GB",
     ("Liverpool", "The University of Liverpool")),
)

COUNTRY_NAMES = {
    "united states": ("United States", "US"),
    "usa": ("United States", "US"),
    "u.s.": ("United States", "US"),
    "u.s.a.": ("United States", "US"),
    "united kingdom": ("United Kingdom", "GB"),
    "uk": ("United Kingdom", "GB"),
    "britain": ("United Kingdom", "GB"),
    "england": ("United Kingdom", "GB"),
    "scotland": ("United Kingdom", "GB"),
    "wales": ("United Kingdom", "GB"),
    "canada": ("Canada", "CA"),
    "switzerland": ("Switzerland", "CH"),
    "germany": ("Germany", "DE"),
    "netherlands": ("Netherlands", "NL"),
    "france": ("France", "FR"),
    "ireland": ("Ireland", "IE"),
    "sweden": ("Sweden", "SE"),
    "denmark": ("Denmark", "DK"),
    "finland": ("Finland", "FI"),
    "austria": ("Austria", "AT"),
    "belgium": ("Belgium", "BE"),
    "spain": ("Spain", "ES"),
    "italy": ("Italy", "IT"),
    "norway": ("Norway", "NO"),
    "portugal": ("Portugal", "PT"),
    "australia": ("Australia", "AU"),
    "singapore": ("Singapore", "SG"),
    "hong kong": ("Hong Kong", "HK"),
    "china": ("China", "CN"),
    "japan": ("Japan", "JP"),
    "south korea": ("South Korea", "KR"),
    "india": ("India", "IN"),
    "new zealand": ("New Zealand", "NZ"),
}

REGION_COUNTRY_CODES = {
    "UK": {"GB"},
    "US": {"US"},
    "CANADA": {"CA"},
    "AUSTRALIA": {"AU"},
    "SINGAPORE": {"SG"},
    "EUROPE": {
        "GB", "CH", "DE", "NL", "FR", "IE", "SE", "DK", "FI", "AT", "BE", "ES", "IT",
        "NO", "PT",
    },
}

EUROPEAN_CODES = REGION_COUNTRY_CODES["EUROPE"]


def normalize_university_name(value: str) -> str:
    text = re.sub(r"[^\w\s]", " ", (value or "").casefold())
    text = re.sub(r"\b(the|university|of|at|and)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_rank_display(display: str) -> tuple[int | None, int | None, int | None]:
    raw = (display or "").strip().replace(",", "")
    if not raw:
        return None, None, None
    band = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", raw)
    if band:
        low, high = int(band.group(1)), int(band.group(2))
        return None, low, high
    tied = re.fullmatch(r"=(\d+)", raw)
    if tied:
        rank = int(tied.group(1))
        return rank, None, None
    if re.fullmatch(r"\d+", raw):
        rank = int(raw)
        return rank, None, None
    return None, None, None


@dataclass(frozen=True)
class UniversityMatch:
    canonical_name: str
    country: str | None = None
    country_code: str | None = None
    qs_ranking_system: str = QS_RANKING_SYSTEM
    qs_ranking_year: int = QS_RANKING_YEAR
    qs_rank_display: str | None = None
    qs_rank_numeric: int | None = None
    qs_rank_band_low: int | None = None
    qs_rank_band_high: int | None = None
    qs_source_url: str = QS_SOURCE_URL
    qs_match_state: str = "UNKNOWN"
    country_source: str = "UNKNOWN"
    country_match_state: str = "UNKNOWN"


def lookup_catalog(university: str) -> UniversityMatch:
    needle = normalize_university_name(university)
    if not needle:
        return UniversityMatch(canonical_name=university.strip(), qs_match_state="UNKNOWN")
    for row in QS_WUR_2027:
        name, display, numeric, band_low, band_high, country, code, aliases = row
        names = (name, *aliases)
        if any(normalize_university_name(alias) == needle for alias in names):
            exact = normalize_university_name(name) == needle
            return UniversityMatch(
                canonical_name=name, country=country, country_code=code,
                qs_rank_display=display, qs_rank_numeric=numeric,
                qs_rank_band_low=band_low, qs_rank_band_high=band_high,
                qs_match_state="EXACT" if exact else "ALIAS_MATCH",
                country_source="QS", country_match_state="QS_ONLY",
            )
    return UniversityMatch(canonical_name=university.strip(), qs_match_state="UNKNOWN")


def extract_country(text: str | None) -> tuple[str | None, str | None, str | None]:
    haystack = f" {(text or '').casefold()} "
    for needle, (country, code) in sorted(COUNTRY_NAMES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(needle)}\b", haystack):
            return country, code, needle
    return None, None, None


def enrich_university(university: str, *, official_country: str | None = None,
                      page_text: str | None = None) -> UniversityMatch:
    match = lookup_catalog(university)
    page_country, page_code, _ = extract_country(page_text)
    official = (official_country or "").strip()
    official_code = COUNTRY_NAMES.get(official.casefold(), (official or None, None))[1] if official else None
    if official:
        country, code, source = official, official_code or match.country_code, "OFFICIAL"
    elif page_country:
        country, code, source = page_country, page_code, "OFFICIAL"
    else:
        country, code, source = match.country, match.country_code, match.country_source
    country_state = source
    if match.country_code and code and match.country_code != code:
        country_state = "COUNTRY_MATCH_NEEDS_REVIEW"
    elif source == "OFFICIAL" and match.country_code and code == match.country_code:
        country_state = "CONFIRMED"
    return UniversityMatch(
        canonical_name=match.canonical_name or university.strip(),
        country=country, country_code=code,
        qs_rank_display=match.qs_rank_display, qs_rank_numeric=match.qs_rank_numeric,
        qs_rank_band_low=match.qs_rank_band_low, qs_rank_band_high=match.qs_rank_band_high,
        qs_match_state=match.qs_match_state, country_source=source,
        country_match_state=country_state,
    )


def qs_sort_key(payload: dict) -> tuple[int, int]:
    numeric = payload.get("qs_rank_numeric")
    if numeric is not None:
        return (0, int(numeric))
    low = payload.get("qs_rank_band_low")
    if low is not None:
        return (1, int(low))
    return (2, 10**9)


def _rank_value(payload: dict) -> int | None:
    if payload.get("qs_rank_numeric") is not None:
        return int(payload["qs_rank_numeric"])
    if payload.get("qs_rank_band_low") is not None:
        return int(payload["qs_rank_band_low"])
    return None


def funding_looks_funded(text: str | None) -> bool:
    value = (text or "").casefold()
    if not value or value in {"unknown", "none"}:
        return False
    if re.search(r"\b(unfunded|self.fund|no funding)\b", value):
        return False
    return bool(re.search(r"\b(fund|studentship|scholarship|stipend|fully funded)\b", value))


def parse_preference_filters(intent_text: str, existing: dict | None = None) -> dict:
    filters = dict(existing or {})
    text = intent_text or ""
    qs_match = re.search(r"\b(?:qs\s+)?top\s*(25|50|100|150|200)\b", text, re.I)
    if qs_match:
        filters["qs_max"] = int(qs_match.group(1))
    countries = set(filters.get("country_codes") or [])
    for region, codes in REGION_COUNTRY_CODES.items():
        if re.search(rf"\b{re.escape(region)}\b", text, re.I) or (
                region == "UK" and re.search(r"\bunited kingdom\b", text, re.I)):
            countries.update(codes)
    for name, (_country, code) in COUNTRY_NAMES.items():
        if len(name) < 4:
            continue
        if re.search(rf"\b{re.escape(name)}\b", text, re.I):
            countries.add(code)
    if countries:
        filters["country_codes"] = sorted(countries)
    return filters


def candidate_matches_filters(payload: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    countries = set(filters.get("country_codes") or [])
    if countries:
        code = (payload.get("country_code") or "").upper()
        if code not in countries:
            return False
    qs_max = filters.get("qs_max")
    if qs_max:
        rank = _rank_value(payload)
        if rank is None or rank > int(qs_max):
            return False
    if filters.get("funded_only") and not funding_looks_funded(payload.get("funding")):
        return False
    minimum_fit = filters.get("min_research_fit")
    if minimum_fit is not None:
        try:
            fit = float(payload.get("research_fit") or 0)
        except (TypeError, ValueError):
            fit = 0
        if fit < float(minimum_fit):
            return False
    return True


def enrichment_payload(match: UniversityMatch) -> dict:
    return {
        "country": match.country,
        "country_code": match.country_code,
        "country_source": match.country_source,
        "country_match_state": match.country_match_state,
        "qs_ranking_system": match.qs_ranking_system,
        "qs_ranking_year": match.qs_ranking_year,
        "qs_rank_display": match.qs_rank_display,
        "qs_rank_numeric": match.qs_rank_numeric,
        "qs_rank_band_low": match.qs_rank_band_low,
        "qs_rank_band_high": match.qs_rank_band_high,
        "qs_source_url": match.qs_source_url,
        "qs_match_state": match.qs_match_state,
    }


def seed_university_rankings(db) -> None:
    from phd_agent.db import utc_now
    now = utc_now()
    for name, display, numeric, band_low, band_high, country, code, aliases in QS_WUR_2027:
        existing = db.execute(
            """SELECT id FROM university_rankings
               WHERE canonical_name=? AND qs_ranking_system=? AND qs_ranking_year=?""",
            (name, QS_RANKING_SYSTEM, QS_RANKING_YEAR),
        ).fetchone()
        if existing:
            ranking_id = existing["id"]
        else:
            ranking_id = db.execute(
                """INSERT INTO university_rankings
                    (canonical_name,country,country_code,qs_ranking_system,qs_ranking_year,
                     qs_rank_display,qs_rank_numeric,qs_rank_band_low,qs_rank_band_high,
                     source_url,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (name, country, code, QS_RANKING_SYSTEM, QS_RANKING_YEAR, display, numeric,
                 band_low, band_high, QS_SOURCE_URL, now),
            ).lastrowid
        for alias in (name, *aliases):
            normalized = normalize_university_name(alias)
            if not normalized:
                continue
            db.execute(
                """INSERT OR IGNORE INTO university_aliases
                    (ranking_id,alias,alias_normalized) VALUES(?,?,?)""",
                (ranking_id, alias, normalized),
            )
