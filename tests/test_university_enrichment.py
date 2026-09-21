"""University country, QS 2027 catalog matching, and preference filters."""

from phd_agent.db import connect, migrate
from phd_agent.university_enrichment import (
    QS_RANKING_YEAR, candidate_matches_filters, enrich_university, lookup_catalog,
    parse_preference_filters, parse_rank_display, qs_sort_key,
)


def test_qs_2027_catalog_uses_alias_or_exact_match_only():
    stanford = lookup_catalog("Stanford University")
    assert stanford.qs_match_state == "EXACT"
    assert stanford.qs_rank_display == "=2"
    assert stanford.qs_rank_numeric == 2
    assert stanford.qs_ranking_year == QS_RANKING_YEAR

    berkeley = lookup_catalog("UC Berkeley")
    assert berkeley.qs_match_state == "ALIAS_MATCH"
    assert berkeley.canonical_name == "University of California, Berkeley"
    assert berkeley.qs_rank_numeric == 20

    oxford = lookup_catalog("Oxford")
    assert oxford.canonical_name == "University of Oxford"
    assert oxford.qs_rank_display == "4"

    liverpool = lookup_catalog("University of Liverpool")
    assert liverpool.qs_rank_numeric == 139

    unknown = lookup_catalog("University of California, Los Angeles")
    assert unknown.qs_match_state == "UNKNOWN"
    assert unknown.qs_rank_numeric is None
    assert unknown.qs_rank_display is None


def test_parse_rank_display_supports_ties_and_bands():
    assert parse_rank_display("=2") == (2, None, None)
    assert parse_rank_display("4") == (4, None, None)
    assert parse_rank_display("721–730") == (None, 721, 730)
    assert parse_rank_display("721-730") == (None, 721, 730)


def test_official_country_disagreement_is_flagged_not_guessed():
    matched = enrich_university("ETH Zurich", official_country="Switzerland")
    assert matched.country == "Switzerland"
    assert matched.country_code == "CH"
    assert matched.country_match_state == "CONFIRMED"
    assert matched.qs_rank_display == "=8"

    mismatched = enrich_university("ETH Zurich", official_country="Germany")
    assert mismatched.country == "Germany"
    assert mismatched.country_match_state == "COUNTRY_MATCH_NEEDS_REVIEW"
    assert mismatched.qs_rank_numeric == 8

    page_only = enrich_university("Unknown College", page_text="Campus address, United Kingdom")
    assert page_only.country == "United Kingdom"
    assert page_only.country_code == "GB"
    assert page_only.country_source == "OFFICIAL"
    assert page_only.qs_match_state == "UNKNOWN"


def test_intent_text_becomes_country_and_qs_filters_without_changing_fit():
    filters = parse_preference_filters(
        "Find funded reliable-AI PhDs in the UK, Europe, US and Canada, preferably QS top 150")
    assert filters["qs_max"] == 150
    assert {"GB", "US", "CA", "CH", "DE"} <= set(filters["country_codes"])

    stanford = {"country_code": "US", "qs_rank_numeric": 2, "research_fit": 8.4, "funding": "fully funded"}
    liverpool = {"country_code": "GB", "qs_rank_numeric": 139, "research_fit": 8.4, "funding": "studentship"}
    unranked = {"country_code": "GB", "research_fit": 9.0, "funding": "fully funded"}
    low_fit = {**stanford, "research_fit": 4.0}

    assert candidate_matches_filters(stanford, {**filters, "funded_only": True, "min_research_fit": 7})
    assert candidate_matches_filters(liverpool, filters)
    assert not candidate_matches_filters(unranked, filters)
    assert not candidate_matches_filters(low_fit, {**filters, "min_research_fit": 7})
    assert qs_sort_key(stanford) < qs_sort_key(liverpool) < qs_sort_key(unranked)


def test_migrate_seeds_qs_2027_catalog(tmp_path):
    database = tmp_path / "rankings.db"
    migrate(database)
    with connect(database) as db:
        stanford = db.execute(
            "SELECT qs_rank_display, qs_rank_numeric FROM university_rankings WHERE canonical_name=?",
            ("Stanford University",),
        ).fetchone()
        aliases = {row[0] for row in db.execute(
            "SELECT alias FROM university_aliases WHERE alias_normalized=?", ("uc berkeley",))}
    assert stanford["qs_rank_display"] == "=2"
    assert stanford["qs_rank_numeric"] == 2
    assert "UC Berkeley" in aliases
