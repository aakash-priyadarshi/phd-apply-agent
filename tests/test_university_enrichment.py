"""University country, QS 2027 catalog matching, and preference filters."""

from phd_agent.db import connect, migrate
from phd_agent.ledger import Ledger
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
    assert page_only.country is None
    assert page_only.country_code is None
    assert page_only.country_source == "UNKNOWN"
    assert page_only.country_match_state == "UNKNOWN"
    assert page_only.qs_match_state == "UNKNOWN"

    confirmed = enrich_university("ETH Zurich", page_text="The campus is located in Switzerland.")
    assert confirmed.country == "Switzerland"
    assert confirmed.country_match_state == "CONFIRMED"
    assert confirmed.country_source == "QS"

    welcome = enrich_university(
        "University of Liverpool", page_text="Applicants from India are welcome.")
    assert welcome.country == "United Kingdom"
    assert welcome.country_code == "GB"
    assert welcome.country_match_state == "QS_ONLY"

    located_elsewhere = enrich_university(
        "ETH Zurich", page_text="The campus is located in Germany.")
    assert located_elsewhere.country == "Switzerland"
    assert located_elsewhere.country_code == "CH"
    assert located_elsewhere.country_match_state == "COUNTRY_MATCH_NEEDS_REVIEW"


def test_intent_text_becomes_country_and_qs_filters_without_changing_fit():
    filters = parse_preference_filters(
        "Find funded reliable-AI PhDs in the UK, Europe, US and Canada, preferably QS top 150")
    assert filters["qs_max"] == 150
    assert {"GB", "US", "CA", "CH", "DE"} <= set(filters["country_codes"])
    pronoun = parse_preference_filters("Find programmes that interest us in reliable AI evaluation")
    assert "US" not in set(pronoun.get("country_codes") or [])
    lowercase_uk = parse_preference_filters("Find funded PhD programmes in the uk")
    assert "GB" not in set(lowercase_uk.get("country_codes") or [])
    spelled_uk = parse_preference_filters("Find funded PhD programmes in the United Kingdom")
    assert "GB" in set(spelled_uk["country_codes"])

    coded = enrich_university("ETH Zurich", official_country="CH")
    assert coded.country == "Switzerland"
    assert coded.country_code == "CH"
    assert coded.country_match_state == "CONFIRMED"
    unknown_code = enrich_university("ETH Zurich", official_country="ZZ")
    assert unknown_code.country is None
    assert unknown_code.country_code is None
    assert unknown_code.country_match_state == "COUNTRY_MATCH_NEEDS_REVIEW"

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


def test_opportunity_only_application_inherits_programme_country_and_rank(tmp_path):
    database = tmp_path / "applications.db"
    migrate(database)
    ledger = Ledger(database)
    programme = ledger.create_programme(
        "Stanford University", "PhD Computer Science",
        country="United States", country_code="US",
        qs_rank_display="=2", qs_ranking_year=2027, qs_match_state="EXACT",
    )
    opportunity = ledger.create_opportunity(
        "PROGRAMME_APPLICATION", "PhD Computer Science", "Stanford University",
        programme_id=programme,
    )
    ledger.create_application("2027", opportunity_id=opportunity)
    row = ledger.list_applications()[0]
    assert row["country_code"] == "US"
    assert row["qs_rank_display"] == "=2"
    assert row["qs_match_state"] == "EXACT"
