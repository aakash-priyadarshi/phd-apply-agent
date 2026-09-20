"""One-time, reviewed Slice 2 pilot. Run only after inspecting the official URLs.

The named source excerpts were reviewed on 2026-09-20. The script fetches the
current official pages again, hashes their full text, and retains the excerpt.
It never approves applicant claims, sends mail, or creates applications.
"""

from __future__ import annotations

import argparse

from phd_agent.config import load_settings
from phd_agent.db import connect
from phd_agent.discovery import Discovery, canonical_url
from phd_agent.ledger import Ledger
from phd_agent.openalex import OpenAlexEnrichment
from phd_agent.profile import ApplicantTruth


PILOT = [
    (62, "Chelsea Finn", "Stanford University", "https://profiles.stanford.edu/chelsea-finn",
     "Computer Science and Electrical Engineering", "Assistant Professor", "Intelligence through Robotic Interaction at Scale",
     "robotics; machine learning; robot learning; reinforcement learning", None, "VERIFIED", "CURRENT", "A5005431772"),
    (67, "Diyi Yang", "Stanford University", "https://cs.stanford.edu/~diyiy/",
     "Computer Science", "Assistant Professor", "Stanford NLP Group",
     "natural language processing; large language models; human-AI interaction; agents",
     "diyiy@cs.stanford.edu", "VERIFIED", "CURRENT", "A5089413311"),
    (61, "Jeannette Bohg", "Stanford University", "https://stanford.edu/~bohg/",
     "Computer Science", "Assistant Professor", "Interactive Perception and Robot Learning Lab",
     "robotics; robot learning; computer vision; manipulation", None, "VERIFIED", "CURRENT", "A5021676288"),
    (70, "Michael Bernstein", "Stanford University", "https://profiles.stanford.edu/michael-bernstein",
     "Computer Science", "Professor", None,
     "human-computer interaction; social computing; generative agents; human-AI interaction",
     None, "VERIFIED", "CURRENT", "A5076189854"),
    (124, "Dr. Pieter Abbeel", "UC Berkeley", "https://www2.eecs.berkeley.edu/Faculty/Homepages/abbeel.html",
     "EECS", "Jim Gray Chair in Engineering", "Berkeley Robot Learning Lab",
     "robot learning; deep reinforcement learning; imitation learning; robotics",
     "pabbeel@cs.berkeley.edu", "VERIFIED", "CURRENT", "A5049349154"),
    (123, "Dr. Trevor Darrell", "UC Berkeley", "https://www2.eecs.berkeley.edu/Faculty/Homepages/darrell.html",
     "EECS / Computer Science", "Professor in Residence", None,
     "computer vision; machine learning; visual recognition; perception",
     "trevor@eecs.berkeley.edu", "VERIFIED", "CURRENT", "A5029105520"),
    (75, "Yarin Gal", "University of Oxford", "https://www.cs.ox.ac.uk/people/yarin.gal/",
     "Computer Science", "Associate Professor of Machine Learning", "OATML",
     "Bayesian deep learning; deep reinforcement learning; trustworthy AI; uncertainty",
     "yarin@cs.ox.ac.uk", "VERIFIED", "CURRENT", "A5029186201"),
    (37, "Asada, Harry", "Massachusetts Institute of Technology",
     "https://meche.mit.edu/people/faculty/asada%40mit.edu",
     "Mechanical Engineering", "Ford Professor of Engineering", "d'Arbeloff BioRobotics Lab",
     "wearable robots; human augmentation; robotics; smart actuators",
     None, "VERIFIED", "CURRENT", None),
    (8, "Dr. Fei-Fei Li", "Stanford University", "https://profiles.stanford.edu/fei-fei-li",
     "Computer Science", "Sequoia Professor of Computer Science", "Stanford Vision and Learning Lab",
     "computer vision; machine learning; robotics; spatial intelligence",
     None, "CONFLICT", "CONFLICT", None),
    (9, "Dr. Andrew Ng", "Stanford University", "https://symsys.stanford.edu/people/andrew-ng",
     "Computer Science", "Adjunct Professor", None,
     None,
     None, "CONFLICT", "CONFLICT", None),
]


TRACK_DRAFTS = (
    ("Reliable LLM and agent evaluation", "How can agent behaviour be evaluated reproducibly across realistic tasks?",
     "Evaluation protocols, failure analysis, and evidence-grounded benchmarks."),
    ("Knowledge-grounded AI reliability", "How can retrieval-based AI stay grounded and reveal uncertainty?",
     "RAG evaluation, source attribution, and controlled experiments."),
    ("Robotics and human augmentation", "How can learning systems support embodied tasks and human capabilities?",
     "Robotics, multimodal perception, and human-centred evaluation."),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply the reviewed pilot to the local SQLite database")
    args = parser.parse_args()
    if not args.apply:
        print("Dry run: 10 historical faculty records, 3 draft tracks, 1 programme opportunity. Pass --apply to write.")
        return
    settings = load_settings()
    discovery = Discovery(settings.database_path)
    ledger = Ledger(settings.database_path)
    truth = ApplicantTruth(settings.database_path)
    oa = OpenAlexEnrichment(settings.database_path)
    discovery.import_targets(settings.targets_path)
    print("Historical duplicate pairs queued:", discovery.scan_historical_duplicates())
    profiles = truth.list_profiles()
    profile_id = profiles[0]["id"] if profiles else truth.create_profile("Aakash Priyadarshi")
    if not truth.list_tracks(profile_id):
        for title, problem, method in TRACK_DRAFTS:
            truth.create_track(profile_id, title, research_problem=problem,
                               proposed_methodology=method,
                               notes="Starting direction for applicant review; experience is not asserted.")
    targets = {t["name"]: t["id"] for t in discovery.list_targets()}
    completed = []
    for (legacy_id, name, institution, url, department, title, lab, topics,
         email, state, affiliation, author_id) in PILOT:
        with connect(settings.database_path) as db:
            row = db.execute("SELECT * FROM faculty_profiles WHERE legacy_professor_id=?", (legacy_id,)).fetchone()
        if not row:
            print("Missing historical row", legacy_id)
            continue
        if row["verification_state"] != "NEEDS_REVERIFICATION":
            completed.append((legacy_id, row["verification_state"], "already reviewed"))
            continue
        existing = next((s for s in discovery.list_sources() if s["canonical_url"] == canonical_url(url)
                         and s["university"] == institution and s["source_type"] == "FACULTY"), None)
        source_id = existing["id"] if existing else discovery.add_source(
            institution, "FACULTY", url, target_id=targets.get(institution),
            department=department, strategy="STATIC_HTML", notes="Slice 2 ten-record manual pilot")
        try:
            evidence_id = discovery.snapshot_source(source_id, manually_verified=True)
        except Exception as error:
            if legacy_id != 37:
                print("Source fetch failed", legacy_id, str(error)[:160])
                continue
            # The MIT faculty page was inspected in the browser but intermittently
            # blocks simple HTTP clients. Record a reviewed excerpt with that fact.
            discovery.update_source(source_id, strategy="MANUAL",
                notes="MIT faculty page manually reviewed; static HTTP intermittently unavailable")
            evidence_id = discovery.snapshot_source(source_id, manually_verified=True,
                excerpt="MIT Mechanical Engineering faculty profile: H. Harry Asada, "
                        "Ford Professor of Engineering and Professor of Mechanical Engineering. "
                        "Interests include augmenting human capabilities with wearable robots.")
        facts = {"IDENTITY": evidence_id, "AFFILIATION": evidence_id}
        if topics:
            facts["TOPICS"] = evidence_id
        if email:
            if email.casefold() in discovery.get_evidence(evidence_id)["relevant_excerpt"].casefold():
                facts["EMAIL"] = evidence_id
            else:
                email = None
        notes = ("Current official Stanford source conflicts with historical MIT attribution; "
                 "the original professor row remains unchanged." if state == "CONFLICT" else
                 "Official institutional profile manually reviewed for current identity and research topics.")
        discovery.verify_faculty(row["id"], state, evidence_by_fact=facts,
            reviewer="Slice 2 manual pilot", reason=notes,
            updates={"institution": institution, "department": department,
                     "official_title": title, "lab": lab, "official_profile_url": url,
                     "research_topics": topics, "email": email,
                     "email_state": "VERIFIED" if email else "UNKNOWN",
                     "affiliation_state": affiliation, "supervision_state": "UNKNOWN", "notes": notes})
        completed.append((legacy_id, state, "official source reviewed"))
        if state == "VERIFIED" and author_id:
            try:
                candidates = oa.search_authors(row["id"])
                if author_id not in {c["id"] for c in candidates}:
                    print("OpenAlex identity not in current candidates", legacy_id, author_id)
                    continue
                oa.resolve_author(row["id"], author_id, reviewer="Slice 2 manual pilot",
                    reason="Name and research area reviewed against the official faculty profile and recent work titles.")
                count = oa.enrich_works(row["id"], max_works=5)
                print("OpenAlex works", legacy_id, author_id, count)
            except Exception as error:
                print("OpenAlex review remains unresolved", legacy_id, str(error)[:160])
    print("Faculty pilot:", completed)

    # A retry may complete source review before OpenAlex is reachable. Enrichment
    # is independent and may be resumed without creating duplicate faculty rows.
    for legacy_id, _, _, _, _, _, _, _, _, state, _, author_id in PILOT:
        if state != "VERIFIED" or not author_id:
            continue
        with connect(settings.database_path) as db:
            row = db.execute("SELECT * FROM faculty_profiles WHERE legacy_professor_id=?", (legacy_id,)).fetchone()
        if not row or row["verification_state"] != "VERIFIED" or row["openalex_resolution_state"] == "RESOLVED":
            continue
        try:
            candidates = oa.search_authors(row["id"])
            if author_id not in {c["id"] for c in candidates}:
                print("OpenAlex identity still ambiguous", legacy_id, author_id)
                continue
            oa.resolve_author(row["id"], author_id, reviewer="Slice 2 manual pilot",
                reason="Name and research area reviewed against the official faculty profile and recent work titles.")
            print("OpenAlex works", legacy_id, author_id, oa.enrich_works(row["id"], max_works=5))
        except Exception as error:
            print("OpenAlex review remains unresolved", legacy_id, str(error)[:160])

    for legacy_id, *_ in PILOT:
        with connect(settings.database_path) as db:
            row = db.execute("SELECT * FROM faculty_profiles WHERE legacy_professor_id=?", (legacy_id,)).fetchone()
            if not row or row["verification_state"] == "NEEDS_REVERIFICATION":
                continue
            prior = db.execute("SELECT 1 FROM match_assessments WHERE faculty_profile_id=?", (row["id"],)).fetchone()
            evidence = db.execute("""SELECT source_evidence_id FROM faculty_evidence_links
                WHERE faculty_profile_id=? ORDER BY id DESC LIMIT 1""", (row["id"],)).fetchone()
        if prior:
            continue
        overlaps = discovery.track_overlap(row["id"], profile_id)
        most_relevant = max(overlaps, key=lambda item: len(item["shared_terms"])) if overlaps else None
        discovery.assess(row["id"],
            track_version_id=most_relevant["track_version_id"] if most_relevant and most_relevant["shared_terms"] else None,
            unknowns=["Applicant research tracks are drafts", "Applicant claims are not yet approved",
                      "Current supervision availability is unknown", "Application route/eligibility not assessed"],
            evidence_ids=[evidence["source_evidence_id"]] if evidence else [],
            notes="No fit or readiness score assigned; lexical track overlap is a review hint only.")

    # OpenAlex's institution metadata is noisy for these two Berkeley authors.
    # Record an independent publication list/publisher page for a stored work.
    for legacy_id, title_fragment, source_url in (
        (124, "Learning Sim-to-Real Humanoid Locomotion in 15 Minutes",
         "https://people.eecs.berkeley.edu/~pabbeel/publications.html"),
        (123, "Activation Reward Models for Few-Shot Model Alignment",
         "https://aclanthology.org/2026.findings-acl.1709/"),
    ):
        with connect(settings.database_path) as db:
            faculty = db.execute("SELECT id FROM faculty_profiles WHERE legacy_professor_id=?", (legacy_id,)).fetchone()
            work = db.execute("""SELECT p.id FROM publications p JOIN faculty_profiles f
                ON f.id=p.faculty_profile_id WHERE f.legacy_professor_id=? AND p.title LIKE ?
                ORDER BY p.id LIMIT 1""", (legacy_id, "%" + title_fragment + "%")).fetchone()
            existing = db.execute("""SELECT 1 FROM faculty_evidence_links
                WHERE faculty_profile_id=? AND fact_type='OPENALEX_CORROBORATION'""",
                (faculty["id"],)).fetchone() if faculty else None
        if faculty and work and not existing:
            try:
                evidence_id = oa.corroborate_publication(faculty["id"], work["id"], source_url,
                    reviewer="Slice 2 manual pilot")
                print("Official publication corroboration", legacy_id, evidence_id)
            except Exception as error:
                print("Publication corroboration remains pending", legacy_id, str(error)[:160])

    programme_url = "https://www.cs.stanford.edu/admissions-graduate-application-deadlines"
    if not any(o["canonical_url"] == canonical_url(programme_url) for o in ledger.list_opportunities()):
        programme = next((p for p in ledger.list_programmes() if p["university"] == "Stanford University"
                          and p["programme_name"] == "PhD in Computer Science (Autumn 2027)"), None)
        programme_id = programme["id"] if programme else ledger.create_programme(
            "Stanford University", "PhD in Computer Science (Autumn 2027)",
            department="Computer Science", degree_type="PhD", cycle="2026-27",
            programme_url="https://www.cs.stanford.edu/admissions/phd-admissions",
            admissions_url=programme_url,
            notes="Pilot programme; eligibility, requirements, and funding remain to be reviewed.")
        source_id = discovery.add_source("Stanford University", "PROGRAMME", programme_url,
            target_id=targets.get("Stanford University"), department="Computer Science",
            strategy="STATIC_HTML", notes="2026 application window and deadline")
        evidence_id = discovery.snapshot_source(source_id, manually_verified=True)
        discovery.add_opportunity("PROGRAMME_APPLICATION", "Stanford CS PhD Autumn 2027",
            "Stanford University", evidence_id, opening_status="OPEN", programme_id=programme_id,
            department_lab="Computer Science", deadline_at="2026-12-08",
            application_route="https://gradadmissions.stanford.edu/apply",
            notes="Official page gives Sept 15, 2026 opening and Dec 8, 2026 deadline. "
                  "Deadline timezone, funding, eligibility, requirements, and supervisor openings remain unknown.")
        print("Programme opportunity recorded; no application created")


if __name__ == "__main__":
    main()
