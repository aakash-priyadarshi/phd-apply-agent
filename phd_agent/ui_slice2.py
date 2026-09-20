"""Review screens for Slice 2. Every approval is an explicit operator action."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from phd_agent.config import load_settings
from phd_agent.discovery import Discovery, FACULTY_STATES, SOURCE_TYPES, TARGET_STATES, freshness
from phd_agent.documents import DocumentVault
from phd_agent.ledger import Ledger, OPPORTUNITY_TYPES, REQUIREMENT_CONTEXTS, REQUIREMENT_STATES
from phd_agent.openalex import OpenAlexEnrichment
from phd_agent.profile import ApplicantTruth, CATEGORIES, CLASSIFICATIONS, INGEST_TYPES, TRACK_FIELDS


def _act(fn, success: str):
    try:
        result = fn()
    except Exception as error:
        st.error(str(error))
        return None
    st.success(success)
    st.rerun()
    return result


def _profile_select(truth: ApplicantTruth, key_prefix: str) -> int | None:
    profiles = truth.list_profiles()
    if not profiles:
        with st.form(f"{key_prefix}_create_applicant_profile"):
            name = st.text_input("Applicant name", value="Aakash Priyadarshi")
            if st.form_submit_button("Create applicant profile"):
                _act(lambda: truth.create_profile(name), "Profile created")
        return None
    options = {r["id"]: r["owner_name"] for r in profiles}
    return st.selectbox("Applicant profile", list(options), format_func=lambda i: options[i],
                        key=f"{key_prefix}_profile")


def render_truth(db_path: Path):
    truth = ApplicantTruth(db_path)
    vault = DocumentVault(db_path)
    st.subheader("Applicant truth library")
    st.caption("Candidate claims stay unapproved until reviewed. Approved profile snapshots preserve the exact claim revisions used.")
    profile_id = _profile_select(truth, "truth")
    if profile_id is None:
        return

    eligible = []
    for document in vault.list_documents():
        if document["document_type"] not in INGEST_TYPES:
            continue
        for version in vault.list_versions(document["id"]):
            eligible.append((document, version))
    if eligible:
        options = {version["id"]: f"{document['title']} v{version['version_number']} · {version['approval_state']}"
                   for document, version in eligible}
        selected = st.selectbox("Vault source version", list(options), format_func=lambda i: options[i])
        if st.button("Extract source text from all pages", key="truth_ingest"):
            _act(lambda: truth.ingest_document(selected), "Source text extracted")
    else:
        st.info("Upload a CV, SOP, proposal, statement, publication, patent record, or certificate in the Vault first.")
    from phd_agent.db import connect
    with connect(db_path) as db:
        extractions = [dict(r) for r in db.execute("""SELECT e.id,e.document_version_id,e.extracted_at,e.page_count,
            e.candidate_count,e.model_used,v.original_filename FROM applicant_document_extractions e
            JOIN document_versions v ON v.id=e.document_version_id ORDER BY e.id DESC""")]
    if extractions:
        st.dataframe(extractions, hide_index=True, width="stretch")
        extraction_id = st.selectbox("Extraction for candidate claims", [r["id"] for r in extractions])
        st.caption("Optional OpenAI extraction creates pending claims only. Exact source quotes are checked against the extracted text.")
        if st.button("Extract typed candidate claims", key="truth_candidates"):
            _act(lambda: truth.extract_candidates(profile_id, extraction_id,
                 api_key=os.environ.get("OPENAI_API_KEY", "")), "Candidate extraction complete")

    with st.form("manual_claim"):
        st.markdown("#### Add a manual claim")
        category = st.selectbox("Category", CATEGORIES)
        statement = st.text_area("Exact reusable statement")
        claim_type = st.text_input("Normalized claim type", placeholder="degree, project, skill, publication status…")
        classification = st.selectbox("Classification", CLASSIFICATIONS)
        source_versions = {version["id"]: f"{doc['title']} v{version['version_number']}"
                           for doc, version in eligible}
        document_id = st.selectbox("Supporting Vault version", [0] + list(source_versions),
                                   format_func=lambda i: "No document" if i == 0 else source_versions[i])
        evidence_rows = Ledger(db_path).list_evidence()
        evidence_options = {r["id"]: r["canonical_url"] for r in evidence_rows}
        evidence_id = st.selectbox("Other source evidence", [0] + list(evidence_options),
                                   format_func=lambda i: "No evidence" if i == 0 else f"#{i} {evidence_options[i]}")
        location = st.text_input("Page, section, or source location")
        notes = st.text_input("Review notes")
        if st.form_submit_button("Create pending claim"):
            _act(lambda: truth.create_claim(profile_id, category, statement, claim_type, classification,
                 source_document_version_id=document_id or None, source_evidence_id=evidence_id or None,
                 source_location=location or None, notes=notes), "Pending claim created")

    claims = truth.list_claims(profile_id)
    st.markdown("#### Claims")
    if not claims:
        st.info("No claims yet. Existing research profile text is not treated as verified applicant truth.")
    else:
        st.dataframe([{"Revision": c["id"], "Claim": c["claim_text"], "Category": c["category"],
                       "Class": c["classification"], "Review": c["review_status"],
                       "Evidence": c["source_filename"] or c["source_url"] or "MISSING",
                       "Application": bool(c["approved_for_application"]),
                       "Outreach": bool(c["approved_for_outreach"])} for c in claims],
                     hide_index=True, width="stretch")
        chosen = st.selectbox("Review claim revision", [c["id"] for c in claims])
        claim = next(c for c in claims if c["id"] == chosen)
        st.write(claim["claim_text"])
        st.caption(f"Source location: {claim['source_location'] or 'unknown'} · {claim['notes']}")
        if claim["review_status"] == "PENDING":
            with st.form("claim_review"):
                edited = st.text_area("Edit wording before review", value=claim["claim_text"])
                classified = st.selectbox("Classification", CLASSIFICATIONS,
                                          index=CLASSIFICATIONS.index(claim["classification"]))
                verification = st.selectbox("Fact verification", ["UNVERIFIED", "VERIFIED", "CONFLICT"])
                app_use = st.checkbox("Approved for application")
                outreach_use = st.checkbox("Approved for outreach")
                reviewer = st.text_input("Reviewer")
                notes = st.text_input("Decision notes")
                decision = st.radio("Decision", ["Approve", "Reject"])
                if st.form_submit_button("Save claim review"):
                    def save_review():
                        revision = chosen
                        if edited != claim["claim_text"] or classified != claim["classification"]:
                            revision = truth.revise_claim(claim["claim_id"], claim_text=edited,
                                                          classification=classified)
                        truth.review_claim(revision, decision == "Approve", reviewer,
                                           for_application=app_use, for_outreach=outreach_use,
                                           verification_state=verification, notes=notes)
                    _act(save_review, "Claim reviewed")
        else:
            if st.button("Clone as a new pending revision", key=f"claim_clone_{chosen}"):
                _act(lambda: truth.revise_claim(claim["claim_id"]), "New revision created")

    st.markdown("#### Profile versions")
    versions = truth.list_profile_versions(profile_id)
    if versions:
        st.dataframe([{"ID": v["id"], "Version": v["version_number"], "State": v["approval_state"],
                       "Created": v["created_at"], "SHA-256": v["claim_snapshot_sha256"]} for v in versions],
                     width="stretch", hide_index=True)
        selected_version = st.selectbox("Inspect frozen snapshot", [v["id"] for v in versions])
        st.json(truth.profile_snapshot(selected_version))
    with st.form("profile_snapshot"):
        selected_claims = st.multiselect("Claim revisions to freeze", [c["id"] for c in claims],
                                         format_func=lambda i: next(c["claim_text"] for c in claims if c["id"] == i))
        approve = st.checkbox("Approve this snapshot")
        reviewer = st.text_input("Snapshot reviewer")
        notes = st.text_input("Version notes")
        if st.form_submit_button("Create profile version"):
            _act(lambda: truth.create_profile_version(profile_id, selected_claims, approve=approve,
                 reviewer=reviewer, notes=notes), "Profile version created")


def render_tracks(db_path: Path):
    truth = ApplicantTruth(db_path)
    st.subheader("Research directions")
    st.caption("Tracks are proposed PhD directions. Supporting claims are explicit; a track never establishes experience on its own.")
    profile_id = _profile_select(truth, "tracks")
    if profile_id is None:
        return
    claims = [c for c in truth.list_claims(profile_id) if c["review_status"] == "APPROVED"]
    claim_labels = {c["id"]: f"#{c['id']} {c['claim_text']}" for c in claims}
    tracks = truth.list_tracks(profile_id)
    if tracks:
        st.dataframe([{"ID": t["id"], "Title": t["title"], "Priority": t["priority"],
                       "Status": t["status"], "Version": t["version_number"],
                       "Review": t["approval_state"], "Problem": t["research_problem"]} for t in tracks],
                     width="stretch", hide_index=True)
    with st.form("create_track"):
        st.markdown("#### New research direction")
        title = st.text_input("Title")
        priority = st.slider("Priority (1 is highest)", 1, 5, 3)
        fields = {field: st.text_area(field.replace("_", " ").title()) for field in TRACK_FIELDS}
        supporting = st.multiselect("Approved supporting claims", list(claim_labels),
                                    format_func=lambda i: claim_labels[i])
        if st.form_submit_button("Create draft track"):
            _act(lambda: truth.create_track(profile_id, title, priority=priority,
                 supporting_claim_revision_ids=supporting, **fields), "Draft track created")
    if not tracks:
        return
    track_id = st.selectbox("Open direction", [t["id"] for t in tracks],
                            format_func=lambda i: next(t["title"] for t in tracks if t["id"] == i))
    track = next(t for t in tracks if t["id"] == track_id)
    history = truth.track_history(track_id)
    st.dataframe([{"Version": v["version_number"], "Review": v["approval_state"],
                   "Problem": v["research_problem"], "Created": v["created_at"]} for v in history],
                 width="stretch", hide_index=True)
    latest = history[0]
    unsupported = [i for i in json.loads(latest["supporting_claim_revision_ids_json"]) if i not in claim_labels]
    if unsupported:
        st.warning(f"Supporting claims no longer approved: {unsupported}")
    if not json.loads(latest["supporting_claim_revision_ids_json"]):
        st.info("No approved applicant claims support experience related to this direction yet.")
    with st.form(f"edit_track_{track_id}"):
        new_title = st.text_input("Edit title", value=track["title"])
        new_priority = st.slider("Edit priority", 1, 5, track["priority"])
        archive = st.checkbox("Archive track", value=track["status"] == "ARCHIVED")
        new_fields = {field: st.text_area(field.replace("_", " ").title(), value=latest[field] or "")
                      for field in TRACK_FIELDS}
        selected_claims = st.multiselect("Supporting claims", list(claim_labels),
                                         default=[i for i in json.loads(latest["supporting_claim_revision_ids_json"])
                                                  if i in claim_labels], format_func=lambda i: claim_labels[i])
        if st.form_submit_button("Save new draft version"):
            def save_track():
                truth.update_track(track_id, title=new_title, priority=new_priority, archive=archive)
                truth.revise_track(track_id, supporting_claim_revision_ids=selected_claims, **new_fields)
            _act(save_track, "New draft version created")
    if latest["approval_state"] == "DRAFT":
        reviewer = st.text_input("Track reviewer", key=f"track_reviewer_{track_id}")
        if st.button("Approve latest version", key=f"track_approve_{track_id}"):
            _act(lambda: truth.approve_track(latest["id"], reviewer), "Research direction approved")


def render_discovery(db_path: Path):
    discovery = Discovery(db_path)
    ledger = Ledger(db_path)
    enrichment = OpenAlexEnrichment(db_path)
    st.subheader("Discovery and verification")
    st.caption("Historical professor scores are not identity checks. Current institutional evidence and unknowns are shown separately.")
    targets_tab, sources_tab, opportunities_tab, faculty_tab = st.tabs(["Targets", "Source catalogue", "Opportunities", "Professors"])
    with targets_tab:
        targets = discovery.list_targets()
        st.dataframe([{"ID": t["id"], "Institution": t["name"], "State": t["state"],
                       "Departments": t["departments"], "Last verified": t["last_verified_at"]}
                      for t in targets], width="stretch", hide_index=True)
        if not targets:
            if st.button("Import historical target CSV as considering"):
                _act(lambda: discovery.import_targets(load_settings().targets_path), "Historical targets imported")
        with st.form("new_target"):
            name = st.text_input("Institution")
            departments = st.text_input("Departments")
            state = st.selectbox("Target state", TARGET_STATES)
            if st.form_submit_button("Add target"):
                _act(lambda: discovery.add_target(name, state=state, departments=departments), "Target added")
        if targets:
            target_id = st.selectbox("Update target", [t["id"] for t in targets],
                                     format_func=lambda i: next(t["name"] for t in targets if t["id"] == i))
            new_state = st.selectbox("New state", TARGET_STATES, key="target_new_state")
            if st.button("Save target state"):
                _act(lambda: discovery.set_target_state(target_id, new_state), "Target updated")

    with sources_tab:
        sources = discovery.list_sources()
        st.dataframe([{"ID": s["id"], "Institution": s["university"], "Department": s["department"],
                       "Type": s["source_type"], "URL": s["canonical_url"],
                       "Strategy": s["discovery_strategy"], "Last verified": s["last_verified_at"]}
                      for s in sources], width="stretch", hide_index=True)
        target_map = {t["id"]: t["name"] for t in discovery.list_targets()}
        with st.form("new_source"):
            target_id = st.selectbox("Target institution", [0] + list(target_map),
                                     format_func=lambda i: "Other" if i == 0 else target_map[i])
            university = st.text_input("Institution name")
            department = st.text_input("Department")
            source_type = st.selectbox("Source type", SOURCE_TYPES)
            url = st.text_input("Canonical official URL")
            strategy = st.selectbox("Strategy", ["MANUAL", "STATIC_HTML"])
            if st.form_submit_button("Add catalogue source"):
                _act(lambda: discovery.add_source(university or target_map.get(target_id, ""), source_type, url,
                     target_id=target_id or None, department=department or None, strategy=strategy), "Source added")
        if sources:
            source_id = st.selectbox("Snapshot source", [s["id"] for s in sources],
                                     format_func=lambda i: next(s["canonical_url"] for s in sources if s["id"] == i))
            excerpt = st.text_area("Reviewed excerpt (leave empty for static HTML fetch)")
            verified = st.checkbox("I reviewed this source and excerpt")
            if st.button("Save append-only evidence snapshot"):
                _act(lambda: discovery.snapshot_source(source_id, excerpt=excerpt or None,
                     manually_verified=verified), "Evidence snapshot saved")
            selected_source = next(s for s in sources if s["id"] == source_id)
            with st.form("edit_catalogue_source"):
                source_strategy = st.selectbox("Discovery strategy", ["MANUAL", "STATIC_HTML"],
                    index=["MANUAL", "STATIC_HTML"].index(selected_source["discovery_strategy"]))
                source_enabled = st.checkbox("Enabled", value=bool(selected_source["enabled"]))
                source_notes = st.text_input("Source notes", value=selected_source["notes"])
                if st.form_submit_button("Update catalogue entry"):
                    _act(lambda: discovery.update_source(source_id, strategy=source_strategy,
                         enabled=source_enabled, notes=source_notes), "Catalogue entry updated")
        evidence = ledger.list_evidence()
        if evidence:
            st.dataframe([{"ID": e["id"], "Type": e["source_type"], "URL": e["canonical_url"],
                           "Retrieved": e["retrieved_at"], "Review": e["verification_state"]}
                          for e in evidence[:50]], width="stretch", hide_index=True)
        faculty_sources = {s["id"]: s for s in sources if s["source_type"] in {"FACULTY", "LAB"}}
        faculty_evidence = {e["id"]: e for e in evidence if e["source_type"] in {"FACULTY", "LAB"}}
        if faculty_sources and faculty_evidence:
            with st.form("faculty_source_candidate"):
                st.markdown("#### Add a faculty candidate from a source")
                source_id = st.selectbox("Faculty source", list(faculty_sources),
                                         format_func=lambda i: faculty_sources[i]["canonical_url"])
                candidate_name = st.text_input("Candidate name")
                candidate_department = st.text_input("Candidate department")
                candidate_url = st.text_input("Candidate official profile URL")
                matching = [e["id"] for e in faculty_evidence.values()
                            if e["canonical_url"] == faculty_sources[source_id]["canonical_url"]]
                evidence_id = st.selectbox("Source snapshot", matching) if matching else 0
                if st.form_submit_button("Add candidate"):
                    if not evidence_id:
                        st.error("Snapshot this source first")
                    else:
                        _act(lambda: discovery.add_faculty_candidate(source_id, candidate_name, evidence_id,
                             profile_url=candidate_url or None,
                             department=candidate_department or None), "Candidate queued")
        candidates = discovery.list_faculty_candidates()
        if candidates:
            st.dataframe(candidates, width="stretch", hide_index=True)
            pending = [c for c in candidates if c["review_state"] == "NEW"]
            if pending:
                selected_candidate = st.selectbox("Review faculty candidate", [c["id"] for c in pending],
                    format_func=lambda i: next(f"#{i} {c['name']} · {c['institution']}" for c in pending if c["id"] == i))
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Create unverified faculty profile"):
                        _act(lambda: discovery.review_faculty_candidate(selected_candidate), "Faculty record created")
                with col_b:
                    if st.button("Dismiss candidate"):
                        _act(lambda: discovery.review_faculty_candidate(selected_candidate, dismiss=True), "Candidate dismissed")

    with opportunities_tab:
        opportunities = ledger.list_opportunities()
        st.dataframe([{"ID": o["id"], "Route": o["opportunity_type"], "Title": o["title"],
                       "Institution": o["institution"], "Deadline": o["deadline_at"],
                       "Funding": o["funding_text"], "Status": o["opening_status"],
                       "Verification": o["verification_state"],
                       "Freshness": freshness(o["last_checked_at"], "OPPORTUNITY_OPENING")}
                      for o in opportunities], width="stretch", hide_index=True)
        source_evidence = {e["id"]: e for e in ledger.list_evidence()
                           if e["source_type"] in {"PROGRAMME", "OPPORTUNITY", "FACULTY", "LAB", "FUNDING"}}
        with st.form("new_discovered_opportunity"):
            route = st.selectbox("Route", OPPORTUNITY_TYPES)
            title = st.text_input("Opportunity title")
            institution = st.text_input("Institution", key="opp_institution")
            source_id = st.selectbox("Institutional evidence", [0] + list(source_evidence),
                                     format_func=lambda i: "Select source" if i == 0 else f"#{i} {source_evidence[i]['canonical_url']}")
            status = st.selectbox("Opening state", ["UNKNOWN", "OPEN", "CLOSED"])
            department = st.text_input("Department / lab")
            supervisor = st.text_input("Named supervisor (only if explicit)")
            area = st.text_input("Research area")
            funding = st.text_input("Funding statement")
            eligibility = st.text_input("Eligibility statement")
            deadline = st.text_input("Deadline (ISO date/time)")
            timezone = st.text_input("Deadline timezone")
            application_route = st.text_input("Application route URL/instructions")
            contact_policy = st.text_input("Contact policy")
            if st.form_submit_button("Create sourced opportunity"):
                if not source_id:
                    st.error("Select evidence")
                else:
                    _act(lambda: discovery.add_opportunity(route, title, institution, source_id,
                         opening_status=status, department_lab=department or None,
                         supervisor_name=supervisor or None, research_area=area or None,
                         funding_text=funding or None, eligibility_text=eligibility or None,
                         deadline_at=deadline or None, deadline_timezone=timezone or None,
                         application_route=application_route or None,
                         contact_policy=contact_policy or None), "Opportunity created")
        verified = [o for o in opportunities if o["verification_state"] == "VERIFIED"]
        if verified:
            selected = st.selectbox("Create application from reviewed opportunity", [o["id"] for o in verified],
                                    format_func=lambda i: next(o["title"] for o in verified if o["id"] == i))
            cycle = st.text_input("Application cycle", value="2026-27")
            if st.button("Create ledger application"):
                _act(lambda: discovery.create_application_from_opportunity(selected, cycle), "Application created")
        applications = ledger.list_applications()
        if applications:
            st.markdown("#### Link discoveries to an existing application")
            app_id = st.selectbox("Application", [a["id"] for a in applications],
                                  format_func=lambda i: next(a["application_name"] for a in applications if a["id"] == i),
                                  key="discovery_app")
            programme_ids = {p["id"]: p["programme_name"] for p in ledger.list_programmes()}
            opportunity_ids = {o["id"]: o["title"] for o in opportunities}
            programme_id = st.selectbox("Discovered programme", [0] + list(programme_ids),
                                         format_func=lambda i: "None" if i == 0 else programme_ids[i])
            opportunity_id = st.selectbox("Discovered opportunity", [0] + list(opportunity_ids),
                                           format_func=lambda i: "None" if i == 0 else opportunity_ids[i])
            if st.button("Link reviewed discovery"):
                _act(lambda: discovery.link_to_application(app_id, programme_id=programme_id or None,
                     opportunity_id=opportunity_id or None), "Link checked; conflicts become review tasks")
            if source_evidence:
                with st.form("discovered_deadline"):
                    source_id = st.selectbox("Deadline evidence", list(source_evidence),
                                             format_func=lambda i: f"#{i} {source_evidence[i]['canonical_url']}")
                    due_at = st.text_input("Discovered deadline (ISO date/time)")
                    timezone_name = st.text_input("Timezone if stated")
                    if st.form_submit_button("Add or review deadline"):
                        _act(lambda: discovery.propose_deadline(app_id, due_at, source_id,
                             timezone_name=timezone_name or None), "Deadline checked")
                with st.form("discovered_requirement"):
                    source_id = st.selectbox("Requirement evidence", list(source_evidence),
                                             format_func=lambda i: f"#{i} {source_evidence[i]['canonical_url']}")
                    context = st.selectbox("Requirement context", REQUIREMENT_CONTEXTS)
                    label = st.text_input("Original requirement wording")
                    state = st.selectbox("Requirement state", REQUIREMENT_STATES)
                    if st.form_submit_button("Add or review requirement"):
                        _act(lambda: discovery.propose_requirement(app_id, context, label, state, source_id),
                             "Requirement checked")

    with faculty_tab:
        faculty = discovery.list_faculty()
        pending_duplicates = discovery.list_duplicate_reviews(pending_only=True)
        if pending_duplicates:
            with st.expander(f"{len(pending_duplicates)} possible duplicate pairs awaiting review"):
                st.dataframe([{"ID": r["id"], "First": f"{r['name_a']} · {r['institution_a']}",
                               "Second": f"{r['name_b']} · {r['institution_b']}", "Signal": r["reason"]}
                              for r in pending_duplicates[:100]], width="stretch", hide_index=True)
                review_id = st.selectbox("Review possible duplicate", [r["id"] for r in pending_duplicates],
                                         key="duplicate_review")
                decision = st.selectbox("Identity decision", ["DISTINCT", "DUPLICATE"])
                st.caption("This records a decision only. Historical rows are never merged or deleted automatically.")
                if st.button("Save duplicate decision"):
                    _act(lambda: discovery.resolve_duplicate_review(review_id, decision), "Identity decision recorded")
        state_filter = st.selectbox("Verification filter", ["ALL", *FACULTY_STATES])
        search = st.text_input("Search professor or institution")
        displayed = [f for f in faculty if (state_filter == "ALL" or f["verification_state"] == state_filter)
                     and (not search or search.casefold() in (f["name"] + " " + f["institution"]).casefold())]
        st.dataframe([{"ID": f["id"], "Name": f["name"], "Institution": f["institution"],
                       "Department": f["department"], "Verification": f["verification_state"],
                       "Affiliation": f["affiliation_state"], "Email": f["email_state"],
                       "OpenAlex": f["openalex_resolution_state"], "Checked": f["last_checked_at"]}
                      for f in displayed[:100]], width="stretch", hide_index=True)
        with st.form("new_faculty"):
            st.markdown("#### Add faculty candidate")
            name = st.text_input("Name")
            institution = st.text_input("Institution", key="faculty_institution")
            department = st.text_input("Department", key="faculty_department")
            url = st.text_input("Official profile URL")
            if st.form_submit_button("Create unverified faculty record"):
                _act(lambda: discovery.create_faculty(name, institution, department=department or None,
                     profile_url=url or None), "Faculty record created for review")
        if not faculty:
            return
        selected = st.selectbox("Professor detail", [f["id"] for f in faculty],
                                format_func=lambda i: next(f"#{i} {f['name']} · {f['institution']}" for f in faculty if f["id"] == i))
        detail = discovery.faculty_detail(selected)
        st.markdown(f"#### {detail['name']}")
        st.write({k: detail[k] for k in ("institution", "department", "official_title", "lab", "official_profile_url",
                                       "email", "email_state", "research_topics", "affiliation_state",
                                       "supervision_state", "verification_state", "openalex_author_id")})
        if detail["legacy_professor_id"]:
            st.caption(f"Historical professor row #{detail['legacy_professor_id']} retained; old fit score is not verification.")
        if detail["evidence"]:
            st.dataframe([{"Fact": e["fact_type"], "URL": e["canonical_url"],
                           "Excerpt": e["relevant_excerpt"], "Review": e["verification_state"],
                           "Freshness": e["freshness"]} for e in detail["evidence"]], width="stretch", hide_index=True)
        else:
            st.warning("No current source evidence for this professor.")
        if detail["publications"]:
            st.dataframe([{"OpenAlex": p["openalex_id"], "Title": p["title"], "Year": p["year"],
                           "DOI": p["doi"], "Venue": p["venue"]} for p in detail["publications"]],
                         width="stretch", hide_index=True)
        if detail["history"]:
            st.dataframe(detail["history"], width="stretch", hide_index=True)
        if detail["assessments"]:
            st.markdown("##### Separate fit and readiness assessments")
            st.dataframe([{"Research Fit": a["research_fit"],
                           "Application Readiness": a["application_readiness"],
                           "Unknowns": ", ".join(json.loads(a["unknowns_json"])),
                           "Notes": a["notes"], "Assessed": a["assessed_at"]}
                          for a in detail["assessments"]], width="stretch", hide_index=True)
        profiles = ApplicantTruth(db_path).list_profiles()
        if profiles:
            overlap = discovery.track_overlap(selected, profiles[0]["id"])
            if overlap:
                st.markdown("##### Research direction overlap (lexical hints only)")
                st.dataframe(overlap, width="stretch", hide_index=True)
            if detail["publications"]:
                selected_work = st.selectbox("Inspect publication overlap", [p["id"] for p in detail["publications"]],
                    format_func=lambda i: next(p["title"] for p in detail["publications"] if p["id"] == i))
                st.dataframe(discovery.publication_relevance(selected_work, profiles[0]["id"]),
                             width="stretch", hide_index=True)
        st.caption("Unknown supervision remains UNKNOWN unless an explicit current source states it.")
        evidence_ids = {e["id"]: e for e in ledger.list_evidence()}
        with st.form(f"verify_faculty_{selected}"):
            state = st.selectbox("Verification state", FACULTY_STATES,
                                 index=FACULTY_STATES.index(detail["verification_state"]))
            reviewer = st.text_input("Reviewer")
            reason = st.text_input("Review reason")
            official = st.text_input("Official profile URL", value=detail["official_profile_url"] or "")
            department = st.text_input("Reviewed department", value=detail["department"] or "")
            title = st.text_input("Official title", value=detail["official_title"] or "")
            topics = st.text_input("Reviewed research topics", value=detail["research_topics"] or "")
            email = st.text_input("Reviewed email", value=detail["email"] or "")
            email_state = st.selectbox("Email state", ["UNKNOWN", "VERIFIED", "UNVERIFIED", "CONFLICT"],
                                       index=["UNKNOWN", "VERIFIED", "UNVERIFIED", "CONFLICT"].index(detail["email_state"]))
            affiliation = st.selectbox("Affiliation state", ["UNKNOWN", "CURRENT", "MOVED", "INACTIVE", "CONFLICT"],
                                       index=["UNKNOWN", "CURRENT", "MOVED", "INACTIVE", "CONFLICT"].index(detail["affiliation_state"]))
            supervision = st.selectbox("Supervision/opening state", ["UNKNOWN", "OPEN", "CLOSED"],
                                       index=["UNKNOWN", "OPEN", "CLOSED"].index(detail["supervision_state"]))
            chosen_evidence = {fact: st.selectbox(f"{fact.title()} evidence", [0] + list(evidence_ids),
                format_func=lambda i: "Unknown" if i == 0 else f"#{i} {evidence_ids[i]['canonical_url']}",
                key=f"faculty_{selected}_{fact}") for fact in ("IDENTITY", "AFFILIATION", "TOPICS", "EMAIL", "SUPERVISION")}
            if st.form_submit_button("Save reviewed verification event"):
                _act(lambda: discovery.verify_faculty(selected, state,
                     evidence_by_fact={k: v for k, v in chosen_evidence.items() if v},
                     reviewer=reviewer, reason=reason,
                     updates={"official_profile_url": official or None, "department": department or None,
                              "official_title": title or None, "research_topics": topics or None,
                              "email": email or None, "email_state": email_state,
                              "affiliation_state": affiliation, "supervision_state": supervision}),
                     "Faculty verification saved")
        if detail["verification_state"] in {"VERIFIED", "PARTIALLY_VERIFIED"}:
            if st.button("Search OpenAlex authors", key=f"oa_search_{selected}"):
                try:
                    st.session_state[f"oa_candidates_{selected}"] = enrichment.search_authors(selected)
                except Exception as error:
                    st.error(str(error))
            candidates = st.session_state.get(f"oa_candidates_{selected}", [])
            if candidates:
                st.dataframe(candidates, width="stretch", hide_index=True)
                with st.form(f"oa_resolve_{selected}"):
                    author_id = st.selectbox("Reviewed OpenAlex author", [c["id"] for c in candidates],
                                             format_func=lambda i: next(f"{c['display_name']} · {i}" for c in candidates if c["id"] == i))
                    reviewer = st.text_input("Author reviewer")
                    reason = st.text_input("Identity rationale")
                    if st.form_submit_button("Resolve author identity"):
                        _act(lambda: enrichment.resolve_author(selected, author_id, reviewer=reviewer,
                             reason=reason), "OpenAlex author resolved")
            if detail["openalex_resolution_state"] == "RESOLVED":
                if st.button("Fetch recent OpenAlex works", key=f"oa_works_{selected}"):
                    _act(lambda: enrichment.enrich_works(selected), "Publications enriched")
                if detail["publications"]:
                    with st.form(f"oa_corroborate_{selected}"):
                        publication_id = st.selectbox("Publication to corroborate",
                            [p["id"] for p in detail["publications"]],
                            format_func=lambda i: next(p["title"] for p in detail["publications"] if p["id"] == i))
                        source_url = st.text_input("Independent publication or official lab URL")
                        reviewer = st.text_input("Corroboration reviewer")
                        if st.form_submit_button("Record publication corroboration"):
                            _act(lambda: enrichment.corroborate_publication(selected, publication_id,
                                 source_url, reviewer=reviewer), "Independent publication evidence linked")
        applications = ledger.list_applications()
        if applications:
            app_id = st.selectbox("Link potential supervisor to application", [a["id"] for a in applications],
                                  format_func=lambda i: next(a["application_name"] for a in applications if a["id"] == i))
            evidence_id = st.selectbox("Link evidence", [0] + list(evidence_ids),
                                       format_func=lambda i: "No source" if i == 0 else f"#{i} {evidence_ids[i]['canonical_url']}")
            if st.button("Link supervisor"):
                _act(lambda: discovery.link_to_application(app_id, faculty_id=selected,
                     evidence_id=evidence_id or None), "Supervisor linked")
