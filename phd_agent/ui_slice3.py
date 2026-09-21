"""Operator screens for Slice 3 matching, materials, package review, and preflight."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from phd_agent.db import connect
from phd_agent.documents import DocumentVault
from phd_agent.matching import MatchEngine
from phd_agent.materials import MaterialStudio
from phd_agent.packages import PackageBuilder
from phd_agent.profile import ApplicantTruth
from phd_agent.ui_slice2 import _act


def _rows(path: Path, sql: str, args=()) -> list[dict]:
    with connect(path) as db:
        return [dict(r) for r in db.execute(sql, args)]


def _approved_context(path: Path, key: str):
    profiles = _rows(path, "SELECT * FROM applicant_profiles ORDER BY id")
    if not profiles:
        st.info("Create an applicant profile and review source claims first.")
        return None
    owner = st.selectbox("Applicant", [p["id"] for p in profiles],
                         format_func=lambda i: next(p["owner_name"] for p in profiles if p["id"] == i), key=key+"_owner")
    snapshots = _rows(path, "SELECT * FROM profile_versions WHERE profile_id=? AND approval_state='APPROVED' ORDER BY id DESC", (owner,))
    tracks = _rows(path, """SELECT v.*,t.title FROM research_track_versions v JOIN research_tracks t ON t.id=v.track_id
        WHERE t.profile_id=? AND t.status='ACTIVE' AND v.approval_state='APPROVED'
        AND v.version_number=(SELECT MAX(x.version_number) FROM research_track_versions x
                              WHERE x.track_id=t.id AND x.approval_state='APPROVED')
        ORDER BY v.id DESC""", (owner,))
    if not snapshots or not tracks:
        st.warning("Approve an application claim set/profile snapshot and a research direction in Applicant Truth and Research Directions.")
        return None
    profile_version_id = st.selectbox("Approved profile version", [p["id"] for p in snapshots],
                                      format_func=lambda i: f"v{next(p['version_number'] for p in snapshots if p['id']==i)} · #{i}", key=key+"_profile")
    track_version_id = st.selectbox("Approved research direction", [t["id"] for t in tracks],
                                    format_func=lambda i: next(f"{t['title']} v{t['version_number']}" for t in tracks if t["id"]==i), key=key+"_track")
    return owner, profile_version_id, track_version_id


def render_match_review(path: Path):
    st.subheader("Match review")
    st.caption("Research Fit uses verified evidence and configurable component weights. Application Readiness is shown separately as known, blocked, or unknown facts.")
    context = _approved_context(path, "match")
    if not context:
        return
    _, profile_version_id, track_version_id = context
    faculty = _rows(path, "SELECT id,name,institution,verification_state FROM faculty_profiles WHERE verification_state IN ('VERIFIED','PARTIALLY_VERIFIED') ORDER BY name")
    apps = _rows(path, "SELECT id,cycle FROM applications ORDER BY id DESC")
    if not faculty:
        st.info("Verify a shortlisted professor in Discovery first. Historical unverified records are not bulk processed.")
        return
    faculty_id = st.selectbox("Shortlisted professor", [f["id"] for f in faculty],
                              format_func=lambda i: next(f"{f['name']} — {f['institution']} ({f['verification_state']})" for f in faculty if f["id"]==i))
    app_id = st.selectbox("Application (optional)", [0] + [a["id"] for a in apps],
                          format_func=lambda i: "Professor only" if i == 0 else f"Application #{i}")
    if st.button("Assess shortlisted match"):
        _act(lambda: MatchEngine(path).assess(faculty_id, profile_version_id, track_version_id, app_id or None), "Match assessment saved")
    assessments = _rows(path, "SELECT * FROM match_assessments WHERE faculty_profile_id=? ORDER BY id DESC LIMIT 20", (faculty_id,))
    if not assessments:
        return
    chosen = st.selectbox("Assessment history", [a["id"] for a in assessments])
    a = next(x for x in assessments if x["id"] == chosen)
    st.metric("Research Fit (0–10; known components only)", "Unknown" if a["research_fit"] is None else f"{a['research_fit']:.2f}")
    st.write("Research Fit components")
    st.json(json.loads(a["research_fit_components_json"]))
    st.write("Application Readiness")
    st.json(json.loads(a["application_readiness_components_json"]))
    st.write("Unknowns", json.loads(a["unknowns_json"]))
    st.caption(f"Evidence IDs: {a['evidence_ids_json']} · Assessed {a['assessed_at']}")
    pubs = _rows(path, "SELECT id,title,year,source_evidence_id FROM publications WHERE faculty_profile_id=? ORDER BY year DESC LIMIT 8", (faculty_id,))
    if pubs:
        st.write("Stored publications")
        st.dataframe(pubs, hide_index=True)
    reviews = _rows(path, "SELECT * FROM match_reviews WHERE assessment_id=? ORDER BY version_number DESC", (chosen,))
    if reviews:
        st.write("Review versions")
        st.dataframe(reviews, hide_index=True)
    with st.form("match_annotation"):
        reviewer = st.text_input("Reviewer", key="match_reviewer")
        annotation = st.text_area("Annotation or risk")
        override = st.text_input("Optional Research Fit override (0–10)")
        if st.form_submit_button("Save review version"):
            _act(lambda: MatchEngine(path).review(chosen, reviewer, annotation,
                 {"research_fit": float(override)} if override.strip() else {}), "Match review saved")


def render_materials(path: Path):
    st.subheader("Application materials")
    st.caption("Drafts use approved claims and research directions. PDF rendering is deterministic; editable source and provenance remain in the database.")
    context = _approved_context(path, "materials")
    if not context:
        return
    owner, profile_version_id, track_version_id = context
    studio = MaterialStudio(path)
    vault = DocumentVault(path)
    claims = _rows(path, """SELECT cr.id,cr.claim_text,c.category FROM profile_version_claims pvc
        JOIN claim_revisions cr ON cr.id=pvc.claim_revision_id JOIN claims c ON c.id=cr.claim_id
        WHERE pvc.profile_version_id=? AND cr.review_status='APPROVED' AND cr.approved_for_application=1 ORDER BY cr.id""", (profile_version_id,))
    if not claims:
        st.warning("Profile snapshot contains no claims approved for application use.")
        return
    claim_labels = {c["id"]: f"#{c['id']} {c['claim_text']}" for c in claims}
    apps = _rows(path, "SELECT id,cycle FROM applications ORDER BY id DESC")
    app_id = st.selectbox("Application context", [0] + [a["id"] for a in apps],
                          format_func=lambda i: "Professor preparation only" if i == 0 else f"Application #{i}")
    faculty = _rows(path, "SELECT id,name,institution FROM faculty_profiles WHERE verification_state='VERIFIED' ORDER BY name")
    faculty_id = st.selectbox("Verified professor (optional)", [0] + [f["id"] for f in faculty],
                              format_func=lambda i: "None" if i == 0 else next(f"{f['name']} — {f['institution']}" for f in faculty if f["id"]==i))
    requirements = _rows(path, "SELECT * FROM requirements WHERE application_id=? ORDER BY id", (app_id,)) if app_id else []

    with st.expander("Master CV and tailored variant", expanded=True):
        defaults = []
        mapping = {"EDUCATION":"Education","RESEARCH_PROJECT":"Research Experience","INDUSTRY_RESEARCH":"Industry and Research Engineering",
                   "PUBLICATION":"Publications","PATENT":"Patents","TEACHING":"Teaching","SKILL_METHOD":"Technical Skills"}
        for section in dict.fromkeys(mapping.get(c["category"], "Profile") for c in claims):
            defaults.append({"name":section,"bullets":[{"text":c["claim_text"],"claim_revision_ids":[c["id"]]}
                for c in claims if mapping.get(c["category"], "Profile")==section]})
        st.caption("Edit structured sections below. Each bullet must name approved claim revision IDs. The uploaded source CV is untouched.")
        source = st.text_area("Master CV sections (JSON)", value=json.dumps(defaults, indent=2, ensure_ascii=False), height=210)
        if st.button("Create draft Master CV"):
            _act(lambda: studio.create_master_cv(profile_version_id, json.loads(source)), "Master CV draft created")
        masters = _rows(path, "SELECT * FROM master_cv_versions WHERE profile_id=? ORDER BY id DESC", (owner,))
        if masters:
            st.dataframe([{"ID":m["id"],"Version":m["version_number"],"Review":m["approval_state"],"Profile":m["profile_version_id"]} for m in masters], hide_index=True)
            master_id = st.selectbox("Master CV version", [m["id"] for m in masters])
            master = next(m for m in masters if m["id"] == master_id)
            if master["approval_state"] == "DRAFT":
                reviewer = st.text_input("Master CV reviewer")
                if st.button("Approve Master CV"):
                    _act(lambda: studio.review_master_cv(master_id, reviewer, True), "Master CV approved")
            if master["approval_state"] == "APPROVED" and master["profile_version_id"] == profile_version_id:
                names = [s["name"] for s in json.loads(master["sections_json"])]
                chosen = st.multiselect("Include approved sections in this order", names, default=names)
                if st.button("Create tailored CV draft"):
                    _act(lambda: studio.tailor_cv(master_id, track_version_id, application_id=app_id or None,
                         faculty_id=faculty_id or None, selected_sections=chosen), "Tailored CV draft created")

    with st.expander("Approved story modules and statements"):
        with st.form("story_module_create"):
            key = st.text_input("Story module key", placeholder="WHY_PHD")
            text = st.text_area("Reviewed source-bound narrative")
            ids = st.multiselect("Supporting approved claims", list(claim_labels), format_func=lambda i: claim_labels[i])
            if st.form_submit_button("Create module draft"):
                _act(lambda: studio.create_module(profile_version_id, key, text, ids), "Story module draft created")
        modules = _rows(path, "SELECT * FROM story_module_versions WHERE profile_id=? ORDER BY id DESC", (owner,))
        if modules:
            st.dataframe([{"ID":m["id"],"Key":m["module_key"],"Version":m["version_number"],"Review":m["approval_state"]} for m in modules], hide_index=True)
            module_id = st.selectbox("Review module", [m["id"] for m in modules])
            module = next(m for m in modules if m["id"] == module_id)
            st.write(module["content"])
            st.caption("Claims: " + module["claim_revision_ids_json"])
            if module["approval_state"] == "DRAFT":
                reviewer = st.text_input("Module reviewer")
                if st.button("Approve story module"):
                    _act(lambda: studio.review_module(module_id, reviewer, True), "Story module approved")
        approved_modules = [m for m in modules if m["approval_state"] == "APPROVED"] if modules else []
        statement_reqs = [r for r in requirements if r["normalized_document_type"] in {"SOP","PERSONAL_STATEMENT","RESEARCH_STATEMENT"}]
        if app_id and statement_reqs and approved_modules:
            requirement_id = st.selectbox("Sourced statement requirement", [r["id"] for r in statement_reqs],
                format_func=lambda i: next(f"#{r['id']} {r['normalized_document_type']}: {r['original_label']}" for r in statement_reqs if r["id"]==i))
            requirement = next(r for r in statement_reqs if r["id"] == requirement_id)
            chosen_modules = st.multiselect("Approved story modules, in narrative order", [m["id"] for m in approved_modules],
                format_func=lambda i: next(m["module_key"] for m in approved_modules if m["id"]==i))
            if st.button("Create statement draft"):
                _act(lambda: studio.create_statement(requirement["normalized_document_type"], profile_version_id,
                    track_version_id, app_id, requirement_id, chosen_modules), "Statement draft created")

    with st.expander("Track-derived proposal and cover letter"):
        proposal_reqs = [r for r in requirements if r["normalized_document_type"] == "RESEARCH_PROPOSAL"]
        proposal_req = st.selectbox("Proposal requirement", [0] + [r["id"] for r in proposal_reqs],
            format_func=lambda i: "Professor preparation only / no formal proposal requirement" if i == 0 else f"Requirement #{i}")
        pubs = _rows(path, "SELECT id,title,year FROM publications WHERE faculty_profile_id=? ORDER BY year DESC", (faculty_id,)) if faculty_id else []
        pub_ids = st.multiselect("Verified stored publications to cite", [p["id"] for p in pubs],
            format_func=lambda i: next(f"{p['title']} ({p['year']})" for p in pubs if p["id"]==i))
        format_name = st.selectbox("Proposal format", ["SHORT_STATEMENT","CONCEPT_NOTE","ONE_PAGE","TWO_PAGE","THREE_PAGE","FULL","INSTITUTION_TEMPLATE"])
        if st.button("Create track-derived proposal draft"):
            _act(lambda: studio.create_proposal(profile_version_id, track_version_id,
                application_id=app_id or None, faculty_id=faculty_id or None, requirement_id=proposal_req or None,
                publication_ids=pub_ids, format_name=format_name), "Proposal draft created")
        letters = [r for r in requirements if r["normalized_document_type"] == "COVER_LETTER"]
        if app_id:
            letter_id = st.selectbox("Cover letter requirement", [0] + [r["id"] for r in letters],
                format_func=lambda i: "No listed requirement — deliberate manual draft" if i == 0 else f"Requirement #{i}")
            intentional = st.checkbox("I intentionally want a cover letter draft", value=False)
            if st.button("Create cover letter draft"):
                _act(lambda: studio.create_cover_letter(profile_version_id, track_version_id, app_id, letter_id or None,
                    intentional=intentional), "Cover letter draft created")

    st.markdown("#### Generation review")
    artifacts = _rows(path, "SELECT * FROM generated_artifacts WHERE profile_version_id=? ORDER BY id DESC LIMIT 100", (profile_version_id,))
    if not artifacts:
        st.info("No generated or tailored drafts yet.")
        return
    artifact_id = st.selectbox("Material version", [a["id"] for a in artifacts],
        format_func=lambda i: next(f"#{a['id']} {a['kind']} · {a['approval_state']} · {a['generated_at']}" for a in artifacts if a["id"]==i))
    artifact = next(a for a in artifacts if a["id"] == artifact_id)
    st.write({"Application":artifact["application_id"],"Professor":artifact["faculty_profile_id"],
        "Profile version":artifact["profile_version_id"],"Track version":artifact["research_track_version_id"],
        "Requirement":artifact["requirement_id"],"Model":artifact["model"],"Template":artifact["template_id"],
        "Generated":artifact["generated_at"]})
    st.text_area("Editable source / rendered content", value=artifact["content_text"], height=260, disabled=True)
    st.write("Evidence and provenance")
    st.json({"claim_ids":json.loads(artifact["claim_revision_ids_json"]),
        "evidence_ids":json.loads(artifact["evidence_ids_json"]),
        "publication_ids":json.loads(artifact["publication_ids_json"]),
        "module_ids":json.loads(artifact["story_module_version_ids_json"])})
    st.write("Master → variant diff")
    st.json(json.loads(artifact["diff_json"]))
    st.write("Quality")
    st.json(json.loads(artifact["quality_json"]))
    version = vault.get_version(artifact["document_version_id"])
    st.download_button("Download rendered PDF", vault.storage.get(version["storage_key"]),
        file_name=version["canonical_filename"], mime="application/pdf")
    if artifact["approval_state"] == "DRAFT":
        reviewer = st.text_input("Document reviewer")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve material"):
                _act(lambda: studio.review_artifact(artifact_id, reviewer, True), "Material approved")
        with col2:
            if st.button("Reject material"):
                _act(lambda: studio.review_artifact(artifact_id, reviewer, False), "Material rejected")
        if artifact["kind"] not in {"CV","RESEARCH_PROPOSAL"}:
            edit = st.text_area("Edit as a new draft version", value=artifact["content_text"], height=200)
            if st.button("Save edited version"):
                _act(lambda: studio.revise_artifact(artifact_id, edit), "New draft version saved")


def render_packages(path: Path):
    st.subheader("Application packages and preflight")
    context = _approved_context(path, "packages")
    if not context:
        return
    _, profile_version_id, track_version_id = context
    apps = _rows(path, "SELECT * FROM applications ORDER BY id DESC")
    if not apps:
        st.info("Create an application and source its requirements first.")
        return
    app_id = st.selectbox("Application", [a["id"] for a in apps], format_func=lambda i: f"Application #{i}")
    purpose = st.selectbox("Package context", ["FORMAL_APPLICATION","FACULTY_OUTREACH","REPLY_REQUEST"])
    builder = PackageBuilder(path)
    rows = builder.ledger.requirement_rows(app_id)
    optional = [r for r in rows if r["context"] == purpose and r["requirement_state"] == "OPTIONAL"]
    selected_optional = set(st.multiselect("Deliberately include optional requirements", [r["id"] for r in optional],
        format_func=lambda i: next(r["original_label"] for r in optional if r["id"]==i)))
    preview = builder.preview(app_id, purpose, selected_optional)
    st.write("Package policy")
    st.dataframe(preview["decisions"], hide_index=True)
    c = preview["completeness"]
    st.info(f"Required items: {c['required_complete']}/{c['required_total']} · Unknown requirements: {c['unknown_requirements']} · Conditional: {c['conditional_requirements']} · Awaiting approval: {preview['awaiting_approval']}")
    combine = st.checkbox("Combine selected PDFs in manifest order (explicit request)")
    zip_export = st.checkbox("Also create ZIP")
    if st.button("Build application package"):
        _act(lambda: builder.build(app_id, profile_version_id, track_version_id, context=purpose,
            selected_optional=selected_optional, combine_pdf=combine, zip_export=zip_export), "Frozen local package built")
    packages = _rows(path, "SELECT * FROM application_packages WHERE application_id=? ORDER BY version_number DESC", (app_id,))
    if not packages:
        return
    package_id = st.selectbox("Review package version", [p["id"] for p in packages],
        format_func=lambda i: next(f"v{p['version_number']} · {p['status']} · #{i}" for p in packages if p["id"]==i))
    package = next(p for p in packages if p["id"] == package_id)
    st.caption(f"Frozen at {package['built_at']} · Export: {package['export_path']} · Package SHA-256: {package['package_sha256']}")
    st.json(json.loads(package["manifest_json"]))
    st.write("Inclusion, exclusion, and review reasons")
    st.dataframe(json.loads(package["decisions_json"]), hide_index=True)
    vault = DocumentVault(path)
    docs = _rows(path, "SELECT * FROM package_documents WHERE package_id=? ORDER BY sort_order", (package_id,))
    for doc in docs:
        version = vault.get_version(doc["document_version_id"])
        st.download_button(f"Download {doc['canonical_filename']} · {doc['sha256'][:12]}",
            vault.storage.get(version["storage_key"]), file_name=doc["canonical_filename"], key=f"package_download_{doc['id']}")
    if st.button("Run application/outreach preflight"):
        try:
            st.session_state["slice3_preflight"] = builder.preflight(package_id)
            st.session_state["slice3_preflight_package"] = package_id
        except Exception as error:
            st.error(str(error))
    result = st.session_state.get("slice3_preflight")
    if result and st.session_state.get("slice3_preflight_package") == package_id:
        st.metric("Preflight", result["status"])
        st.dataframe(result["rules"], hide_index=True)
        st.write("Administrative completeness", result["completeness"])
    reviewer = st.text_input("Package reviewer")
    if st.button("Mark package READY (reruns preflight)"):
        _act(lambda: builder.mark_ready(package_id, reviewer), "Package READY after preflight")
