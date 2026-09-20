"""Small manual Streamlit console for Slice 1."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import streamlit as st

from phd_agent.documents import DOCUMENT_CLASSES, DOCUMENT_TYPES, SENSITIVITIES, DocumentVault
from phd_agent.ledger import (
    APPLICATION_STATUSES, DEADLINE_TYPES, OPPORTUNITY_TYPES,
    REQUIREMENT_CONTEXTS, REQUIREMENT_STATES, Ledger,
)


def _none(value):
    return value or None


def _run(action, success: str):
    try:
        result = action()
    except Exception as error:
        st.error(str(error))
        return None
    st.success(success)
    return True if result is None else result


def _source_label(source: dict) -> str:
    return f"#{source['id']} · {source['source_type']} · {source['canonical_url']}"


def _app_label(app: dict) -> str:
    return f"#{app['id']} · {app['institution']} · {app['application_name']} ({app['cycle']})"


def _readiness_text(readiness: dict) -> str:
    return (
        f"Required items completed: {readiness['required_complete']}/{readiness['required_total']} · "
        f"Unknown requirements: {readiness['unknown_requirements']} · "
        f"Conditional requirements: {readiness['conditional_requirements']}"
    )


def render_today(ledger: Ledger):
    st.subheader("Today")
    data = ledger.today()
    counts = [
        ("Upcoming deadlines (60 days)", len(data["upcoming_deadlines"])),
        ("Missing required items", len(data["missing_required"])),
        ("Unknown requirements", len(data["unknown_requirements"])),
        ("Overdue tasks", len(data["overdue_tasks"])),
        ("Document alerts", len(data["document_alerts"])),
    ]
    cols = st.columns(len(counts))
    for col, (label, count) in zip(cols, counts):
        col.metric(label, count)

    st.markdown("#### Upcoming deadlines")
    if data["upcoming_deadlines"]:
        st.dataframe([
            {"Application": d["application_id"], "Institution": d["institution"],
             "Type": d["deadline_type"], "Due": d["due_at"],
             "Timezone": d["timezone"], "Verified": d["verification_state"],
             "Source": d["source_url"]}
            for d in data["upcoming_deadlines"]
        ], width='stretch', hide_index=True)
    else:
        st.info("No deadlines recorded in the next 60 days.")
    for heading, key, fields in (
        ("Missing required items", "missing_required", ("application_id", "institution", "original_label", "context", "document_state", "source_url")),
        ("Unknown requirements", "unknown_requirements", ("application_id", "institution", "original_label", "context", "source_url")),
        ("Overdue tasks", "overdue_tasks", ("application_id", "institution", "description", "due_at", "status")),
        ("Documents needing approval, update, or expiry review", "document_alerts", ("id", "title", "version_id", "approval_state", "expiry_date")),
    ):
        st.markdown(f"#### {heading}")
        if data[key]:
            st.dataframe([{field: item.get(field) for field in fields} for item in data[key]],
                         width='stretch', hide_index=True)
        else:
            st.caption("None recorded.")


def _create_forms(ledger: Ledger):
    programmes = ledger.list_programmes()
    opportunities = ledger.list_opportunities()
    sources = ledger.list_evidence()
    programme_by_id = {p["id"]: p for p in programmes}
    opportunity_by_id = {o["id"]: o for o in opportunities}
    source_by_id = {s["id"]: s for s in sources}

    with st.expander("Add official source evidence"):
        with st.form("cms_new_source"):
            url = st.text_input("Canonical source URL")
            source_type = st.selectbox("Source type", ["ADMISSIONS", "PROGRAMME", "VACANCY", "PORTAL", "EMAIL", "OTHER"])
            excerpt = st.text_area("Relevant excerpt (manual snapshot)")
            verification = st.selectbox("Verification", ["UNVERIFIED", "VERIFIED", "NEEDS_REVIEW"])
            if st.form_submit_button("Save source snapshot"):
                if _run(lambda: ledger.create_evidence(url, source_type, excerpt, verification), "Source snapshot saved"):
                    st.rerun()

    with st.expander("Add programme"):
        with st.form("cms_new_programme"):
            university = st.text_input("University / institution")
            name = st.text_input("Programme name")
            department = st.text_input("Department")
            degree = st.text_input("Degree type")
            cycle = st.text_input("Programme cycle", value="2026–27")
            programme_url = st.text_input("Programme URL")
            admissions_url = st.text_input("Admissions URL")
            portal_url = st.text_input("Portal URL")
            notes = st.text_area("Programme notes")
            if st.form_submit_button("Save programme"):
                if _run(lambda: ledger.create_programme(
                    university, name, department=_none(department), degree_type=_none(degree),
                    cycle=_none(cycle), programme_url=_none(programme_url),
                    admissions_url=_none(admissions_url), portal_url=_none(portal_url), notes=notes,
                ), "Programme saved"):
                    st.rerun()

    with st.expander("Add opportunity"):
        with st.form("cms_new_opportunity"):
            opportunity_type = st.selectbox("Opportunity type", OPPORTUNITY_TYPES)
            title = st.text_input("Opportunity title")
            institution = st.text_input("Opportunity institution")
            programme_id = st.selectbox(
                "Linked programme (if applicable)", [0] + list(programme_by_id),
                format_func=lambda i: "None" if i == 0 else f"#{i} {programme_by_id[i]['university']} — {programme_by_id[i]['programme_name']}",
            )
            canonical_url = st.text_input("Official opportunity URL")
            department_lab = st.text_input("Department / lab")
            supervisor = st.text_input("Supervisor (only if explicitly named)")
            research_area = st.text_input("Research area")
            funding = st.text_area("Funding as stated")
            eligibility = st.text_area("Eligibility as stated")
            deadline = st.text_input("Advertised deadline (YYYY-MM-DD or ISO datetime)")
            timezone = st.text_input("Deadline timezone")
            route = st.text_input("Application route")
            contact = st.text_input("Contact policy")
            opening_status = st.selectbox("Opening status", ["UNKNOWN", "OPEN", "CLOSED"])
            verification = st.selectbox("Opportunity verification", ["UNVERIFIED", "VERIFIED", "NEEDS_REVIEW"])
            source_id = st.selectbox("Source snapshot", [0] + list(source_by_id),
                                     format_func=lambda i: "None" if i == 0 else _source_label(source_by_id[i]))
            notes = st.text_area("Opportunity notes")
            if st.form_submit_button("Save opportunity"):
                if _run(lambda: ledger.create_opportunity(
                    opportunity_type, title, institution, programme_id=programme_id or None,
                    canonical_url=_none(canonical_url), department_lab=_none(department_lab),
                    supervisor_name=_none(supervisor), research_area=_none(research_area),
                    funding_text=_none(funding), eligibility_text=_none(eligibility),
                    deadline_at=_none(deadline), deadline_timezone=_none(timezone),
                    application_route=_none(route), contact_policy=_none(contact),
                    opening_status=opening_status, verification_state=verification,
                    source_evidence_id=source_id or None, notes=notes,
                ), "Opportunity saved"):
                    st.rerun()

    with st.expander("Add application"):
        with st.form("cms_new_application"):
            programme_id = st.selectbox(
                "Programme", [0] + list(programme_by_id), key="new_app_programme",
                format_func=lambda i: "None" if i == 0 else f"#{i} {programme_by_id[i]['university']} — {programme_by_id[i]['programme_name']}",
            )
            opportunity_id = st.selectbox(
                "Opportunity", [0] + list(opportunity_by_id),
                format_func=lambda i: "None" if i == 0 else f"#{i} {opportunity_by_id[i]['institution']} — {opportunity_by_id[i]['title']}",
            )
            cycle = st.text_input("Application cycle", value="2026–27")
            portal_url = st.text_input("Application portal URL")
            next_action = st.text_input("Next action")
            owner_notes = st.text_area("Owner notes")
            if st.form_submit_button("Save application"):
                if _run(lambda: ledger.create_application(
                    cycle, programme_id or None, opportunity_id or None,
                    portal_url=_none(portal_url), next_action=next_action,
                    owner_notes=owner_notes,
                ), "Application saved"):
                    st.rerun()


def _detail(ledger: Ledger, vault: DocumentVault, app: dict):
    app_id = app["id"]
    st.subheader(_app_label(app))
    st.write(_readiness_text(ledger.readiness(app_id)))
    programme = ledger.get("programmes", app["programme_id"]) if app["programme_id"] else None
    opportunity = ledger.get("opportunities", app["opportunity_id"]) if app["opportunity_id"] else None
    if programme:
        st.caption(f"Programme: {programme['university']} — {programme['programme_name']}")
        for label, field in (("Programme", "programme_url"), ("Admissions", "admissions_url")):
            if programme[field]:
                st.markdown(f"[{label} source]({programme[field]})")
        with st.expander("Edit programme details"):
            with st.form(f"cms_edit_programme_{programme['id']}"):
                p_name = st.text_input("Programme name", value=programme["programme_name"])
                p_department = st.text_input("Department", value=programme["department"] or "")
                p_url = st.text_input("Programme URL", value=programme["programme_url"] or "")
                p_admissions = st.text_input("Admissions URL", value=programme["admissions_url"] or "")
                p_portal = st.text_input("Programme portal URL", value=programme["portal_url"] or "")
                p_notes = st.text_area("Programme notes", value=programme["notes"])
                if st.form_submit_button("Update programme"):
                    if _run(lambda: ledger.update_programme(
                        programme["id"], programme_name=p_name, department=_none(p_department),
                        programme_url=_none(p_url), admissions_url=_none(p_admissions),
                        portal_url=_none(p_portal), notes=p_notes,
                    ), "Programme updated") is not None:
                        st.rerun()
    if opportunity:
        st.caption(f"Opportunity: {opportunity['opportunity_type']} · {opportunity['opening_status']} · {opportunity['verification_state']}")
        if opportunity["canonical_url"]:
            st.markdown(f"[Official opportunity]({opportunity['canonical_url']})")
        with st.expander("Edit opportunity details"):
            with st.form(f"cms_edit_opportunity_{opportunity['id']}"):
                o_title = st.text_input("Opportunity title", value=opportunity["title"])
                o_url = st.text_input("Official opportunity URL", value=opportunity["canonical_url"] or "")
                o_status = st.selectbox("Opening status", ["UNKNOWN", "OPEN", "CLOSED"],
                                        index=["UNKNOWN", "OPEN", "CLOSED"].index(opportunity["opening_status"]))
                o_verification = st.selectbox("Opportunity verification", ["UNVERIFIED", "VERIFIED", "NEEDS_REVIEW"],
                                              index=["UNVERIFIED", "VERIFIED", "NEEDS_REVIEW"].index(opportunity["verification_state"]))
                o_funding = st.text_area("Funding as stated", value=opportunity["funding_text"] or "")
                o_eligibility = st.text_area("Eligibility as stated", value=opportunity["eligibility_text"] or "")
                o_route = st.text_input("Application route", value=opportunity["application_route"] or "")
                o_contact = st.text_input("Contact policy", value=opportunity["contact_policy"] or "")
                o_notes = st.text_area("Opportunity notes", value=opportunity["notes"])
                if st.form_submit_button("Update opportunity"):
                    if _run(lambda: ledger.update_opportunity(
                        opportunity["id"], title=o_title, canonical_url=_none(o_url),
                        opening_status=o_status, verification_state=o_verification,
                        funding_text=_none(o_funding), eligibility_text=_none(o_eligibility),
                        application_route=_none(o_route), contact_policy=_none(o_contact),
                        notes=o_notes,
                    ), "Opportunity updated") is not None:
                        st.rerun()
    if app["portal_url"]:
        st.markdown(f"[Application portal]({app['portal_url']})")

    with st.form(f"cms_edit_app_{app_id}"):
        status = st.selectbox("Application status", APPLICATION_STATUSES,
                              index=APPLICATION_STATUSES.index(app["status"]) if app["status"] in APPLICATION_STATUSES else 0)
        funding = st.selectbox("Funding state", ["UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING", "CONFIRMED"],
                               index=["UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING", "CONFIRMED"].index(app["funding_state"]) if app["funding_state"] in ["UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING", "CONFIRMED"] else 0)
        eligibility = st.selectbox("Eligibility state", ["UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING"],
                                   index=["UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING"].index(app["eligibility_state"]) if app["eligibility_state"] in ["UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING"] else 0)
        contact = st.selectbox("Supervisor / contact state", ["UNKNOWN", "NOT_CONTACTED", "CONTACTED", "REPLIED", "NOT_REQUIRED"],
                               index=["UNKNOWN", "NOT_CONTACTED", "CONTACTED", "REPLIED", "NOT_REQUIRED"].index(app["supervisor_contact_state"]) if app["supervisor_contact_state"] in ["UNKNOWN", "NOT_CONTACTED", "CONTACTED", "REPLIED", "NOT_REQUIRED"] else 0)
        next_action = st.text_input("Next action", value=app["next_action"])
        owner_notes = st.text_area("Owner notes", value=app["owner_notes"])
        if st.form_submit_button("Update application"):
            if _run(lambda: ledger.update_application(
                app_id, status=status, funding_state=funding, eligibility_state=eligibility,
                supervisor_contact_state=contact, next_action=next_action,
                owner_notes=owner_notes,
            ), "Application updated") is not None:
                st.rerun()

    sources = ledger.list_evidence()
    source_by_id = {source["id"]: source for source in sources}
    st.markdown("#### Source snapshots")
    if sources:
        st.dataframe([{
            "ID": s["id"], "Type": s["source_type"], "URL": s["canonical_url"],
            "Checked": s["last_manually_verified_at"], "State": s["verification_state"],
            "Excerpt": s["relevant_excerpt"],
        } for s in sources], width='stretch', hide_index=True)
    else:
        st.info("Add an official source snapshot above before recording deadlines or requirements.")

    st.markdown("#### Deadlines")
    deadlines = ledger.list_deadlines(app_id)
    if deadlines:
        st.dataframe([{
            "Type": d["deadline_type"], "Due": d["due_at"], "Timezone": d["timezone"],
            "Verification": d["verification_state"], "Source": d["source_url"], "Notes": d["notes"],
        } for d in deadlines], width='stretch', hide_index=True)
        for deadline in deadlines:
            with st.expander(f"Edit {deadline['deadline_type']} deadline #{deadline['id']}"):
                with st.form(f"cms_edit_deadline_{deadline['id']}"):
                    due = st.text_input("Edit due date/time", value=deadline["due_at"])
                    tz = st.text_input("Edit timezone", value=deadline["timezone"] or "")
                    verification = st.selectbox(
                        "Edit verification", ["UNVERIFIED", "VERIFIED", "NEEDS_REVIEW"],
                        index=["UNVERIFIED", "VERIFIED", "NEEDS_REVIEW"].index(deadline["verification_state"]),
                    )
                    source = st.selectbox(
                        "Edit deadline source", list(source_by_id),
                        index=list(source_by_id).index(deadline["source_evidence_id"]),
                        format_func=lambda i: _source_label(source_by_id[i]),
                    )
                    notes = st.text_input("Edit deadline notes", value=deadline["notes"])
                    if st.form_submit_button("Update deadline"):
                        if _run(lambda d=deadline: ledger.update_deadline(
                            d["id"], due_at=due, timezone=_none(tz),
                            verification_state=verification, source_evidence_id=source,
                            notes=notes,
                        ), "Deadline updated") is not None:
                            st.rerun()
    with st.form(f"cms_add_deadline_{app_id}"):
        deadline_type = st.selectbox("Deadline type", DEADLINE_TYPES)
        due_at = st.text_input("Due date/time (YYYY-MM-DD or ISO datetime)")
        timezone = st.text_input("Timezone (e.g. Europe/London)")
        source_id = st.selectbox("Deadline source", [0] + list(source_by_id),
                                 format_func=lambda i: "Select source" if i == 0 else _source_label(source_by_id[i]))
        verification = st.selectbox("Deadline verification", ["UNVERIFIED", "VERIFIED", "NEEDS_REVIEW"])
        notes = st.text_input("Deadline notes")
        if st.form_submit_button("Add deadline"):
            if source_id == 0:
                st.error("A source snapshot is required")
            elif _run(lambda: ledger.create_deadline(
                app_id, deadline_type, due_at, source_id, timezone=_none(timezone),
                verification_state=verification, notes=notes,
            ), "Deadline added"):
                st.rerun()

    st.markdown("#### Requirements and selected documents")
    rows = ledger.requirement_rows(app_id)
    if rows:
        st.dataframe([{
            "ID": r["id"], "Context": r["context"], "State": r["requirement_state"],
            "Original label": r["original_label"], "Type": r["normalized_document_type"],
            "Condition": r["condition_text"], "Document state": r["document_state"],
            "Selected version": r["document_version_id"], "Source": r["source_url"],
        } for r in rows], width='stretch', hide_index=True)
    with st.form(f"cms_add_requirement_{app_id}"):
        context = st.selectbox("Requirement context", REQUIREMENT_CONTEXTS, index=REQUIREMENT_CONTEXTS.index("FORMAL_APPLICATION"))
        state = st.selectbox("Requirement state", REQUIREMENT_STATES)
        label = st.text_input("Original requirement label")
        doc_type = st.selectbox("Normalized document type", ["NONE"] + list(DOCUMENT_TYPES))
        condition = st.text_input("Applicability / condition")
        page_limit = st.number_input("Page limit (0 if none)", min_value=0, step=1)
        word_limit = st.number_input("Word limit (0 if none)", min_value=0, step=1)
        file_format = st.text_input("File format")
        filename_rule = st.text_input("Filename rule")
        upload_field = st.text_input("Portal upload field")
        source_id = st.selectbox("Requirement source", [0] + list(source_by_id),
                                 format_func=lambda i: "Select source" if i == 0 else _source_label(source_by_id[i]))
        notes = st.text_input("Requirement notes")
        if st.form_submit_button("Add requirement"):
            if source_id == 0:
                st.error("A source snapshot is required")
            elif _run(lambda: ledger.create_requirement(
                app_id, context, state, label, source_id,
                normalized_document_type=None if doc_type == "NONE" else doc_type,
                condition_text=_none(condition), page_limit=page_limit or None,
                word_limit=word_limit or None, file_format=_none(file_format),
                filename_rule=_none(filename_rule), upload_field=_none(upload_field),
                notes=notes,
            ), "Requirement added"):
                st.rerun()

    documents = vault.list_documents()
    for r in rows:
        with st.expander(f"#{r['id']} · {r['original_label']} · {r['document_state']}"):
            st.caption(f"{r['context']} · {r['requirement_state']} · {r['source_url']}")
            if r["document_version_id"]:
                st.write(f"Selected: {r['document_title']} · version {r['version_number']} · SHA-256 {r['sha256']}")
            if r["normalized_document_type"]:
                candidates = [d for d in documents if d["document_type"] == r["normalized_document_type"]]
                version_labels = {}
                for document in candidates:
                    for version in vault.list_versions(document["id"]):
                        version_labels[version["id"]] = (
                            f"#{document['id']} {document['title']} · v{version['version_number']} "
                            f"· {version['approval_state']}"
                        )
                if version_labels:
                    chosen = st.selectbox("Select existing version", list(version_labels),
                                          format_func=lambda i: version_labels[i], key=f"cms_map_select_{r['id']}")
                    mapping_notes = st.text_input("Mapping notes", value=r["mapping_notes"] or "",
                                                  key=f"cms_map_notes_{r['id']}")
                    if st.button("Link selected version", key=f"cms_map_{r['id']}"):
                        if _run(lambda: vault.link_to_application(app_id, r["id"], chosen, mapping_notes),
                                "Document linked") is not None:
                            st.rerun()
                else:
                    st.caption("No matching document type in the Vault. Upload one there first.")
                if r["mapping_id"] and st.button("Unlink version", key=f"cms_unmap_{r['id']}"):
                    if _run(lambda: vault.unlink_from_application(app_id, r["id"]), "Document unlinked") is not None:
                        st.rerun()
            else:
                label = "Mark incomplete" if r["fulfilled_at"] else "Mark complete without a file"
                if st.button(label, key=f"cms_fulfill_{r['id']}"):
                    if _run(lambda: ledger.mark_requirement_fulfilled(r["id"], not bool(r["fulfilled_at"])),
                            "Requirement updated") is not None:
                        st.rerun()
            with st.form(f"cms_edit_req_{r['id']}"):
                new_state = st.selectbox("Update state", REQUIREMENT_STATES,
                                         index=REQUIREMENT_STATES.index(r["requirement_state"]))
                new_condition = st.text_input("Update condition", value=r["condition_text"] or "")
                new_notes = st.text_input("Update notes", value=r["notes"])
                if st.form_submit_button("Update requirement"):
                    if _run(lambda: ledger.update_requirement(
                        r["id"], requirement_state=new_state,
                        condition_text=_none(new_condition), notes=new_notes,
                    ), "Requirement updated") is not None:
                        st.rerun()

    st.markdown("#### Tasks")
    tasks = ledger.list_tasks(app_id)
    if tasks:
        st.dataframe([{
            "ID": t["id"], "Type": t["task_type"], "Task": t["description"],
            "Status": t["status"], "Due": t["due_at"], "Priority": t["priority"],
        } for t in tasks], width='stretch', hide_index=True)
        for task in tasks:
            if task["status"] != "DONE" and st.button(f"Complete task #{task['id']}", key=f"cms_task_done_{task['id']}"):
                if _run(lambda t=task: ledger.update_task(t["id"], status="DONE"), "Task completed") is not None:
                    st.rerun()
    with st.form(f"cms_add_task_{app_id}"):
        task_type = st.text_input("Task type", value="APPLICATION")
        description = st.text_input("Task description")
        due_at = st.text_input("Task due date (YYYY-MM-DD)")
        priority = st.selectbox("Priority", ["MEDIUM", "HIGH", "LOW"])
        source_context = st.text_input("Task source/context")
        if st.form_submit_button("Add task"):
            if _run(lambda: ledger.create_task(
                app_id, task_type, description, due_at=_none(due_at),
                priority=priority, source_context=_none(source_context),
            ), "Task added"):
                st.rerun()

    st.markdown("#### Referees")
    referees = ledger.list_referees(app_id)
    if referees:
        st.dataframe([{
            "ID": r["id"], "Name": r["referee_name"], "Institution": r["institution"],
            "Invitation": r["invitation_state"], "Deadline": r["deadline_at"],
            "Submission": r["submission_state"], "Submitted": r["submitted_at"],
        } for r in referees], width='stretch', hide_index=True)
        for referee in referees:
            if referee["submission_state"] != "SUBMITTED" and st.button(
                f"Mark referee #{referee['id']} submitted", key=f"cms_ref_submitted_{referee['id']}"
            ):
                if _run(lambda r=referee: ledger.update_referee(
                    r["id"], submission_state="SUBMITTED", submitted_at=date.today().isoformat(),
                ), "Referee marked submitted") is not None:
                    st.rerun()
    with st.form(f"cms_add_referee_{app_id}"):
        name = st.text_input("Referee name")
        institution = st.text_input("Referee institution")
        email = st.text_input("Referee email")
        relationship = st.text_input("Relationship")
        deadline = st.text_input("Referee deadline (YYYY-MM-DD)")
        notes = st.text_input("Referee notes")
        if st.form_submit_button("Add referee"):
            if _run(lambda: ledger.create_referee(
                app_id, name, institution=_none(institution), email=_none(email),
                relationship=_none(relationship), deadline_at=_none(deadline), notes=notes,
            ), "Referee added"):
                st.rerun()


def render_applications(ledger: Ledger, vault: DocumentVault):
    st.subheader("Applications")
    _create_forms(ledger)
    applications = ledger.list_applications()
    if not applications:
        st.info("No applications yet. Add an official source, programme or opportunity, then an application.")
        return
    st.dataframe([{
        "ID": app["id"], "Institution": app["institution"],
        "Programme / opportunity": app["application_name"], "Status": app["status"],
        "Nearest deadline": app["nearest_deadline"], "Next action": app["next_action"],
        "Administrative readiness": _readiness_text(ledger.readiness(app["id"])),
    } for app in applications], width='stretch', hide_index=True)
    by_id = {app["id"]: app for app in applications}
    selected = st.selectbox("Open application", list(by_id), format_func=lambda i: _app_label(by_id[i]))
    _detail(ledger, vault, by_id[selected])


def render_vault(vault: DocumentVault):
    st.subheader("Document Vault")
    st.caption("Files remain local under data/documents. Uploads create immutable file versions; matching bytes are reused.")
    documents = vault.list_documents()
    by_id = {doc["id"]: doc for doc in documents}
    with st.form("cms_upload_document"):
        file = st.file_uploader("Choose a document")
        document_id = st.selectbox(
            "Add as a new version of an existing document", [0] + list(by_id),
            format_func=lambda i: "New document" if i == 0 else f"#{i} {by_id[i]['title']} ({by_id[i]['document_type']})",
        )
        st.caption("For a new version, choose the same class and type. Existing sensitivity is retained.")
        document_class = st.selectbox("Class", DOCUMENT_CLASSES)
        document_type = st.selectbox("Document type", DOCUMENT_TYPES)
        title = st.text_input("Document title")
        issuer = st.text_input("Issuer / institution")
        degree = st.text_input("Degree / programme")
        issue_date = st.text_input("Issue date (YYYY-MM-DD)")
        expiry_date = st.text_input("Expiry date (YYYY-MM-DD)")
        sensitivity_options = (["HIGHLY_SENSITIVE"] if document_type in {"PASSPORT", "GOVERNMENT_ID"}
                               else SENSITIVITIES)
        sensitivity = st.selectbox("Sensitivity (identity documents are local-only)", sensitivity_options)
        notes = st.text_area("Document notes")
        if st.form_submit_button("Upload document"):
            if file is None:
                st.error("Select a file")
            else:
                result = _run(lambda: vault.upload(
                    file.getvalue(), file.name, document_class, document_type, title,
                    document_id=document_id or None, issuer=_none(issuer),
                    degree_programme=_none(degree), issue_date=_none(issue_date),
                    expiry_date=_none(expiry_date),
                    sensitivity=by_id[document_id]["sensitivity"] if document_id else sensitivity,
                    notes=notes,
                ), "Upload checked")
                if result:
                    if result["status"] == "duplicate":
                        st.info(f"Identical bytes already exist as document #{result['document_id']}, "
                                f"version #{result['version_id']}. Link that version from an application requirement.")
                    else:
                        st.success(f"Saved document #{result['document_id']} version {result['version_number']}")
                        st.rerun()

    if not documents:
        st.info("No Vault documents yet. The preserved legacy CV can be uploaded here when ready.")
        return
    st.dataframe([{
        "ID": d["id"], "Class": d["document_class"], "Type": d["document_type"],
        "Title": d["title"], "Version": d["version_number"],
        "Approval": d["approval_state"], "Sensitivity": d["sensitivity"],
        "Allowed storage": d["permitted_storage_policy"], "Expiry": d["expiry_date"],
        "SHA-256": d["sha256"],
    } for d in documents], width='stretch', hide_index=True)
    selected = st.selectbox("Open document", list(by_id),
                            format_func=lambda i: f"#{i} {by_id[i]['title']}")
    doc = by_id[selected]
    versions = vault.list_versions(selected)
    with st.form(f"cms_edit_doc_{selected}"):
        title = st.text_input("Edit title", value=doc["title"])
        issuer = st.text_input("Edit issuer", value=doc["issuer"] or "")
        degree = st.text_input("Edit degree / programme", value=doc["degree_programme"] or "")
        issue = st.text_input("Edit issue date", value=doc["issue_date"] or "")
        expiry = st.text_input("Edit expiry date", value=doc["expiry_date"] or "")
        edit_sensitivities = (["HIGHLY_SENSITIVE"] if doc["document_type"] in {"PASSPORT", "GOVERNMENT_ID"}
                              else SENSITIVITIES)
        sensitivity = st.selectbox("Edit sensitivity", edit_sensitivities,
                                   index=edit_sensitivities.index(doc["sensitivity"]))
        notes = st.text_area("Edit notes", value=doc["notes"])
        if st.form_submit_button("Save metadata"):
            if _run(lambda: vault.update_document(
                selected, title=title, issuer=_none(issuer), degree_programme=_none(degree),
                issue_date=_none(issue), expiry_date=_none(expiry),
                sensitivity=sensitivity, notes=notes,
            ), "Document metadata updated") is not None:
                st.rerun()
    st.markdown("#### Versions")
    st.dataframe([{
        "Version ID": v["id"], "Version": v["version_number"],
        "Original filename": v["original_filename"], "Bytes": v["byte_size"],
        "SHA-256": v["sha256"], "Approval": v["approval_state"],
        "Uploaded": v["uploaded_at"],
    } for v in versions], width='stretch', hide_index=True)
    for version in versions:
        col_a, col_b = st.columns(2)
        with col_a:
            approved = version["approval_state"] == "APPROVED"
            if st.button(("Withdraw approval" if approved else "Approve") + f" version {version['version_number']}",
                         key=f"cms_approve_{version['id']}"):
                if _run(lambda v=version, a=approved: vault.set_approval(v["id"], not a, "Local operator"),
                        "Approval updated") is not None:
                    st.rerun()
        with col_b:
            if vault.storage.verify_hash(version["storage_key"], version["sha256"]):
                st.download_button(
                    f"Download version {version['version_number']}",
                    data=vault.storage.get(version["storage_key"]),
                    file_name=version["original_filename"],
                    mime=version["mime_type"], key=f"cms_download_{version['id']}",
                )
            else:
                st.error(f"Version {version['version_number']} is missing or corrupt")


def render_cms(db_path: Path):
    from phd_agent.ui_slice2 import render_discovery, render_tracks, render_truth

    ledger = Ledger(db_path)
    vault = DocumentVault(db_path)
    st.title("PhD applications")
    st.caption("Local application ledger, Document Vault, applicant truth, and reviewed discovery")
    today, applications, documents, truth, tracks, discovery = st.tabs([
        "Today", "Applications", "Document Vault", "Applicant Truth", "Research Directions", "Discovery",
    ])
    with today:
        render_today(ledger)
    with applications:
        render_applications(ledger, vault)
    with documents:
        render_vault(vault)
    with truth:
        render_truth(db_path)
    with tracks:
        render_tracks(db_path)
    with discovery:
        render_discovery(db_path)
