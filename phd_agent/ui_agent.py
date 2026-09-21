"""Intent-first Streamlit workspace over the reviewed application services."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import streamlit as st

from phd_agent.applicant_context import ApplicantResearchContextService
from phd_agent.browser_worker import FormPlanService, LocalPlaywrightWorker, fields_from_html
from phd_agent.config import load_settings
from phd_agent.db import connect
from phd_agent.discovery import Discovery
from phd_agent.documents import DocumentVault
from phd_agent.ledger import Ledger
from phd_agent.orchestration import ProgrammeOrchestrator
from phd_agent.outreach import OutreachService
from phd_agent.portal import PortalAssistance


NAVIGATION = (
    "Today / Agent", "Discover", "My Applications",
    "Professors & Messages", "Documents", "Browser Assistant",
)


def _companion_command(plan_id: int, platform_name: str | None = None) -> tuple[str, str]:
    if (platform_name or os.name) == "nt":
        return f".\\.venv\\Scripts\\python.exe -m scripts.browser_companion {plan_id}", "powershell"
    return f"./.venv/bin/python -m scripts.browser_companion {plan_id}", "bash"


def _styles() -> None:
    st.markdown("""
    <style>
      :root { --paper:#f7f6f2; --ink:#17202a; --muted:#667085; --rule:#d9ddd8;
              --blue:#2457d6; --green:#147a5b; --amber:#a86513; --red:#b33a3a; }
      .stApp { background:var(--paper); color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(247,246,242,.94); }
      [data-testid="stSidebar"] { background:#eef0ec; border-right:1px solid var(--rule); }
      h1,h2,h3 { letter-spacing:-.025em; color:var(--ink); }
      .agent-kicker { font:600 .72rem/1.2 ui-monospace,SFMono-Regular,Consolas,monospace;
                      letter-spacing:.1em; text-transform:uppercase; color:var(--blue); }
      .agent-hero { background:#fff; border:1px solid var(--rule); border-top:3px solid var(--blue);
                    padding:1.1rem 1.25rem; margin:.35rem 0 1rem; box-shadow:0 2px 8px rgba(23,32,42,.05); }
      .agent-hero h2 { margin:.2rem 0 .35rem; }
      .status-line { font:500 .76rem/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;
                     color:var(--muted); text-transform:uppercase; letter-spacing:.035em; }
      [data-testid="stMetric"] { background:#fff; border:1px solid var(--rule); padding:.75rem; }
      [data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:var(--rule); }
      .stButton button { border-radius:4px; min-height:2.5rem; font-weight:650; }
      .stButton button[kind="primary"] { background:var(--blue); border-color:var(--blue); }
      .verified { color:var(--green); font-weight:650; }
      .review { color:var(--amber); font-weight:650; }
      .blocked { color:var(--red); font-weight:650; }
      @media (max-width: 760px) {
        .agent-hero { padding:.9rem; }
        [data-testid="stMetric"] { min-width:9rem; }
      }
    </style>
    """, unsafe_allow_html=True)


def _act(action, success: str):
    try:
        result = action()
    except Exception as error:
        st.error(str(error))
        return None
    st.success(success)
    return True if result is None else result


def _reviewer() -> str:
    default = st.session_state.get("agent_reviewer") or os.getenv("USER_NAME", "")
    value = st.sidebar.text_input("Reviewer name", value=default,
                                  help="Recorded with shortlist, approval, and acceptance decisions")
    st.session_state.agent_reviewer = value
    return value


def _context(path: Path) -> dict | None:
    service = ApplicantResearchContextService(path)
    inputs = service.available_inputs()
    if not inputs:
        st.error("Approve an applicant profile snapshot, a Master CV, and an active research direction to activate the agent.")
        st.caption("Use Advanced / Legacy → Data & audit → Applicant Truth, Research Directions, and Materials.")
        return None
    labels = {
        index: (f"{row['owner_name']} · {row['title']} · CV v{row['master_cv_version']} · "
                f"track v{row['research_track_version']}")
        for index, row in enumerate(inputs)
    }
    selected = st.sidebar.selectbox("Applicant context", list(labels), format_func=labels.get,
                                    key="agent_context_input")
    row = inputs[selected]
    try:
        return service.build(row["profile_version_id"], row["master_cv_version_id"],
                             row["research_track_version_id"])
    except Exception as error:
        st.error(f"Applicant context could not be built: {error}")
        return None


def _header(context: dict | None) -> None:
    st.markdown('<div class="agent-kicker">Evidence-backed application workspace</div>', unsafe_allow_html=True)
    st.title("PhD Application Agent")
    if context:
        track = context["context"]["research_track"]["title"]
        st.caption(f"{context['context']['owner_name']} · {track} · context v{context['version_number']} · "
                   f"{context['context_sha256'][:10]}")
    else:
        st.caption("Complete the approved applicant context to start discovery and reuse.")


def _intent_box(orchestrator: ProgrammeOrchestrator, context: dict, *, key: str) -> None:
    st.markdown("""<div class="agent-hero"><div class="agent-kicker">Start with intent</div>
      <h2>What do you want to do?</h2>
      <div>Describe the research direction, cycle, funding need, and regions. The agent retrieves only relevant approved applicant evidence.</div></div>""",
      unsafe_allow_html=True)
    with st.form(f"intent_{key}"):
        intent = st.text_area("Research and application intent", height=110,
            placeholder="Find funded 2027 PhD programmes in reliable AI, agent evaluation and RAG reliability in the UK, Europe, US and Canada.",
            label_visibility="collapsed")
        submitted = st.form_submit_button("Discover programmes", type="primary")
    if submitted:
        created = _act(lambda: orchestrator.create_intent(intent, context["id"]),
                       "Intent saved with applicant-context provenance")
        if created:
            st.session_state.agent_intent_id = created["id"]
            _act(lambda: orchestrator.discover_from_ledger(created["id"]),
                 "Reviewed local sources searched")
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if api_key:
                with st.spinner("Searching current official university sources…"):
                    _act(lambda: orchestrator.discover_official_web(created["id"], api_key=api_key),
                         "Official web sources searched and acquired for review")
            st.rerun()


def _candidate_card(orchestrator: ProgrammeOrchestrator, candidate: dict, reviewer: str) -> None:
    payload = candidate["payload"]
    title = payload.get("programme") or "Programme name needs review"
    institution = payload.get("university") or "Institution needs review"
    with st.container(border=True):
        top, state = st.columns([4, 1])
        top.markdown(f"### {html.escape(title)}")
        top.caption(f"{institution} · {payload.get('degree') or 'Degree unknown'} · {payload.get('intake') or 'Intake unknown'}")
        state.markdown(f"**{candidate['review_state']}**")
        cols = st.columns(4)
        cols[0].metric("Research Fit", f"{payload.get('research_fit', 0):.1f}/10")
        cols[1].metric("Confidence", f"{candidate['confidence']:.0%}")
        cols[2].metric("Deadline", payload.get("deadline") or "UNKNOWN")
        cols[3].metric("Unknown fields", len(payload.get("unknown_fields", [])))
        if payload.get("funding"):
            st.write("**Funding evidence:**", payload["funding"])
        if payload.get("relevant_applicant_experience"):
            st.write("**Relevant approved applicant context**")
            for item in payload["relevant_applicant_experience"]:
                st.write("•", item)
        st.caption("Research Fit describes evidence-backed alignment; it is not an admission probability.")
        with st.expander("Review extracted fields and evidence"):
            st.dataframe([
                {"Field": field, "Value": payload.get(field), "Evidence state": evidence.get("state"),
                 "Evidence": str(evidence.get("excerpt", ""))[:500]}
                for field, evidence in candidate["field_evidence"].items() if field != "applicant_context"
            ], width="stretch", hide_index=True)
            with st.form(f"accept_candidate_{candidate['id']}"):
                university = st.text_input("University", value=payload.get("university") or "")
                programme = st.text_input("Programme", value=payload.get("programme") or "")
                c1, c2, c3 = st.columns(3)
                department = c1.text_input("Department", value=payload.get("department") or "")
                degree = c2.text_input("Degree", value=payload.get("degree") or "")
                intake = c3.text_input("Cycle / intake", value=payload.get("intake") or "2027")
                deadline = st.text_input("Deadline (ISO date when confirmed)", value=payload.get("deadline") or "")
                contact_policy = st.selectbox("Supervisor contact policy",
                    ["UNKNOWN", "CONTACT_ALLOWED", "CONTACT_REQUIRED", "DO_NOT_CONTACT"],
                    index=["UNKNOWN", "CONTACT_ALLOWED", "CONTACT_REQUIRED", "DO_NOT_CONTACT"].index(
                        payload.get("supervisor_contact_policy") if payload.get("supervisor_contact_policy") in
                        {"UNKNOWN", "CONTACT_ALLOWED", "CONTACT_REQUIRED", "DO_NOT_CONTACT"} else "UNKNOWN"))
                accepted = st.form_submit_button("Accept and add application", type="primary")
            if accepted:
                result = _act(lambda: orchestrator.accept_candidate(candidate["id"], reviewer,
                    cycle=intake or None, field_overrides={
                        "university": university, "programme": programme, "department": department,
                        "degree": degree, "intake": intake, "deadline": deadline or None,
                        "supervisor_contact_policy": contact_policy,
                    }), "Programme and application added to the reviewed ledger")
                if result:
                    st.rerun()
        if candidate["review_state"] != "ACCEPTED":
            left, right, _ = st.columns([1, 1, 3])
            if left.button("Shortlist", key=f"shortlist_{candidate['id']}"):
                if _act(lambda: orchestrator.review_candidate(candidate["id"], "SHORTLISTED", reviewer),
                        "Programme shortlisted"):
                    st.rerun()
            if right.button("Reject", key=f"reject_{candidate['id']}"):
                if _act(lambda: orchestrator.review_candidate(candidate["id"], "REJECTED", reviewer),
                        "Programme rejected"):
                    st.rerun()
        st.link_button("Open official source", candidate["canonical_url"])


def _today(path: Path, context: dict, reviewer: str) -> None:
    ledger = Ledger(path)
    orchestrator = ProgrammeOrchestrator(path)
    _intent_box(orchestrator, context, key="home")
    today = ledger.today()
    stale = ApplicantResearchContextService(path).stale_outputs()
    cols = st.columns(5)
    cols[0].metric("Deadlines · 60 days", len(today["upcoming_deadlines"]))
    cols[1].metric("Missing required", len(today["missing_required"]))
    cols[2].metric("Unknown requirements", len(today["unknown_requirements"]))
    cols[3].metric("Overdue tasks", len(today["overdue_tasks"]))
    cols[4].metric("Context review", len(stale))
    candidates = orchestrator.list_candidates(states=("NEW", "SHORTLISTED"))[:4]
    st.subheader("Next reviews")
    if candidates:
        for candidate in candidates:
            _candidate_card(orchestrator, candidate, reviewer)
    elif today["overdue_tasks"]:
        st.dataframe(today["overdue_tasks"][:8], width="stretch", hide_index=True)
    else:
        st.info("No programme candidates are waiting. Start with an intent or analyse a programme URL.")
    workload = orchestrator.workload_summary()
    if workload:
        with st.expander("Workload avoided"):
            st.dataframe([{"Measure": key.replace("_", " ").title(), "Count": value}
                          for key, value in sorted(workload.items())], width="stretch", hide_index=True)


def _discover(path: Path, context: dict, reviewer: str) -> None:
    orchestrator = ProgrammeOrchestrator(path)
    st.header("Discover")
    intents = orchestrator.list_intents()
    if not intents:
        _intent_box(orchestrator, context, key="discover")
        return
    by_id = {row["id"]: row for row in intents}
    selected = st.selectbox("Research intent", list(by_id),
        index=list(by_id).index(st.session_state.agent_intent_id) if st.session_state.get("agent_intent_id") in by_id else 0,
        format_func=lambda value: by_id[value]["intent_text"][:120])
    st.caption("Context expansion: " + ", ".join(by_id[selected]["filters"].get("context_terms", [])))
    search, new_intent = st.columns([1, 1])
    if search.button("Search official sources", type="primary"):
        _act(lambda: orchestrator.discover_from_ledger(selected), "Reviewed sources searched")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            with st.spinner("Searching current official university sources…"):
                _act(lambda: orchestrator.discover_official_web(selected, api_key=api_key),
                     "Official web sources searched and acquired for review")
        else:
            st.info("Set OPENAI_API_KEY to add current official web search; reviewed local sources were searched.")
        st.rerun()
    if new_intent.button("Start another intent"):
        st.session_state.pop("agent_intent_id", None)
        st.session_state.show_new_intent = True
    if st.session_state.get("show_new_intent"):
        _intent_box(orchestrator, context, key="discover_new")
    st.divider()
    st.subheader("Analyse Programme URL")
    with st.form("analyse_programme_url"):
        url = st.text_input("Official programme, admissions, studentship, or project URL")
        local_playwright = False
        if not load_settings().hosted:
            local_playwright = st.checkbox("Use local Playwright if the static page is blocked or JavaScript-only")
        analyse = st.form_submit_button("Analyse URL", type="primary")
    if analyse:
        worker = LocalPlaywrightWorker() if local_playwright else None
        result = _act(lambda: orchestrator.analyse_url(
            url, context["id"], intent_id=selected, browser_worker=worker),
                      "Programme page analysed")
        if result:
            st.session_state.last_ingestion_result = result
            st.rerun()
    result = st.session_state.get("last_ingestion_result")
    if result and result.get("status") == "HUMAN_INPUT_REQUIRED":
        st.warning(result["message"])
        st.caption(result.get("reason", ""))
    with st.expander("Paste or upload a page when automated retrieval fails", expanded=bool(result and result.get("status") == "HUMAN_INPUT_REQUIRED")):
        supplied_url = st.text_input("Original official URL", key="supplied_url")
        uploaded = st.file_uploader("Saved HTML or PDF", type=["html", "htm", "pdf"])
        pasted = st.text_area("Or paste the relevant page text", height=180)
        if st.button("Structure supplied evidence"):
            if uploaded:
                method = "UPLOADED_PDF" if uploaded.name.lower().endswith(".pdf") else "UPLOADED_HTML"
                supplied = uploaded.getvalue()
                filename = uploaded.name
            else:
                method, supplied, filename = "PASTED_TEXT", pasted, None
            if _act(lambda: orchestrator.analyse_supplied(
                supplied_url, supplied, context["id"], intent_id=selected,
                method=method, filename=filename), "Supplied evidence structured for review"):
                st.rerun()
    st.subheader("Candidate programmes")
    candidates = orchestrator.list_candidates(intent_id=selected, states=("NEW", "SHORTLISTED", "ACCEPTED"))
    if not candidates:
        st.info("No candidates yet. Search reviewed sources or analyse an official URL.")
    for candidate in candidates:
        _candidate_card(orchestrator, candidate, reviewer)


def _applications(path: Path, context: dict) -> None:
    orchestrator = ProgrammeOrchestrator(path)
    apps = Ledger(path).list_applications()
    st.header("My Applications")
    if not apps:
        st.info("Accepted programme candidates will appear here.")
        return
    for app in apps:
        overview = orchestrator.application_overview(app["id"])
        readiness = overview["readiness"]
        with st.container(border=True):
            title, status = st.columns([4, 1])
            title.markdown(f"### {html.escape(app['institution'])} · {html.escape(app['application_name'])}")
            title.caption(f"{app['cycle']} · deadline {app['nearest_deadline'] or 'UNKNOWN'}")
            status.markdown(f"**{app['status']}**")
            cols = st.columns(4)
            cols[0].metric("Required", f"{readiness['required_complete']}/{readiness['required_total']}")
            cols[1].metric("Unknown", readiness["unknown_requirements"])
            cols[2].metric("Conditional", readiness["conditional_requirements"])
            cols[3].metric("Open tasks", len(overview["open_tasks"]))
            st.write("**Next action:**", overview["next_action"])
            if st.button("Prepare Application", key=f"prepare_app_{app['id']}", type="primary"):
                prepared = _act(lambda: orchestrator.prepare_application(
                    app["id"], context_id=context["id"]), "Application preparation checked")
                if prepared:
                    st.session_state[f"prepared_app_{app['id']}"] = prepared
            prepared = st.session_state.get(f"prepared_app_{app['id']}")
            if prepared:
                if prepared["status"] == "READY":
                    st.success("Ready for operator submission review")
                else:
                    st.warning("Not ready")
                    for reason in prepared["blocking"]:
                        st.write("•", reason)
            with st.expander("Requirements and generated documents"):
                st.dataframe([{ "Requirement": row["original_label"], "State": row["requirement_state"],
                               "Document": row.get("document_state") or "MISSING", "Context": row["context"]}
                              for row in overview["requirements"]], width="stretch", hide_index=True)
                if overview["generated_documents"]:
                    st.dataframe(overview["generated_documents"], width="stretch", hide_index=True)


def _professors(path: Path, context: dict, reviewer: str) -> None:
    orchestrator = ProgrammeOrchestrator(path)
    apps = Ledger(path).list_applications()
    st.header("Professors & Messages")
    if not apps:
        st.info("Add an application before finding relevant supervisors.")
        return
    app_map = {app["id"]: app for app in apps}
    app_id = st.selectbox("Application", list(app_map),
        format_func=lambda value: f"{app_map[value]['institution']} · {app_map[value]['application_name']}")
    if st.button("Find Relevant Supervisors", type="primary"):
        cards = _act(lambda: orchestrator.professor_cards(app_id, context["id"]), "Verified faculty assessed against applicant context")
        if cards is not None:
            st.session_state[f"professor_cards_{app_id}"] = cards
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            with st.spinner("Searching and acquiring official faculty pages…"):
                discovered = _act(lambda: orchestrator.discover_faculty_official_web(
                    app_id, context["id"], api_key=api_key),
                    "Official faculty pages queued for identity and evidence review")
                if discovered:
                    st.session_state[f"faculty_discovery_{app_id}"] = discovered
        else:
            st.info("Set OPENAI_API_KEY to add current official faculty-page search; verified local faculty were assessed.")
    cards = st.session_state.get(f"professor_cards_{app_id}", [])
    with connect(path) as db:
        linked = {row[0] for row in db.execute("SELECT faculty_profile_id FROM application_faculty WHERE application_id=?", (app_id,))}
        packages = [dict(row) for row in db.execute("""SELECT * FROM application_packages
            WHERE application_id=? AND context='FACULTY_OUTREACH' AND status='READY'
            ORDER BY version_number DESC""", (app_id,))]
    if not cards:
        st.caption("This action uses current verified faculty records for the application's institution. Add or verify faculty sources in Advanced when none are available.")
    discovered = st.session_state.get(f"faculty_discovery_{app_id}")
    if discovered and discovered["queued"]:
        st.subheader("Faculty candidates requiring verification")
        for item in discovered["queued"]:
            with st.container(border=True):
                st.markdown(f"#### {html.escape(item['name'])}")
                st.caption(" · ".join(filter(None, (item["department"], item["institution"]))))
                st.write(item["relevance_reason"])
                st.write("**Applicant context used:**", "; ".join(item["relevant_applicant_experience"][:3]))
                st.warning("Identity, affiliation, topics, email, and supervision state still require official evidence review.")
                left, right, _ = st.columns([1, 1, 3])
                if left.button("Create profile for verification", key=f"review_faculty_candidate_{item['candidate_id']}"):
                    if _act(lambda candidate_id=item["candidate_id"]: Discovery(path).review_faculty_candidate(candidate_id),
                            "Unverified faculty profile created for evidence review"):
                        st.session_state.pop(f"faculty_discovery_{app_id}", None)
                        st.rerun()
                if right.button("Dismiss", key=f"dismiss_faculty_candidate_{item['candidate_id']}"):
                    if _act(lambda candidate_id=item["candidate_id"]: Discovery(path).review_faculty_candidate(candidate_id, dismiss=True),
                            "Faculty candidate dismissed"):
                        st.session_state.pop(f"faculty_discovery_{app_id}", None)
                        st.rerun()
                st.link_button("Open official profile", item["official_url"])
    for card in cards:
        with st.container(border=True):
            st.markdown(f"### {html.escape(card['name'])}")
            st.caption(" · ".join(filter(None, (card["department"], card["lab"], card["institution"]))))
            c1, c2, c3 = st.columns(3)
            c1.metric("Research Fit", f"{card['research_fit']:.1f}/10")
            c2.metric("Contact policy", card["contact_policy_state"])
            c3.metric("Unknowns", len(card["unknowns"]))
            st.write("**Professor research:**", card["research_topics"] or "UNKNOWN")
            if card["recent_work"]:
                st.write("**Recent verified work:**", "; ".join(card["recent_work"]))
            st.write("**Relevant applicant CV experience:**")
            for item in card["relevant_applicant_experience"] or ["No demonstrated overlap retrieved"]:
                st.write("•", item)
            if card["proposed_overlap"]:
                st.write("**Proposed direction overlap:**", card["proposed_overlap"][0])
            if card["unknowns"]:
                st.warning("Unknown: " + ", ".join(card["unknowns"]))
            if card["faculty_id"] not in linked:
                if st.button("Add to application", key=f"link_faculty_{app_id}_{card['faculty_id']}"):
                    evidence_id = card["evidence_ids"][0] if card["evidence_ids"] else None
                    if _act(lambda: Discovery(path).link_to_application(
                        app_id, faculty_id=card["faculty_id"], evidence_id=evidence_id),
                        "Supervisor added to application"):
                        st.rerun()
            elif packages and card["contact_policy_state"] == "PASS":
                if st.button("Prepare Outreach", key=f"prepare_outreach_{app_id}_{card['faculty_id']}", type="primary"):
                    if _act(lambda: OutreachService(path).prepare(
                        card["faculty_id"], app_id, packages[0]["id"],
                        context["profile_version_id"], context["research_track_version_id"]),
                        "Grounded outreach draft prepared for review"):
                        st.rerun()
            else:
                st.caption("Prepare a READY faculty outreach package and verify contact policy before drafting.")
    with st.expander("Review messages and contact memory"):
        from phd_agent.ui_slice4 import render_outreach
        render_outreach(path)


def _documents(path: Path, context: dict, reviewer: str) -> None:
    contexts = ApplicantResearchContextService(path)
    portal = PortalAssistance(path)
    profile = contexts.profile_review(context["id"])
    st.header("Documents & Application Profile")
    a, b, c = st.columns(3)
    a.metric("Approved facts", len(profile["demonstrated"]))
    b.metric("Reusable answers", len(profile["approved_answers"]))
    c.metric("Approved documents", len(profile["approved_documents"]))
    with st.container(border=True):
        st.subheader("Application Profile")
        st.write("Create reusable draft fields from the approved Master CV and profile, then review them once.")
        if st.button("Extract reusable profile fields", type="primary"):
            created = _act(lambda: contexts.bootstrap_profile_answers(context["id"]),
                           "Draft Application Profile fields created")
            if created is not None:
                st.rerun()
        drafts = [row for row in portal.list_answers() if row["approval_state"] == "DRAFT"]
        for answer in drafts:
            with st.expander(f"Review · {answer['label']}"):
                st.write(answer["value_text"])
                st.caption(f"Claim #{answer['claim_revision_id'] or 'profile metadata'} · version {answer['version_number']}")
                if st.button("Approve reusable answer", key=f"approve_answer_{answer['id']}"):
                    if _act(lambda answer_id=answer["id"]: portal.approve_answer(answer_id, reviewer),
                            "Reusable answer approved"):
                        st.rerun()
    common_types = {"DEGREE_CERTIFICATE", "TRANSCRIPT", "MARKSHEET", "ENGLISH_TEST", "STANDARDIZED_TEST",
                    "PASSPORT", "GOVERNMENT_ID", "CERTIFICATE", "PUBLICATION", "PATENT_DOCUMENT"}
    common = [row for row in profile["approved_documents"] if row["document_type"] in common_types]
    specific = [row for row in profile["approved_documents"] if row["document_type"] not in common_types]
    left, right = st.columns(2)
    with left:
        st.subheader("Common / reusable")
        st.dataframe(common, width="stretch", hide_index=True) if common else st.caption("No approved reusable documents")
    with right:
        st.subheader("Application-specific")
        st.dataframe(specific, width="stretch", hide_index=True) if specific else st.caption("No approved application-specific documents")
    with st.expander("Upload, review, and inspect Document Vault"):
        from phd_agent.ui import render_vault
        render_vault(DocumentVault(path))


def _browser(path: Path, context: dict, reviewer: str) -> None:
    apps = Ledger(path).list_applications()
    st.header("Browser Assistant")
    st.caption("Analyse a portal form here, approve the fill plan, then run it in the local Playwright companion. The final submit, declarations, MFA, and payment always remain manual.")
    if not apps:
        st.info("Add an application before analysing a portal form.")
        return
    app_map = {app["id"]: app for app in apps}
    app_id = st.selectbox("Application", list(app_map), key="browser_app",
        format_func=lambda value: f"{app_map[value]['institution']} · {app_map[value]['application_name']}")
    page_url = st.text_input("Current portal page URL")
    uploaded = st.file_uploader("Save the current page as HTML and upload it", type=["html", "htm"], key="portal_html")
    pasted = st.text_area("Or paste the page HTML", height=150)
    if st.button("Analyse form fields", type="primary"):
        markup = uploaded.getvalue().decode("utf-8", errors="replace") if uploaded else pasted
        fields = fields_from_html(markup)
        if not fields:
            st.error("No form fields were found in the supplied HTML.")
        else:
            plan = _act(lambda: FormPlanService(path).create_plan(
                app_id, page_url, fields, context_id=context["id"]), "Fill plan created")
            if plan:
                st.session_state.browser_plan_id = plan["id"]
    plan_id = st.session_state.get("browser_plan_id")
    if plan_id:
        plan = FormPlanService(path).get(plan_id)
        cols = st.columns(4)
        cols[0].metric("Fields detected", plan["field_count"])
        cols[1].metric("Safe autofill", plan["safe_count"])
        cols[2].metric("Generated · review", plan["review_count"])
        cols[3].metric("Manual", plan["manual_count"])
        st.dataframe(plan["items"], width="stretch", hide_index=True)
        if plan["status"] == "REVIEW_REQUIRED" and st.button("Approve fill plan"):
            if _act(lambda: FormPlanService(path).approve(plan_id, reviewer), "Fill plan approved for local execution"):
                st.rerun()
        if plan["status"] in {"APPROVED", "FILLED"}:
            command, language = _companion_command(plan_id)
            st.code(command, language=language)
            st.caption("Run this on the applicant's computer. The companion uses a local browser profile and closes without submitting.")


def render_agent(db_path: Path) -> None:
    _styles()
    reviewer = _reviewer()
    context = _context(db_path)
    _header(context)
    if not context:
        return
    page = st.sidebar.radio("Navigate", NAVIGATION, label_visibility="collapsed")
    st.sidebar.caption("INTENT → DISCOVER → SHORTLIST → PROFESSORS → CONTACT → REPLY → DOCUMENTS → APPLICATION → SUBMISSION")
    if page == "Today / Agent":
        _today(db_path, context, reviewer)
    elif page == "Discover":
        _discover(db_path, context, reviewer)
    elif page == "My Applications":
        _applications(db_path, context)
    elif page == "Professors & Messages":
        _professors(db_path, context, reviewer)
    elif page == "Documents":
        _documents(db_path, context, reviewer)
    else:
        _browser(db_path, context, reviewer)
