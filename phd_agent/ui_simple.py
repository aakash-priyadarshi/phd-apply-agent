"""Simple task-based Streamlit workspace for one PhD applicant."""

from __future__ import annotations

import html
import os
from pathlib import Path

import streamlit as st

from phd_agent.applicant_context import ApplicantResearchContextService
from phd_agent.browser_worker import FormPlanService, companion_command, fields_from_html
from phd_agent.config import load_settings
from phd_agent.db import connect
from phd_agent.discovery import Discovery
from phd_agent.ledger import APPLICATION_STATUSES, Ledger
from phd_agent.operations import OperationService
from phd_agent.orchestration import ProgrammeOrchestrator
from phd_agent.outreach import OutreachService
from phd_agent.profile_workspace import ProfileUpload, ProfileWorkspace
from phd_agent.record_controls import PROGRAMME_TYPES, RecordControls, filter_programmes, group_programmes
from phd_agent.university_enrichment import REGION_COUNTRY_CODES, qs_sort_key, resolve_country_label


PAGES = ("Home", "Find programmes", "Applications", "Universities", "People", "My documents", "Searches", "Operations")
COUNTRY_CHOICES = ("UK", "US", "Canada", "Switzerland", "Australia", "Singapore", "Germany",
                   "Netherlands", "France", "Ireland", "Sweden", "Europe")


def _country_codes(names: list[str]) -> list[str]:
    codes = set()
    for name in names:
        if name.upper() in REGION_COUNTRY_CODES:
            codes.update(REGION_COUNTRY_CODES[name.upper()])
        else:
            code = resolve_country_label(name)[1]
            if code:
                codes.add(code)
    return sorted(codes)


def _country_selections(codes: list[str]) -> list[str]:
    selected = set(codes)
    return [name for name in COUNTRY_CHOICES if set(_country_codes([name])).issubset(selected)]


def _css() -> None:
    st.markdown("""
    <style>
      :root { --canvas:#f5f6f8; --surface:#ffffff; --ink:#18202b; --muted:#667085;
              --line:#e3e7ed; --brand:#3157d5; --brand-soft:#edf1ff; --good:#147a5b; --warn:#a56315; }
      .stApp { background:var(--canvas); color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(245,246,248,.94); }
      [data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
      [data-testid="stSidebar"] [role="radiogroup"] label { padding:.42rem .55rem; border-radius:8px; }
      h1,h2,h3 { color:var(--ink); letter-spacing:-.028em; }
      h1 { font-size:2rem; }
      .simple-hero { background:var(--surface); border:1px solid var(--line); border-radius:16px;
                     padding:1.4rem 1.5rem; margin:.2rem 0 1.25rem; box-shadow:0 5px 18px rgba(23,32,42,.05); }
      .simple-hero h2 { margin:.2rem 0 .35rem; }
      .eyebrow { color:var(--brand); font-size:.74rem; font-weight:750; letter-spacing:.09em; text-transform:uppercase; }
      .muted { color:var(--muted); }
      .ready-pill { display:inline-block; color:var(--good); background:#eaf7f1; border-radius:999px;
                    padding:.25rem .65rem; font-size:.78rem; font-weight:700; }
      [data-testid="stVerticalBlockBorderWrapper"] { background:var(--surface); border-color:var(--line); border-radius:14px; }
      [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:.7rem; }
      .stButton button { min-height:2.65rem; border-radius:9px; font-weight:680; }
      .stButton button[kind="primary"] { background:var(--brand); border-color:var(--brand); }
      .stTextArea textarea,.stTextInput input { border-radius:9px; }
      .source-note { color:var(--muted); font-size:.82rem; }
      @media (max-width:760px) { .simple-hero { padding:1rem; } h1 { font-size:1.65rem; } }
    </style>
    """, unsafe_allow_html=True)


def _run(call, success: str | None = None):
    try:
        result = call()
    except Exception as error:
        st.error(str(error))
        return None
    if success:
        st.success(success)
    return True if result is None else result


def _qs_caption(payload_or_app: dict) -> str:
    country = payload_or_app.get("country")
    flag = {"United States": "🇺🇸", "United Kingdom": "🇬🇧", "Canada": "🇨🇦",
            "Switzerland": "🇨🇭", "Australia": "🇦🇺", "Singapore": "🇸🇬"}.get(country or "", "")
    country_bit = " ".join(part for part in (flag, country) if part)
    display = payload_or_app.get("qs_rank_display")
    year = payload_or_app.get("qs_ranking_year") or 2027
    state = payload_or_app.get("qs_match_state")
    if display and state in {"EXACT", "ALIAS_MATCH"}:
        qs_bit = f"QS World Rank {year}: {display}"
    elif state == "NOT_RANKED":
        qs_bit = f"QS {year}: Not ranked"
    else:
        qs_bit = f"QS {year}: Unknown"
    return " · ".join(part for part in (country_bit, qs_bit) if part)


def _profile_banner(path: Path, context: dict) -> None:
    if context.get("trust_level") != "EXPLORATION":
        return
    workspace = ProfileWorkspace(path)
    latest = workspace.latest_summary()
    counts = {}
    for item in context["context"].get("items", []):
        if item.get("kind") == "PROPOSED_DIRECTION":
            continue
        heading = item.get("category") or "OTHER"
        counts[heading] = counts.get(heading, 0) + 1
    st.warning("Extracted facts are available for search. Confirm this profile before professor emails, SOPs, proposals, or portal answers.")
    with st.container(border=True):
        st.markdown("### Review extracted profile")
        if latest:
            st.markdown(latest[1])
        if counts:
            st.caption(" · ".join(f"{heading.replace('_', ' ').title()} ✓ {count}" for heading, count in counts.items()))
        if st.button("Use this profile", type="primary", key="confirm_extracted_profile"):
            result = _run(lambda: workspace.confirm(context["context"]["profile_id"]),
                          "Profile confirmed. Outreach and application documents can now use these facts.")
            if result:
                st.session_state.simple_profile_notice = {
                    "message": f"Trusted profile is active with {result.facts_in_profile} confirmed facts.",
                    "warnings": result.warnings,
                }
                st.rerun()


def _browser_continue(path: Path, application: dict, context: dict) -> None:
    portal = application.get("portal_url")
    st.markdown("**Continue application in browser**")
    if portal:
        st.link_button("Open application portal", portal)
    else:
        st.caption("Add the official application URL to continue in the browser.")
        return
    if context.get("trust_level") != "TRUSTED":
        st.caption("Confirm your profile before the companion can fill reviewed answers.")
        return
    if load_settings().hosted:
        st.caption("Form filling runs on your computer, not on Railway. Open this workspace locally to analyse the portal page.")
        return
    html_upload = st.file_uploader("Save the current portal page as HTML", type=["html", "htm"],
                                   key=f"portal_html_{application['id']}")
    pasted = st.text_area("Or paste the page HTML", height=120, key=f"portal_paste_{application['id']}")
    if st.button("Analyse this page", key=f"analyse_portal_{application['id']}"):
        markup = html_upload.getvalue().decode("utf-8", errors="replace") if html_upload else pasted
        fields = fields_from_html(markup)
        if not fields:
            st.error("No form fields were found in the supplied HTML.")
        else:
            plan = _run(lambda: FormPlanService(path).create_plan(
                application["id"], portal, fields, context_id=context["id"]),
                "Fill plan created for local review")
            if plan:
                st.session_state[f"browser_plan_{application['id']}"] = plan["id"]
    plan_id = st.session_state.get(f"browser_plan_{application['id']}")
    if not plan_id:
        return
    plan = FormPlanService(path).get(plan_id)
    cols = st.columns(4)
    cols[0].metric("Fields", plan["field_count"])
    cols[1].metric("Safe autofill", plan["safe_count"])
    cols[2].metric("Review", plan["review_count"])
    cols[3].metric("Manual", plan["manual_count"])
    if plan["status"] == "REVIEW_REQUIRED" and st.button("Approve fill plan", key=f"approve_plan_{plan_id}"):
        if _run(lambda: FormPlanService(path).approve(plan_id, "Local operator"), "Fill plan approved"):
            st.rerun()
    if plan["status"] in {"APPROVED", "FILLED"}:
        command, language = companion_command(plan_id)
        st.code(command, language=language)
        st.caption("Run this locally. The companion never submits the form.")


def _profile_setup(path: Path) -> None:
    workspace = ProfileWorkspace(path)
    existing = workspace.source_documents()
    st.markdown("""<div class="simple-hero"><div class="eyebrow">Start here</div>
      <h2>Build your applicant profile</h2>
      <div class="muted">Add your CV and any useful academic documents. The app reads them immediately for search,
      then asks you to confirm one readable profile before emails and applications use those facts.</div></div>""",
      unsafe_allow_html=True)
    if existing:
        st.success(f"Found {len(existing)} document{'s' if len(existing) != 1 else ''} already in your Vault. You can use them without uploading again.")
    with st.form("simple_profile_setup"):
        name = st.text_input("Your name", value=workspace.suggested_name(), placeholder="Aakash Priyadarshi")
        focus = st.text_area(
            "What do you want to research?",
            value=workspace.suggested_focus(), height=90,
            placeholder="Reliable AI agents, agent evaluation, RAG reliability and multimodal AI",
            help="A short direction is enough. You can refine it later.",
        )
        files = st.file_uploader(
            "Add CV and supporting documents",
            type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
            max_upload_size=25,
            help="CV, degree, transcript, proposal, statement, publication or certificate. Existing files are reused.",
        )
        submitted = st.form_submit_button("Build my profile", type="primary", use_container_width=True)
    if submitted:
        uploads = [ProfileUpload(file.name, file.getvalue()) for file in files]
        with st.spinner("Reading documents and preparing your profile…"):
            result = _run(lambda: workspace.build(
                name, focus, uploads, api_key=os.getenv("OPENAI_API_KEY", "").strip()))
        if result:
            st.session_state.simple_profile_notice = {
                "message": f"I found {result.facts_in_profile} applicant facts. Review the summary, then choose Use this profile before outreach or applications.",
                "warnings": result.warnings,
            }
            st.rerun()
    st.caption("PDF, DOCX, TXT and Markdown are supported. Highly sensitive identity documents should stay outside this profile builder.")


def _create_intent(path: Path, context: dict, *, key: str) -> None:
    with st.form(key):
        title = st.text_input("Name this search", placeholder="Funded AI PhDs · 2027")
        intent = st.text_area(
            "What are you looking for?", height=105,
            placeholder="Funded 2027 PhD programmes in reliable AI and agent evaluation in the UK, Europe, US and Canada",
        )
        with st.expander("Search settings"):
            countries = st.multiselect("Countries", COUNTRY_CHOICES, key=f"{key}_countries")
            kinds = st.multiselect("Programme types", PROGRAMME_TYPES, format_func=lambda value: value.replace("_", " ").title(), key=f"{key}_kinds")
            funding = st.selectbox("Funding", ("Any", "Funded", "Unknown"), key=f"{key}_funding")
            qs = st.selectbox("QS World Rank", ("Any", "Top 25", "Top 50", "Top 100", "Top 200"), key=f"{key}_qs")
            min_fit = st.slider("Minimum research fit", 0.0, 10.0, 0.0, 0.5, key=f"{key}_fit")
            pages = st.slider("Official pages to check", 1, 12, 8, key=f"{key}_pages")
        submitted = st.form_submit_button("Start search", type="primary", use_container_width=True)
    if not submitted:
        return
    criteria = {"country_codes": _country_codes(countries),
                "programme_types": kinds, "funding": funding,
                "qs_max": int(qs.split()[-1]) if qs != "Any" else None,
                "min_research_fit": min_fit, "max_pages": pages}
    service = OperationService(path)
    def start():
        search = service.create_search(title or intent[:64], intent, context["id"], criteria)
        operation_id = service.queue("PROGRAMME_SEARCH", search_id=search["id"])
        service.launch(operation_id)
        return search
    created = _run(start)
    if created:
        st.session_state.simple_search_id = created["id"]
        st.session_state.simple_intent_id = created["intent_id"]
        st.session_state.simple_nav_target = "Find programmes"
        st.rerun()


@st.fragment(run_every="2s")
def _operation_live(path: Path, *, search_id: int | None = None, application_id: int | None = None) -> None:
    service = OperationService(path)
    operations = [item for item in service.list() if
                  (search_id is None or item["search_id"] == search_id) and
                  (application_id is None or item["application_id"] == application_id)]
    for item in operations[:5]:
        with st.container(border=True):
            st.write(f"**{item['title']}** · {item['status'].replace('_', ' ').title()}")
            st.caption(" · ".join(filter(None, (item["stage"], item["current_item"], item["model_route"]))))
            if item["total_units"]:
                st.progress(min(1., item["completed_units"] / item["total_units"]),
                            text=f"{item['completed_units']}/{item['total_units']} official pages checked · {item['results_found']} saved")
            else:
                st.caption(f"{item['completed_units']} checked · {item['results_found']} saved; finding pages or checking saved records")
            if item["error_summary"]:
                st.warning(item["error_summary"])
            if item["status"] in {"QUEUED", "RUNNING"}:
                if st.button("Stop", key=f"simple_stop_{item['id']}"):
                    _run(lambda: service.stop(item["id"]), "Stop requested. The current page may finish first.")
                    st.rerun(scope="fragment")
            elif item["status"] in {"CANCELLED", "INTERRUPTED", "FAILED"}:
                left, right = st.columns(2)
                if left.button("Resume", key=f"simple_resume_{item['id']}"):
                    if _run(lambda: (service.resume(item["id"]), service.launch(item["id"]))):
                        st.rerun(scope="fragment")
                if right.button("Retry", key=f"simple_retry_{item['id']}"):
                    if _run(lambda: (service.resume(item["id"], retry=True), service.launch(item["id"]))):
                        st.rerun(scope="fragment")
            with st.expander("Activity"):
                for event in service.events(item["id"])[-12:]:
                    st.caption(f"{event['created_at']} · {event['event_text']}")


def _candidate(orchestrator: ProgrammeOrchestrator, candidate: dict) -> None:
    payload = candidate["payload"]
    title = payload.get("programme") or "Programme name needs checking"
    institution = payload.get("university") or "Institution needs checking"
    with st.container(border=True):
        left, right = st.columns([4, 1])
        left.markdown(f"### {html.escape(title)}")
        left.caption(" · ".join(filter(None, (institution, payload.get("degree"), payload.get("intake"), _qs_caption(payload)))))
        right.markdown(f"**{candidate['review_state'].title()}**")
        facts = st.columns(3)
        facts[0].metric("Deadline", payload.get("deadline") or "Unknown")
        facts[1].metric("Research fit", f"{payload.get('research_fit', 0):.1f}/10")
        facts[2].metric("Missing details", len(payload.get("unknown_fields", [])))
        if payload.get("funding"):
            st.write("**Funding:**", payload["funding"])
        relevant = payload.get("relevant_applicant_experience", [])
        if relevant:
            st.write("**Why it matches your profile:**", relevant[0])
        if candidate["review_state"] != "ACCEPTED":
            st.caption("Adding accepts the university and programme. Deadline and supervisor contact stay unverified.")
            add, save, dismiss, remove = st.columns([1.25, 1, 1, 1])
            if add.button("Add application", key=f"simple_add_{candidate['id']}", type="primary"):
                confirmed = {key: payload.get(key) for key in (
                    "university", "programme", "department", "degree", "intake",
                ) if payload.get(key) is not None}
                if _run(lambda: orchestrator.accept_candidate(
                        candidate["id"], "Local operator", cycle=payload.get("intake"),
                        field_overrides=confirmed), "Application added"):
                    st.rerun()
            if save.button("Save", key=f"simple_save_{candidate['id']}"):
                if _run(lambda: orchestrator.review_candidate(candidate["id"], "SHORTLISTED", "Local operator"),
                        "Saved to shortlist"):
                    st.rerun()
            if dismiss.button("Dismiss", key=f"simple_dismiss_{candidate['id']}"):
                if _run(lambda: orchestrator.review_candidate(candidate["id"], "REJECTED", "Local operator"),
                        "Candidate dismissed"):
                    st.rerun()
            if remove.button("Remove", key=f"simple_remove_{candidate['id']}"):
                if _run(lambda: RecordControls(orchestrator.db_path).archive_candidate(candidate["id"]),
                        "Result removed; you can restore it later"):
                    st.rerun()
        with st.expander("Details and source"):
            for label, value in (
                ("Eligibility", payload.get("eligibility")), ("English requirement", payload.get("english_requirements")),
                ("Supervisor contact", payload.get("supervisor_contact_policy")),
                ("Application route", payload.get("official_application_url")),
            ):
                st.write(f"**{label}:** {value or 'Unknown'}")
            st.link_button("Open official page", candidate["canonical_url"])


def _home(path: Path, context: dict) -> None:
    orchestrator = ProgrammeOrchestrator(path)
    applications = Ledger(path).list_applications()
    owner = context["context"]["owner_name"].split()[0]
    st.markdown(f"""<div class="simple-hero"><div class="eyebrow">Application workspace</div>
      <h2>Welcome back, {html.escape(owner)}</h2><div class="muted">Tell the app what you want to find.
      Your CV and supporting documents are already available as context.</div></div>""", unsafe_allow_html=True)
    _profile_banner(path, context)
    _create_intent(path, context, key="simple_home_intent")
    active = [operation for operation in OperationService(path).list(10)
              if operation["status"] in {"QUEUED", "RUNNING", "CANCEL_REQUESTED"}]
    if active:
        st.subheader("Running now")
        _operation_live(path)
    waiting = orchestrator.list_candidates(states=("NEW", "SHORTLISTED"))
    st.subheader("Your next step")
    if waiting:
        st.info(f"Review {len(waiting)} programme candidate{'s' if len(waiting) != 1 else ''}.")
        for item in waiting[:2]:
            _candidate(orchestrator, item)
    elif applications:
        app = applications[0]
        st.success(f"Continue {app['institution']} · {app['application_name']}")
        st.write(app["next_action"] or "Review missing requirements and documents.")
    else:
        st.caption("Start with a research intent above, or analyse a programme URL in Find programmes.")


def _discover(path: Path, context: dict) -> None:
    orchestrator = ProgrammeOrchestrator(path)
    st.title("Find programmes")
    st.caption("Search from your research goals or paste one official programme page.")
    notice = st.session_state.pop("simple_discovery_notice", None)
    if notice:
        st.success(notice)
    _create_intent(path, context, key="simple_discover_intent")
    searches = OperationService(path).list_searches()
    search_options = {item["id"]: item for item in searches}
    search_id = st.selectbox("Search results", [None, *search_options],
        format_func=lambda value: "All searches" if value is None else search_options[value]["title"],
        key="simple_search_id")
    if search_id is not None:
        st.caption("Saved settings: " + ", ".join(f"{key.replace('_', ' ')}: {value}"
                   for key, value in search_options[search_id]["criteria"].items() if value not in (None, [], "Any", 0)))
    _operation_live(path, search_id=search_id)
    st.divider()
    filters = st.columns(4)
    countries = filters[0].multiselect("Countries", COUNTRY_CHOICES)
    qs_choice = filters[1].selectbox("QS World Rank", ["Any", "Top 25", "Top 50", "Top 100", "Top 150", "Top 200"])
    funded_only = filters[2].checkbox("Funded only")
    min_fit = filters[3].slider("Minimum research fit", 0.0, 10.0, 0.0, 0.5)
    kinds = st.multiselect("Programme types", PROGRAMME_TYPES,
                           format_func=lambda value: value.replace("_", " ").title())
    extra = {"funded_only": funded_only or None, "min_research_fit": min_fit or None}
    if countries:
        extra["country_codes"] = _country_codes(countries)
    if qs_choice != "Any":
        extra["qs_max"] = int(qs_choice.split()[-1])
    extra = {key: value for key, value in extra.items() if value}
    with st.form("simple_url"):
        url = st.text_input("Official programme URL", placeholder="https://university.example/phd-programme")
        analyse = st.form_submit_button("Analyse page")
    if analyse:
        result = _run(lambda: orchestrator.analyse_url(url, context["id"],
            intent_id=st.session_state.get("simple_intent_id")))
        if result:
            st.session_state.simple_url_result = result
            if result.get("status") == "CANDIDATE_READY":
                st.session_state.simple_discovery_notice = "Programme page analysed. Results are below."
            st.rerun()
    result = st.session_state.get("simple_url_result")
    if result and result.get("status") == "HUMAN_INPUT_REQUIRED":
        st.warning("That page could not be read automatically. Paste the useful text below.")
        with st.form("simple_paste"):
            text = st.text_area("Programme page text", height=170)
            pasted = st.form_submit_button("Use pasted text")
        if pasted and _run(lambda: orchestrator.analyse_supplied(
                url or result.get("url", ""), text, context["id"],
                intent_id=st.session_state.get("simple_intent_id"))):
            st.rerun()
    _live_results(path, search_options[search_id] if search_id else None, extra, kinds)


@st.fragment(run_every="2s")
def _live_results(path: Path, search: dict | None, extra: dict, kinds: list[str]) -> None:
    orchestrator = ProgrammeOrchestrator(path)
    st.subheader("Results")
    candidates = orchestrator.list_candidates(intent_id=search["intent_id"] if search else None,
                                              states=("NEW", "SHORTLISTED", "ACCEPTED"), extra_filters=extra)
    if search:
        candidates = filter_programmes(candidates, search["criteria"])
    if kinds:
        candidates = filter_programmes(candidates, {"programme_types": kinds})
    candidates = sorted(candidates, key=lambda item: (-float(item["payload"].get("research_fit") or 0), qs_sort_key(item["payload"])))
    st.caption(f"{len(candidates)} visible result{'s' if len(candidates) != 1 else ''}")
    if not candidates:
        st.caption("No programme results yet.")
    removable = [item for item in candidates if item["review_state"] != "ACCEPTED"]
    if removable:
        with st.expander("Remove several results"):
            selected = st.multiselect("Select results", [item["id"] for item in removable],
                format_func=lambda candidate_id: next(
                    f"{item['payload'].get('university') or 'Unknown'} · {item['payload'].get('programme') or item['canonical_url']}"
                    for item in removable if item["id"] == candidate_id))
            if st.button(f"Remove {len(selected)} selected results", disabled=not selected):
                if _run(lambda: RecordControls(path).archive_candidates(selected), "Results removed; restoration is available below"):
                    st.rerun()
    layout = st.radio("Arrange results by", ("Country", "University", "Research fit"), horizontal=True)
    page_size = 10
    page_count = max(1, (len(candidates) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=page_count, value=1,
                           key="simple_results_page")
    visible = candidates[(page - 1) * page_size: page * page_size]
    if layout == "Research fit":
        for candidate in visible:
            _candidate(orchestrator, candidate)
    else:
        for group, children in group_programmes(visible, view=layout).items():
            st.subheader(group)
            for subgroup, items in children.items():
                st.markdown(f"**{subgroup}**")
                for candidate in items:
                    _candidate(orchestrator, candidate)
    with st.expander("Removed results"):
        archived = [item for item in orchestrator.list_candidates(states=("NEW", "SHORTLISTED", "REJECTED"),
                    include_archived=True) if item.get("archived_at")]
        for item in archived:
            st.write(item["payload"].get("programme") or item["canonical_url"])
            if st.button("Restore result", key=f"restore_candidate_{item['id']}"):
                if _run(lambda item=item: RecordControls(path).archive_candidate(item["id"], False)):
                    st.rerun()


def _applications(path: Path, context: dict) -> None:
    orchestrator = ProgrammeOrchestrator(path)
    ledger = Ledger(path)
    controls = RecordControls(path)
    applications = ledger.list_applications()
    st.title("Applications")
    st.caption("One place for deadlines, missing items and the next action.")
    if not applications:
        st.info("Add a programme from Find programmes and it will appear here.")
    view = st.radio("Arrange applications by", ("Deadline", "Country", "University"), horizontal=True)
    if view != "Deadline":
        applications = [item for children in group_programmes(applications, view=view).values()
                        for items in children.values() for item in items]
    for application in applications:
        overview = orchestrator.application_overview(application["id"])
        readiness = overview["readiness"]
        with st.container(border=True):
            st.markdown(f"### {html.escape(application['institution'])} · {html.escape(application['application_name'])}")
            st.caption(" · ".join(filter(None, (
                application["cycle"],
                _qs_caption(application),
                f"deadline {application['nearest_deadline'] or 'not confirmed'}",
            ))))
            counts = st.columns(3)
            counts[0].metric("Ready", f"{readiness['required_complete']}/{readiness['required_total']}")
            counts[1].metric("Need checking", readiness["unknown_requirements"])
            counts[2].metric("Open tasks", len(overview["open_tasks"]))
            st.write("**Next:**", overview["next_action"])
            if st.button("Prepare this application", key=f"simple_prepare_{application['id']}", type="primary"):
                if context.get("trust_level") != "TRUSTED":
                    st.warning("Confirm your profile with Use this profile before preparing application documents.")
                else:
                    prepared = _run(lambda: orchestrator.prepare_application(
                        application["id"], context_id=context["id"]))
                    if prepared:
                        st.session_state[f"simple_prepared_{application['id']}"] = prepared
            prepared = st.session_state.get(f"simple_prepared_{application['id']}")
            if prepared and prepared["blocking"]:
                st.warning("Still needed: " + " · ".join(prepared["blocking"]))
            with st.expander("Edit details and fill gaps"):
                with st.form(f"edit_app_{application['id']}"):
                    cycle = st.text_input("Application cycle", value=application["cycle"] or "")
                    status = st.selectbox("Status", APPLICATION_STATUSES,
                        index=APPLICATION_STATUSES.index(application["status"]))
                    portal = st.text_input("Official application portal", value=application["portal_url"] or "")
                    funding_state = st.selectbox("Funding status", ("UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING", "CONFIRMED"),
                        index=("UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING", "CONFIRMED").index(application["funding_state"]))
                    eligibility_state = st.selectbox("Eligibility status", ("UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING"),
                        index=("UNKNOWN", "ELIGIBLE", "INELIGIBLE", "PENDING").index(application["eligibility_state"]))
                    contact_state = st.selectbox("Professor contact", ("UNKNOWN", "NOT_CONTACTED", "CONTACTED", "REPLIED", "NOT_REQUIRED"),
                        index=("UNKNOWN", "NOT_CONTACTED", "CONTACTED", "REPLIED", "NOT_REQUIRED").index(application["supervisor_contact_state"]))
                    details_source = st.text_input("Official source for eligibility or funding confirmation")
                    next_action = st.text_input("Next action", value=application["next_action"] or "")
                    notes = st.text_area("Your notes", value=application["owner_notes"] or "")
                    save = st.form_submit_button("Save application")
                if save and _run(lambda: controls.edit_application(application["id"], cycle=cycle,
                          status=status, portal_url=portal, next_action=next_action, owner_notes=notes,
                          funding_state=funding_state, eligibility_state=eligibility_state,
                          supervisor_contact_state=contact_state, source_url=details_source or None),
                          "Application updated"):
                    st.rerun()
                if application["programme_id"]:
                    programme = ledger.get("programmes", application["programme_id"])
                    with st.form(f"edit_programme_{application['id']}"):
                        university = st.text_input("University", value=programme["university"] or "")
                        name = st.text_input("Programme", value=programme["programme_name"] or "")
                        country = st.text_input("Country", value=programme["country"] or "")
                        department = st.text_input("Department", value=programme["department"] or "")
                        degree = st.text_input("Degree type", value=programme["degree_type"] or "")
                        link = st.text_input("Programme URL", value=programme["programme_url"] or "")
                        source = st.text_input("Official source for confirmed corrections")
                        verified = st.checkbox("I checked these changes against the official source")
                        save_programme = st.form_submit_button("Save programme")
                    if save_programme and _run(lambda: controls.edit_programme(programme["id"],
                            university=university, programme_name=name, country=country,
                            department=department, degree_type=degree, programme_url=link,
                            source_url=source or None, verified=verified), "Programme updated"):
                        st.rerun()
                with st.form(f"deadline_{application['id']}"):
                    due = st.text_input("Add deadline (YYYY-MM-DD)", placeholder="2027-01-15")
                    deadline_source = st.text_input("Official deadline source URL")
                    checked = st.checkbox("I verified the deadline on this page")
                    add_deadline = st.form_submit_button("Add deadline")
                if add_deadline and _run(lambda: controls.set_deadline(application["id"], due,
                             source_url=deadline_source, verified=checked), "Deadline recorded"):
                    st.rerun()
                for deadline in ledger.list_deadlines(application["id"]):
                    st.caption(f"{deadline['due_at']} · {deadline['verification_state']} · {deadline['source_url']}")
            with st.expander("Continue in browser"):
                _browser_continue(path, application, context)
            with st.expander("Requirements"):
                for requirement in overview["requirements"]:
                    state = requirement.get("document_state") or requirement["requirement_state"]
                    st.write(f"- {requirement['original_label']}: **{state.replace('_', ' ').title()}**")
            with st.expander("Remove application"):
                impact = controls.impact(application["id"])
                st.caption(f"Hides this application from your workspace. Keeps {impact['professors']} professor links, "
                           f"{impact['tasks']} tasks and {impact['documents']} document links for restoration.")
                if st.button("Archive application", key=f"archive_app_{application['id']}"):
                    if _run(lambda: controls.archive_application(application["id"]), "Application archived"):
                        st.rerun()
    archived = [item for item in ledger.list_applications(include_archived=True) if item["archived_at"]]
    with st.expander(f"Archived applications ({len(archived)})"):
        for application in archived:
            st.write(f"{application['institution']} · {application['application_name']}")
            if st.button("Restore application", key=f"restore_app_{application['id']}"):
                if _run(lambda application=application: controls.archive_application(application["id"], False)):
                    st.rerun()


def _people(path: Path, context: dict) -> None:
    orchestrator = ProgrammeOrchestrator(path)
    applications = Ledger(path).list_applications()
    st.title("People")
    st.caption("Find supervisors whose current research overlaps with your demonstrated experience.")
    if not applications:
        st.info("Add an application before looking for supervisors.")
        return
    mapping = {row["id"]: row for row in applications}
    application_id = st.selectbox("Application", list(mapping),
        format_func=lambda value: f"{mapping[value]['institution']} · {mapping[value]['application_name']}")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        st.caption("Official faculty search needs the configured research provider; saved professors are shown below.")
    if st.button("Find relevant supervisors", type="primary",
                 disabled=not bool(os.getenv("OPENAI_API_KEY", "").strip())):
        service = OperationService(path)
        operation_id = _run(lambda: service.queue("FACULTY_DISCOVERY", application_id=application_id,
                                                 context_id=context["id"]))
        if operation_id and _run(lambda: service.launch(operation_id)):
            st.rerun()
    _operation_live(path, application_id=application_id)
    _live_faculty_leads(path, mapping[application_id]["institution"])
    cards = _run(lambda: orchestrator.professor_cards(application_id, context["id"])) or []
    if not cards:
        st.caption("No reviewed supervisor matches yet. Use Find relevant supervisors to check current records.")
    with connect(path) as db:
        linked = {row[0] for row in db.execute(
            "SELECT faculty_profile_id FROM application_faculty WHERE application_id=?", (application_id,))}
        packages = [dict(row) for row in db.execute("""SELECT * FROM application_packages
            WHERE application_id=? AND context='FACULTY_OUTREACH' AND status='READY' ORDER BY version_number DESC""",
            (application_id,))]
    for card in cards:
        with st.container(border=True):
            st.markdown(f"### {html.escape(card['name'])}")
            st.caption(" · ".join(filter(None, (card["department"], card["lab"], card["institution"]))))
            st.metric("Research fit", f"{card['research_fit']:.1f}/10")
            st.write(card["research_topics"] or "Research topics need checking.")
            if card["relevant_applicant_experience"]:
                st.write("**Your relevant experience:**", card["relevant_applicant_experience"][0])
            if card["unknowns"]:
                st.caption("Still checking: " + ", ".join(card["unknowns"]))
            if card["faculty_id"] not in linked:
                if st.button("Add to application", key=f"simple_person_{card['faculty_id']}"):
                    evidence = card["evidence_ids"][0] if card["evidence_ids"] else None
                    if _run(lambda: Discovery(path).link_to_application(
                            application_id, faculty_id=card["faculty_id"], evidence_id=evidence), "Supervisor added"):
                        st.rerun()
            elif packages and card["contact_policy_state"] == "PASS":
                if context.get("trust_level") != "TRUSTED":
                    st.caption("Confirm your profile before drafting a professor email.")
                elif st.button("Draft email", key=f"simple_email_{card['faculty_id']}", type="primary"):
                    _run(lambda: OutreachService(path).prepare(
                        card["faculty_id"], application_id, packages[0]["id"],
                        context["profile_version_id"], context["research_track_version_id"]),
                        "Email draft prepared")
            else:
                st.caption("The app will enable email drafting when contact policy and attachments are ready.")


@st.fragment(run_every="2s")
def _live_faculty_leads(path: Path, institution: str) -> None:
    pending = [item for item in Discovery(path).list_faculty_candidates()
               if item["review_state"] == "NEW" and item["institution"].casefold() == institution.casefold()]
    if pending:
        st.subheader(f"New professor pages ({len(pending)})")
        st.caption("These are source-backed leads. Check the official page before relying on affiliation or research fit.")
        for candidate in pending[:20]:
            with st.container(border=True):
                st.write(f"**{candidate['name']}** · {candidate['department'] or 'Department unknown'}")
                if candidate["profile_url"]:
                    st.link_button("Check official profile", candidate["profile_url"])
                if st.button("Dismiss lead", key=f"dismiss_faculty_{candidate['id']}"):
                    if _run(lambda candidate=candidate: Discovery(path).review_faculty_candidate(
                            candidate["id"], dismiss=True), "Lead dismissed"):
                        st.rerun()


def _universities(path: Path) -> None:
    st.title("Universities")
    st.caption("Browse your programme applications by country and university. Unknown countries stay visible.")
    applications = Ledger(path).list_applications()
    view = st.radio("Group by", ("Country", "University"), horizontal=True, key="university_group")
    for group, children in group_programmes(applications, view=view).items():
        st.subheader(group)
        for subgroup, items in children.items():
            with st.expander(f"{subgroup} · {len(items)} application{'s' if len(items) != 1 else ''}"):
                for item in items:
                    st.markdown(f"**{html.escape(item['application_name'])}**")
                    st.caption(" · ".join(filter(None, (item["cycle"], item["status"].replace("_", " ").title(),
                                                     _qs_caption(item), item["nearest_deadline"]))))
                    if item["portal_url"]:
                        st.link_button("Open portal", item["portal_url"])
    if not applications:
        st.info("Add a programme to see it here.")


def _searches(path: Path, context: dict) -> None:
    st.title("Searches")
    st.caption("Keep separate searches for each topic or cycle. Change saved filters and run them again.")
    _create_intent(path, context, key="simple_saved_intent")
    service = OperationService(path)
    for search in service.list_searches():
        with st.container(border=True):
            st.markdown(f"### {html.escape(search['title'])}")
            st.caption(search["intent_text"])
            with st.form(f"search_edit_{search['id']}"):
                title = st.text_input("Search name", value=search["title"])
                criteria = search["criteria"]
                countries = st.multiselect("Countries", COUNTRY_CHOICES,
                    default=_country_selections(criteria.get("country_codes") or []))
                kinds = st.multiselect("Programme types", PROGRAMME_TYPES,
                                       default=criteria.get("programme_types") or [])
                funding = st.selectbox("Funding", ("Any", "Funded", "Unknown"),
                                       index=("Any", "Funded", "Unknown").index(criteria.get("funding") or "Any"))
                rank_options = ("Any", "Top 25", "Top 50", "Top 100", "Top 150", "Top 200")
                current_rank = f"Top {criteria['qs_max']}" if criteria.get("qs_max") else "Any"
                qs = st.selectbox("QS World Rank", rank_options,
                                  index=rank_options.index(current_rank) if current_rank in rank_options else 0)
                min_fit = st.slider("Minimum research fit", 0.0, 10.0,
                                    float(criteria.get("min_research_fit") or 0), 0.5)
                pages = st.slider("Official pages to check", 1, 12, int(criteria.get("max_pages") or 8))
                save = st.form_submit_button("Save settings")
            if save:
                updated = {**criteria, "country_codes": _country_codes(countries),
                           "programme_types": kinds, "funding": funding, "max_pages": pages,
                           "qs_max": int(qs.split()[-1]) if qs != "Any" else None,
                           "min_research_fit": min_fit}
                if _run(lambda: service.update_search(search["id"], title=title, criteria=updated),
                        "Search settings saved"):
                    st.rerun()
            left, right = st.columns(2)
            if left.button("Run again", key=f"rerun_search_{search['id']}"):
                operation_id = _run(lambda: service.queue("PROGRAMME_SEARCH", search_id=search["id"]))
                if operation_id and _run(lambda: service.launch(operation_id)):
                    st.rerun()
            if right.button("Archive search", key=f"archive_search_{search['id']}"):
                if _run(lambda: service.archive_search(search["id"])):
                    st.rerun()
    with st.expander("Archived searches"):
        for search in service.list_searches(include_archived=True):
            if search["status"] == "ARCHIVED":
                st.write(search["title"])
                if st.button("Restore search", key=f"restore_search_{search['id']}"):
                    if _run(lambda search=search: service.archive_search(search["id"], False)):
                        st.rerun()


def _operations(path: Path) -> None:
    st.title("Operations")
    st.caption("See current work, stop after the current page, or resume from saved progress.")
    _operation_live(path)


def _documents(path: Path, context: dict) -> None:
    workspace = ProfileWorkspace(path)
    latest = workspace.latest_summary()
    st.title("My documents")
    st.caption("Add a document once. Search can use the extracted summary immediately; confirm the profile before using facts in applications.")
    _profile_banner(path, context)
    if latest:
        summary_path, summary = latest
        with st.expander("Your readable applicant profile", expanded=True):
            st.markdown(summary)
            st.download_button("Save Markdown copy", summary.encode(), file_name=summary_path.name,
                               mime="text/markdown")
    with st.container(border=True):
        st.subheader("Add or update profile documents")
        name = st.text_input("Your name", value=workspace.suggested_name(), key="simple_docs_name")
        focus = st.text_area("Research direction", value=workspace.suggested_focus(), height=80,
                             key="simple_docs_focus")
        files = st.file_uploader("Choose files", type=["pdf", "docx", "txt", "md"],
                                 accept_multiple_files=True, max_upload_size=25,
                                 key="simple_docs_upload")
        if st.button("Update my profile", type="primary"):
            uploads = [ProfileUpload(file.name, file.getvalue()) for file in files]
            result = _run(lambda: workspace.build(
                name, focus, uploads, api_key=os.getenv("OPENAI_API_KEY", "").strip()),
                "Profile updated")
            if result:
                st.session_state.simple_profile_notice = {
                    "message": f"Profile updated with {result.facts_in_profile} source-backed facts.",
                    "warnings": result.warnings,
                }
                st.rerun()
    st.subheader("Stored source documents")
    documents = workspace.source_documents()
    if not documents:
        st.caption("No documents stored yet.")
    for document in documents:
        with st.container(border=True):
            st.write(f"**{document['original_filename']}**")
            document_type = document["document_type"].replace("_", " ").title()
            if document["approval_state"] == "APPROVED":
                st.caption(document_type + " · Included in your profile")
            else:
                st.caption(document_type + " · Stored safely · Academic-document check needed")


def render_workspace(db_path: Path) -> None:
    _css()
    workspace = ProfileWorkspace(db_path)
    context = ApplicantResearchContextService(db_path).latest()
    if not context:
        st.title("PhD Application Assistant")
        _profile_setup(db_path)
        return
    track = context["context"]["research_track"]["title"]
    notice = st.session_state.pop("simple_profile_notice", None)
    if notice:
        st.success(notice["message"])
        for warning in notice["warnings"]:
            st.warning(warning)
    st.sidebar.markdown("### PhD Assistant")
    if context.get("trust_level") == "TRUSTED":
        st.sidebar.markdown('<span class="ready-pill">Profile confirmed</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="ready-pill">Review profile</span>', unsafe_allow_html=True)
    st.sidebar.caption(track)
    nav_target = st.session_state.pop("simple_nav_target", None)
    if nav_target in PAGES:
        st.session_state.simple_nav = nav_target
    page = st.sidebar.radio("Navigation", PAGES, key="simple_nav", label_visibility="collapsed")
    if page == "Home":
        _home(db_path, context)
    elif page == "Find programmes":
        _discover(db_path, context)
    elif page == "Applications":
        _applications(db_path, context)
    elif page == "Universities":
        _universities(db_path)
    elif page == "People":
        _people(db_path, context)
    elif page == "Searches":
        _searches(db_path, context)
    elif page == "Operations":
        _operations(db_path)
    else:
        _documents(db_path, context)
