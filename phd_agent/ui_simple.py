"""Simple task-based Streamlit workspace for one PhD applicant."""

from __future__ import annotations

import html
import os
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

from phd_agent.applicant_context import ApplicantResearchContextService
from phd_agent.application_rescan import FIELDS, HUMAN_MESSAGE, SCAN_FULL, SCAN_MISSING, ApplicationRescan
from phd_agent.browser_worker import FormPlanService, companion_command, fields_from_html
from phd_agent.config import load_settings
from phd_agent.db import connect
from phd_agent.discovery import Discovery
from phd_agent.faculty_research import FacultyResearch
from phd_agent.ledger import APPLICATION_STATUSES, Ledger
from phd_agent.operations import OperationService
from phd_agent.orchestration import ProgrammeOrchestrator
from phd_agent.outreach import OutreachService
from phd_agent.planning import DailyPlanner, EVENT_TYPES
from phd_agent.profile_workspace import ProfileUpload, ProfileWorkspace
from phd_agent.record_controls import PROGRAMME_TYPES, RecordControls, filter_programmes, group_programmes
from phd_agent.university_enrichment import REGION_COUNTRY_CODES, qs_sort_key, resolve_country_label


PAGES = ("Today", "Find programmes", "Applications", "Universities", "People", "My documents", "Searches", "Operations", "Calendar")
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
def _operation_live(path: Path, *, search_id: int | None = None, application_id: int | None = None,
                    operation_types: tuple[str, ...] | None = None) -> None:
    service = OperationService(path)
    operations = service.list(search_id=search_id, application_id=application_id, operation_types=operation_types)
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
            elif item["status"] in {"CANCELLED", "INTERRUPTED", "FAILED", "PAUSED"}:
                left, right = st.columns(2)
                if left.button("Resume", key=f"simple_resume_{item['id']}"):
                    if _run(lambda: (service.resume(item["id"]), service.launch(item["id"]))):
                        st.rerun(scope="fragment")
                if right.button("Retry", key=f"simple_retry_{item['id']}"):
                    if _run(lambda: (service.resume(item["id"], retry=True), service.launch(item["id"]))):
                        st.rerun(scope="fragment")
            if item["operation_type"] in {SCAN_MISSING, SCAN_FULL} and item["status"] == "PAUSED":
                st.warning(HUMAN_MESSAGE)
                pending = [url for url in item["checkpoint"].get("human_input") or [] if url not in item["checkpoint"].get("supplied", {})]
                page_url = pending[-1] if pending else ""
                if page_url.startswith("http"):
                    st.link_button("Open page", page_url)
                with st.form(f"scan_supply_{item['id']}"):
                    pasted = st.text_area("Paste page text")
                    upload = st.file_uploader("Upload saved HTML or PDF", type=["html", "htm", "pdf"])
                    supplied = st.form_submit_button("Use this page")
                if supplied and page_url:
                    def use_page(item=item, pasted=pasted, upload=upload, page_url=page_url):
                        scanner = ApplicationRescan(path)
                        if upload is not None:
                            name = (upload.name or "").casefold()
                            method = "UPLOADED_PDF" if name.endswith(".pdf") else "UPLOADED_HTML"
                            scanner.supply(item["id"], page_url, upload.getvalue(), method=method)
                        else:
                            scanner.supply(item["id"], page_url, pasted)
                        service.resume(item["id"])
                        service.launch(item["id"])
                    if _run(use_page, "Page added. Scan resumed."):
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


def focus_keys_for_today_item(item: dict) -> dict[str, int | None]:
    """Keep each Today destination from writing another page's application filter."""
    application_id = item.get("application_id")
    page = item.get("page")
    return {
        "simple_focus_application_id": application_id if application_id and page == "Applications" else None,
        "simple_focus_people_application_id": application_id if application_id and page == "People" else None,
    }


def _home(path: Path, context: dict) -> None:
    planner = DailyPlanner(path)
    dashboard = planner.dashboard()
    counts = dashboard["counts"]
    owner = context["context"]["owner_name"].split()[0]
    st.markdown(f"""<div class="simple-hero"><div class="eyebrow">Application workspace</div>
      <h2>Welcome back, {html.escape(owner)}</h2><div class="muted">Here is what needs attention today.</div></div>""",
      unsafe_allow_html=True)
    _profile_banner(path, context)
    scanner = ApplicationRescan(path)
    notes = scanner.notifications()[:3]
    for note in notes:
        st.info(note["message"])
    if notes:
        scanner.mark_notifications_read([note["id"] for note in notes])
    summary = (("Deadlines · 14 days", "deadlines_14_days"), ("Professor replies", "new_replies"),
               ("Programme results", "programme_results"), ("Missing documents", "missing_required"))
    for column, (label, key) in zip(st.columns(4), summary):
        column.metric(label, counts[key])
    st.caption(f"{counts['unknown_requirements']} requirements need checking · "
               f"{counts['document_alerts']} document alerts · {counts['active_operations']} active operations")
    st.subheader("Needs attention")
    if not dashboard["actions"]:
        st.success("Nothing urgent is recorded. Start a search or review your applications when ready.")
    for index, item in enumerate(dashboard["actions"][:10]):
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{html.escape(item['title'])}**")
            left.caption(f"{item['priority'].title()} · {item['detail']}")
            if right.button("Open", key=f"today_open_{index}"):
                st.session_state.simple_nav_target = item["page"]
                for key, value in focus_keys_for_today_item(item).items():
                    if value is None:
                        st.session_state.pop(key, None)
                    else:
                        st.session_state[key] = value
                st.rerun()
            if item.get("scan_application_id") and st.button("Scan now", key=f"today_scan_{index}"):
                def start_scan(application_id=item["scan_application_id"]):
                    operation_id = ApplicationRescan(path).queue(application_id)
                    OperationService(path).launch(operation_id)
                if _run(start_scan, "Scan started"):
                    st.session_state.simple_nav_target = "Operations"
                    st.rerun()
            if item.get("task_id") and st.button("Mark task done", key=f"today_complete_{item['task_id']}"):
                if _run(lambda item=item: planner.complete_task(item["task_id"]), "Task completed"):
                    st.rerun()
    if counts["active_operations"]:
        st.subheader("Running now")
        _operation_live(path)
    with st.expander("Start another programme search"):
        _create_intent(path, context, key="simple_home_intent")


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
        archived = [item for item in orchestrator.list_candidates(
                    intent_id=search["intent_id"] if search else None,
                    states=("NEW", "SHORTLISTED", "REJECTED"),
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
    _operation_live(path, operation_types=(SCAN_MISSING, SCAN_FULL))
    if not applications:
        st.info("Add a programme from Find programmes and it will appear here.")
    application_map = {item["id"]: item for item in applications}
    focus = st.session_state.pop("simple_focus_application_id", None)
    if focus in application_map:
        st.session_state.simple_application_filter = focus
    selected = st.selectbox("Show application", [None, *application_map],
        format_func=lambda value: "All applications" if value is None else
            f"{application_map[value]['institution']} · {application_map[value]['application_name']}",
        key="simple_application_filter")
    if selected is not None:
        applications = [application_map[selected]]
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
            scan_summary = ApplicationRescan(path).completeness(application["id"])
            st.caption(f"Application data {scan_summary['known']}/{scan_summary['total']} known · "
                       f"Requirements {scan_summary['requirements_known']}/{scan_summary['requirements_total']} resolved")
            if scan_summary["missing"]:
                st.write(f"Missing details: {len(scan_summary['missing'])}")
                for group, items in scan_summary["groups"].items():
                    st.caption(f"{group}: " + ", ".join(item["label"] for item in items))
                scan_col, edit_col = st.columns(2)
                if scan_col.button("Scan missing details", key=f"scan_missing_{application['id']}"):
                    def start_missing(application_id=application["id"]):
                        operation_id = ApplicationRescan(path).queue(application_id)
                        OperationService(path).launch(operation_id)
                    if _run(start_missing, "Scan started"):
                        st.rerun()
                if edit_col.button("Edit manually", key=f"edit_manual_{application['id']}"):
                    st.session_state[f"manual_edit_{application['id']}"] = True
                    st.rerun()
            report = ApplicationRescan(path).latest_report(application["id"])
            if report and report["summary"]:
                summary = report["summary"]
                st.caption(f"Last scan: {len(summary.get('resolved', []))} resolved · "
                           f"{len(summary.get('conflicts', []))} conflicts · "
                           f"{len(summary.get('still_unknown', []))} still unknown · "
                           f"{summary.get('sources_checked', 0)} sources checked")
            for conflict in ApplicationRescan(path).conflicts(application["id"]):
                st.warning(f"Possible change detected · {FIELDS[conflict['field_name']]['label']}")
                st.caption(f"Current: {conflict['old_value'] or 'Unknown'} · Source now says: {conflict['new_value']}")
                accept, keep, unresolved = st.columns(3)
                if accept.button("Accept new", key=f"accept_conflict_{conflict['id']}"):
                    if _run(lambda conflict=conflict: ApplicationRescan(path).resolve_conflict(conflict["id"], "accept"),
                            "New value accepted"):
                        st.rerun()
                if keep.button("Keep current", key=f"keep_conflict_{conflict['id']}"):
                    if _run(lambda conflict=conflict: ApplicationRescan(path).resolve_conflict(conflict["id"], "keep"),
                            "Current value kept"):
                        st.rerun()
                if unresolved.button("Mark unresolved", key=f"unresolved_conflict_{conflict['id']}"):
                    if _run(lambda conflict=conflict: ApplicationRescan(path).resolve_conflict(conflict["id"], "unresolved"),
                            "Left unresolved"):
                        st.rerun()
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
            with st.expander("More actions"):
                if st.button("Full refresh from official sources", key=f"full_refresh_{application['id']}"):
                    def start_refresh(application_id=application["id"]):
                        operation_id = ApplicationRescan(path).queue(application_id, mode="FULL")
                        OperationService(path).launch(operation_id)
                    if _run(start_refresh, "Full refresh started"):
                        st.rerun()
                past = ApplicationRescan(path).history(application["id"])
                if past:
                    st.caption("Application scans")
                    for item in past[:5]:
                        resolved = len(item["summary"].get("resolved", []))
                        conflicts = len(item["summary"].get("conflicts", []))
                        st.caption(f"{item['created_at'][:10]} · {resolved} fields resolved · {conflicts} conflicts")
            with st.expander("Edit manually", expanded=bool(st.session_state.get(f"manual_edit_{application['id']}"))):
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
    focus = st.session_state.pop("simple_focus_people_application_id", None)
    if not applications:
        st.info("Add an application before looking for supervisors.")
        return
    mapping = {row["id"]: row for row in applications}
    if focus in mapping:
        st.session_state.simple_people_application = focus
    elif st.session_state.get("simple_people_application") not in mapping:
        st.session_state.pop("simple_people_application", None)
    application_id = st.selectbox("Application", list(mapping),
        format_func=lambda value: f"{mapping[value]['institution']} · {mapping[value]['application_name']}",
        key="simple_people_application")
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
    _live_faculty_leads(path, mapping[application_id]["institution"], application_id, context["id"])
    cards = _run(lambda: orchestrator.professor_cards(application_id, context["id"])) or []
    if not cards:
        st.caption("No reviewed supervisor matches yet. Use Find relevant supervisors to check current records.")
    research = FacultyResearch(path)
    topics = sorted({topic.strip() for card in cards for topic in
                     (card["research_topics"] or "").split(",") if topic.strip()})
    col_sort, col_topic = st.columns(2)
    order = col_sort.selectbox("Sort professors", ("Research Fit", "Name", "Most recent research", "Decision state"))
    topic = col_topic.selectbox("Research topic", ("Any", *topics))
    col_decision, col_evidence = st.columns(2)
    decision_filter = col_decision.selectbox("Decision", ("Recommended", "All", "Undecided", "Pursuing", "Rejected", "Contacted"))
    evidence_filter = col_evidence.selectbox("Research evidence", ("Any", "Has research interests", "Has recent verified work"))
    minimum = st.slider("Minimum Research Fit", 0.0, 10.0, 0.0, 0.5)
    states = {"Undecided": "UNDECIDED", "Pursuing": "PURSUE", "Rejected": "REJECTED", "Contacted": "CONTACTED"}
    visible = [card for card in cards if card["research_fit"] >= minimum
               and (topic == "Any" or topic.casefold() in (card["research_topics"] or "").casefold())
               and (decision_filter == "All" or
                    (decision_filter == "Recommended" and card["decision_state"] not in {"REJECTED", "ARCHIVED"}) or
                    card["decision_state"] == states.get(decision_filter))
               and (evidence_filter == "Any" or
                    evidence_filter == "Has research interests" and bool(card["research_interest_summary"] or card["research_topics"]) or
                    evidence_filter == "Has recent verified work" and bool(card["recent_work"]))]
    if order == "Name":
        visible.sort(key=lambda card: card["name"].casefold())
    elif order == "Most recent research":
        visible.sort(key=lambda card: -(card["recent_work"][0]["year"] or 0) if card["recent_work"] else 0)
    elif order == "Decision state":
        visible.sort(key=lambda card: (card["decision_state"], -card["research_fit"]))
    with connect(path) as db:
        linked = {row[0] for row in db.execute(
            "SELECT faculty_profile_id FROM application_faculty WHERE application_id=?", (application_id,))}
        packages = [dict(row) for row in db.execute("""SELECT * FROM application_packages
            WHERE application_id=? AND context='FACULTY_OUTREACH' AND status='READY' ORDER BY version_number DESC""",
            (application_id,))]
    for card in visible:
        with st.container(border=True):
            st.markdown(f"### {html.escape(card['name'])}")
            st.caption(" · ".join(filter(None, (card["department"], card["lab"], card["institution"]))))
            st.caption(f"{card['decision_state'].replace('_', ' ').title()} · Identity: {card['verification_state']} · Affiliation: {card['affiliation_state']} · Research checked: {card['research_checked_at'] or 'Unknown'} ({card['research_freshness']})")
            st.metric("Research Fit · alignment aid", f"{card['research_fit']:.1f}/10")
            st.write(f"**Research interests · {card['research_interest_state']}:**", card["research_interest_summary"] or card["research_topics"] or "Not yet extracted")
            if card["research_topics"]:
                st.caption(card["research_topics"])
            st.write("**Recent verified research**")
            if card["recent_work"]:
                for work in card["recent_work"][:2]:
                    st.write(f"{work['year'] or 'Year unknown'} · {work['title']}")
            else:
                st.caption("No verified recent work yet. Official-page listings may need review.")
                if card["openalex_resolution_state"] == "AMBIGUOUS":
                    st.caption("Recent publications need author identity review.")
            for work in card["official_recent_work"][:2]:
                st.write(f"Official-page listing · Reviewed · {work['year'] or 'Year unknown'} · {work['title']}")
            for snapshot in card["unreviewed_research"][:1]:
                for work in snapshot["metadata"].get("recent_research_candidates", [])[:2]:
                    st.write(f"Official-page listing · NEEDS_REVIEW · {work['year'] or 'Year unknown'} · {work['title']}")
            if card["current_projects"]:
                st.write("**Current work · reviewed official source:**", card["current_projects"][0]["title"])
            if card["relevant_applicant_experience"]:
                st.write("**Why this matches you · demonstrated:**", card["relevant_applicant_experience"][0])
            if card["proposed_overlap"]:
                st.caption("Proposed direction: " + card["proposed_overlap"][0])
            st.caption(f"Supervision availability: {card['supervision_state']} · Programme contact policy: {card['contact_policy'] or 'Unknown'}")
            if card["unknowns"]:
                st.caption("Still checking: " + ", ".join(card["unknowns"]))
            actions = st.columns(3)
            if card["decision_state"] not in {"PURSUE", "CONTACTED", "REPLIED"} and actions[0].button("Pursue", key=f"pursue_{application_id}_{card['faculty_id']}"):
                if _run(lambda: research.decide(application_id, "PURSUE", faculty_id=card["faculty_id"]), "Professor shortlisted"):
                    st.rerun()
            if card["decision_state"] != "REJECTED" and actions[1].button("Reject", key=f"reject_{application_id}_{card['faculty_id']}"):
                if _run(lambda: research.decide(application_id, "REJECTED", faculty_id=card["faculty_id"]), "Professor moved to Rejected"):
                    st.rerun()
            if actions[2].button("Research deeper", key=f"research_{application_id}_{card['faculty_id']}",
                                 disabled=not bool(card["official_profile_url"])):
                operation_id = _run(lambda: OperationService(path).queue(
                    "FACULTY_DISCOVERY", application_id=application_id, context_id=context["id"],
                    faculty_id=card["faculty_id"]))
                if operation_id and _run(lambda: OperationService(path).launch(operation_id)):
                    st.rerun()
            if card["decision_state"] == "PURSUE" and card["verification_state"] == "VERIFIED" and card["faculty_id"] in linked and packages and card["contact_policy_state"] == "PASS":
                if context.get("trust_level") != "TRUSTED":
                    st.caption("Confirm your profile before drafting a professor email.")
                elif st.button("Draft email", key=f"simple_email_{card['faculty_id']}", type="primary"):
                    _run(lambda: OutreachService(path).prepare(
                        card["faculty_id"], application_id, packages[0]["id"],
                        context["profile_version_id"], context["research_track_version_id"]),
                        "Email draft prepared")
            with st.expander("Research details · evidence and unknowns"):
                st.caption("Research Fit explains shared evidence; it is not an admission or supervision prediction.")
                st.caption(f"Affiliation checked: {card['affiliation_checked_at'] or 'Unknown'} · Verified publications checked: {card['publication_checked_at'] or 'Unknown'}")
                st.write("Matched professor topics:", card["overlapping_terms"] or "Not established")
                st.write("Matched demonstrated applicant evidence:", card["relevant_applicant_experience"] or "None found")
                st.write("Proposed direction (separate):", card["proposed_overlap"] or "None found")
                for index, work in enumerate(card["recent_work"][:8]):
                    st.write(f"{work['year'] or 'Year unknown'} · {work['title']} · Verified source")
                    st.link_button("Publication source", work["source_url"], key=f"work_{card['faculty_id']}_{index}")
                for index, work in enumerate(card["official_recent_work"][:8]):
                    st.write(f"Official-page listing · Reviewed · {work['year'] or 'Year unknown'} · {work['title']}")
                    st.link_button("Listing source", work["source_url"], key=f"listing_{card['faculty_id']}_{index}")
                for index, project in enumerate(card["current_projects"]):
                    st.write("Current project (reviewed):", project["title"])
                    st.link_button("Project source", project["source_url"], key=f"project_{card['faculty_id']}_{index}")
                for snapshot in research.snapshots(faculty_id=card["faculty_id"]):
                    if snapshot["extraction_state"] == "VERIFIED":
                        st.caption(f"Reviewed research extraction · {snapshot['checked_at']}")
                        st.link_button("Research source", snapshot["source_url"], key=f"reviewed_research_{snapshot['id']}")
                for snapshot in card["unreviewed_research"]:
                    st.caption(f"Official-page extraction · NEEDS_REVIEW · {snapshot['checked_at']}")
                    st.write(snapshot["metadata"].get("research_interest_summary") or "Interests not extracted")
                    for work in snapshot["metadata"].get("recent_research_candidates", [])[:8]:
                        st.write(f"Possible work · {work['year'] or 'Year unknown'} · {work['title']}")
                    for project in snapshot["metadata"].get("current_projects", []):
                        st.write("Possible current project ·", project)
                    st.link_button("Review official source", snapshot["source_url"], key=f"source_{snapshot['id']}")
                    if st.button("Confirm extracted research", key=f"confirm_research_{snapshot['id']}"):
                        if _run(lambda snapshot=snapshot: research.review_snapshot(snapshot["id"], "Applicant")):
                            st.rerun()
                st.caption("Still unknown: " + (", ".join(card["unknowns"]) or "None identified"))
                if card["official_profile_url"]:
                    st.link_button("Official profile", card["official_profile_url"], key=f"profile_{card['faculty_id']}")
    rejected = [card for card in cards if card["decision_state"] == "REJECTED"]
    with st.expander(f"Rejected professors ({len(rejected)})"):
        for card in rejected:
            st.write(f"{card['name']} · {card['department'] or 'Department unknown'}")
            st.caption((card["research_interest_summary"] or card["research_topics"] or "Research not reviewed") +
                       f" · Rejected {card['decision_at'] or 'date unknown'}")
            if st.button("Restore", key=f"restore_professor_{application_id}_{card['faculty_id']}"):
                if _run(lambda card=card: research.decide(application_id, "UNDECIDED", faculty_id=card["faculty_id"])):
                    st.rerun()


@st.fragment(run_every="2s")
def _live_faculty_leads(path: Path, institution: str, application_id: int, context_id: int) -> None:
    research = FacultyResearch(path)
    pending = [item for item in Discovery(path).list_faculty_candidates()
               if item["review_state"] == "NEW" and item["institution"].casefold() == institution.casefold()]
    rejected = []
    active = []
    for item in pending:
        item["decision"] = research.decision(application_id, candidate_id=item["id"])
        (rejected if item["decision"] and item["decision"]["state"] == "REJECTED" else active).append(item)
    if active:
        st.subheader(f"New professor pages ({len(active)})")
        st.caption("These are source-backed leads. Check the official page before relying on affiliation or research fit.")
        for candidate in active[:20]:
            with st.container(border=True):
                st.write(f"**{candidate['name']}** · {candidate['department'] or 'Department unknown'}")
                snapshots = research.snapshots(candidate_id=candidate["id"])
                snapshot = snapshots[0] if snapshots else None
                metadata = snapshot["metadata"] if snapshot else {}
                st.caption(f"Extracted from official page · NEEDS_REVIEW · checked {snapshot['checked_at'] if snapshot else 'unknown'} · {candidate['decision']['state'] if candidate['decision'] else 'UNDECIDED'}")
                st.write("Research interests:", metadata.get("research_interest_summary") or "Not yet extracted")
                if metadata.get("research_topics"):
                    st.caption(" · ".join(metadata["research_topics"]))
                for work in metadata.get("recent_research_candidates", [])[:2]:
                    st.write(f"Possible recent work · {work['year'] or 'Year unknown'} · {work['title']}")
                if metadata.get("research_interest_summary") or metadata.get("research_topics"):
                    from phd_agent.applicant_context import ApplicantResearchContextService
                    overlap = ApplicantResearchContextService(path).retrieve(
                        context_id, "FACULTY_ALIGNMENT",
                        metadata.get("research_interest_summary") or " ".join(metadata["research_topics"]),
                        top_k=3, use="exploration", include_proposed=True)
                    demonstrated = [item.text for item in overlap.items if item.classification == "DEMONSTRATED"]
                    proposed = [item.text for item in overlap.items if item.classification == "PROPOSED"]
                    if demonstrated:
                        st.write("Why it may match you · demonstrated:", demonstrated[0])
                    if proposed:
                        st.caption("Proposed direction: " + proposed[0])
                if snapshot:
                    with st.expander("Extracted details · needs review"):
                        st.write("Source excerpt:", metadata.get("source_excerpt") or "No labelled section found")
                        for project in metadata.get("current_projects", []):
                            st.write("Possible current project:", project)
                        st.caption("This extraction does not verify identity or supervision.")
                actions = st.columns(3)
                if actions[0].button("Pursue", key=f"pursue_lead_{application_id}_{candidate['id']}"):
                    if _run(lambda candidate=candidate: research.decide(application_id, "PURSUE", candidate_id=candidate["id"])):
                        st.rerun()
                if actions[1].button("Reject", key=f"reject_lead_{application_id}_{candidate['id']}"):
                    if _run(lambda candidate=candidate: research.decide(application_id, "REJECTED", candidate_id=candidate["id"])):
                        st.rerun()
                if actions[2].button("Review professor", key=f"review_lead_{candidate['id']}"):
                    if _run(lambda candidate=candidate: Discovery(path).review_faculty_candidate(candidate["id"]),
                            "Professor added for verification"):
                        st.rerun()
                if candidate["profile_url"]:
                    st.link_button("Check official profile", candidate["profile_url"])
                if st.button("Dismiss lead", key=f"dismiss_faculty_{candidate['id']}"):
                    if _run(lambda candidate=candidate: Discovery(path).review_faculty_candidate(
                            candidate["id"], dismiss=True), "Lead dismissed"):
                        st.rerun()
    with st.expander(f"Rejected professor leads ({len(rejected)})"):
        for candidate in rejected:
            st.write(candidate["name"], candidate["department"] or "")
            st.caption(f"Rejected {candidate['decision']['decided_at']}")
            if st.button("Restore", key=f"restore_lead_{application_id}_{candidate['id']}"):
                if _run(lambda candidate=candidate: research.decide(application_id, "UNDECIDED", candidate_id=candidate["id"])):
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


def _calendar(path: Path) -> None:
    planner = DailyPlanner(path)
    applications = {item["id"]: item for item in Ledger(path).list_applications()}
    st.title("Calendar")
    st.caption("Deadlines, tasks, referee dates and your own events in one place. Unverified deadlines stay labeled.")
    with st.expander("Add an event"):
        with st.form("new_calendar_event"):
            title = st.text_input("Event title", placeholder="Oxford interview preparation")
            event_type = st.selectbox("Event type", EVENT_TYPES,
                format_func=lambda value: value.replace("_", " ").title())
            when = st.date_input("Date", value=date.today())
            application_id = st.selectbox("Application", [None, *applications],
                format_func=lambda value: "Personal / no application" if value is None else
                    f"{applications[value]['institution']} · {applications[value]['application_name']}")
            notes = st.text_area("Notes")
            add = st.form_submit_button("Add event", type="primary")
        if add and _run(lambda: planner.create_event(title, event_type, when.isoformat(),
                     application_id=application_id, notes=notes), "Event added"):
            st.rerun()
    dates = st.columns(2)
    start = dates[0].date_input("From", value=date.today(), key="calendar_from")
    end = dates[1].date_input("To", value=date.today() + timedelta(days=60), key="calendar_to")
    events = _run(lambda: planner.calendar(start, end))
    if events is None:
        return
    st.caption(f"{len(events)} event{'s' if len(events) != 1 else ''} in this range")
    if not events:
        st.info("No events are recorded for these dates.")
    day = None
    for event in events:
        event_day = event["when"][:10]
        if event_day != day:
            st.subheader(event_day)
            day = event_day
        with st.container(border=True):
            st.markdown(f"**{html.escape(event['title'])}**")
            st.caption(" · ".join(filter(None, (event["kind"].title(), event.get("institution"),
                                                event.get("verification_state"), event["when"]))))
            if event.get("source_url"):
                st.link_button("Open source", event["source_url"])
            if event["kind"] == "TASK" and st.button("Mark task done", key=f"calendar_done_{event['id']}"):
                if _run(lambda event=event: planner.complete_task(event["id"]), "Task completed"):
                    st.rerun()
            if event["kind"] == "MANUAL":
                with st.expander("Edit event"):
                    with st.form(f"edit_event_{event['id']}"):
                        revised_title = st.text_input("Title", value=event["title"])
                        revised_type = st.selectbox("Type", EVENT_TYPES,
                            index=EVENT_TYPES.index(event["event_type"]))
                        revised_date = st.date_input("New date", value=date.fromisoformat(event_day))
                        revised_notes = st.text_area("Notes", value=event["notes"])
                        save = st.form_submit_button("Save event")
                    if save and _run(lambda event=event: planner.edit_event(event["id"],
                            title=revised_title, event_type=revised_type,
                            starts_at=revised_date.isoformat(), notes=revised_notes), "Event updated"):
                        st.rerun()
                    if st.button("Cancel event", key=f"cancel_event_{event['id']}"):
                        if _run(lambda event=event: planner.set_event_active(event["id"], False),
                                "Event cancelled; it can be restored"):
                            st.rerun()
    with st.expander("Cancelled events"):
        cancelled = [event for event in planner.calendar(start, end, include_cancelled=True)
                     if event["kind"] == "MANUAL" and event["status"] == "CANCELLED"]
        for event in cancelled:
            st.write(f"{event['when']} · {event['title']}")
            if st.button("Restore event", key=f"restore_event_{event['id']}"):
                if _run(lambda event=event: planner.set_event_active(event["id"], True)):
                    st.rerun()


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
    if page == "Today":
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
    elif page == "Calendar":
        _calendar(db_path)
    else:
        _documents(db_path, context)
