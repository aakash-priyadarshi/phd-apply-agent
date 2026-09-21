"""Operator-only outreach review, Sent reconciliation, and campaign controls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from phd_agent.db import connect
from phd_agent.documents import DocumentVault
from phd_agent.gmail_gateway import GmailGateway
from phd_agent.outreach import CampaignPolicy, OutreachService


def _rows(path: Path, query: str, params=()) -> list[dict]:
    with connect(path) as db:
        return [dict(r) for r in db.execute(query, params)]


def _act(action, success: str):
    try:
        result = action()
    except Exception as error:
        st.error(str(error))
        return
    st.success(success)
    st.rerun()
    return result


def render_outreach(path: Path):
    st.subheader("Outreach review")
    st.caption("A verified context and ready faculty-outreach document package are required before drafting. Sending is a separate reviewed action.")
    service = OutreachService(path)
    gateway = GmailGateway(data_dir=path.parent)
    prep, review, history, campaigns = st.tabs(["Prepare", "Review queue", "Gmail Sent memory", "Campaign dry run"])

    with prep:
        profiles = _rows(path, "SELECT id,profile_id,version_number FROM profile_versions WHERE approval_state='APPROVED' ORDER BY id DESC")
        tracks = _rows(path, """SELECT v.id,v.version_number,t.title FROM research_track_versions v
            JOIN research_tracks t ON t.id=v.track_id WHERE v.approval_state='APPROVED' AND t.status='ACTIVE'
            ORDER BY v.id DESC""")
        apps = _rows(path, "SELECT id,cycle FROM applications ORDER BY id DESC")
        faculty = _rows(path, "SELECT id,name,institution FROM faculty_profiles WHERE verification_state='VERIFIED' ORDER BY name")
        if not profiles or not tracks:
            st.warning("Active applicant claims and a research direction still need explicit review. Sandbox approvals were not imported.")
        elif not apps or not faculty:
            st.info("Link a verified professor to an application or faculty enquiry, then build its outreach document package.")
        else:
            profile_id = st.selectbox("Approved applicant profile snapshot", [p["id"] for p in profiles],
                format_func=lambda i: f"Profile version #{i}", key="outreach_profile")
            track_id = st.selectbox("Approved research direction", [t["id"] for t in tracks],
                format_func=lambda i: next(f"{t['title']} v{t['version_number']}" for t in tracks if t["id"]==i))
            app_id = st.selectbox("Application or faculty enquiry", [a["id"] for a in apps],
                format_func=lambda i: f"Application #{i}")
            linked = _rows(path, """SELECT f.id,f.name,f.institution FROM application_faculty af
                JOIN faculty_profiles f ON f.id=af.faculty_profile_id WHERE af.application_id=? ORDER BY f.name""", (app_id,))
            packages = _rows(path, """SELECT id,version_number,status FROM application_packages
                WHERE application_id=? AND context='FACULTY_OUTREACH' ORDER BY id DESC""", (app_id,))
            if not linked:
                st.warning("Link a verified professor to this application in Discovery.")
            elif not packages:
                st.warning("Build and mark a FACULTY_OUTREACH document package ready in Packages & Preflight.")
            else:
                faculty_id = st.selectbox("Professor", [f["id"] for f in linked],
                    format_func=lambda i: next(f"{f['name']} — {f['institution']}" for f in linked if f["id"]==i))
                document_package_id = st.selectbox("Frozen outreach document package", [p["id"] for p in packages],
                    format_func=lambda i: next(f"#{i} v{p['version_number']} · {p['status']}" for p in packages if p["id"]==i))
                available_campaigns = _rows(path, "SELECT id,name FROM campaigns WHERE status='ACTIVE' ORDER BY id DESC")
                campaign_id = st.selectbox("Campaign (optional)", [0]+[c["id"] for c in available_campaigns],
                    format_func=lambda i: "One-off" if i==0 else next(c["name"] for c in available_campaigns if c["id"]==i))
                if st.button("Check eligibility and draft outreach"):
                    _act(lambda: service.prepare(faculty_id,app_id,document_package_id,profile_id,track_id,
                        campaign_id=campaign_id or None), "Grounded draft added to review queue")

    with review:
        packages = _rows(path, """SELECT p.*,f.name,f.institution FROM outreach_packages p
            JOIN faculty_profiles f ON f.id=p.faculty_profile_id ORDER BY p.id DESC LIMIT 100""")
        if not packages:
            st.info("No outreach packages yet. Real drafting remains blocked by the active applicant review gate.")
        else:
            package_id = st.selectbox("Outreach package", [p["id"] for p in packages],
                format_func=lambda i: next(f"#{i} · {p['name']} · {p['status']} · v{p['version_number']}" for p in packages if p["id"]==i))
            package = next(p for p in packages if p["id"]==package_id)
            stale_reasons = service.refresh_staleness(package_id) if package["status"] in {"APPROVED","SCHEDULED"} else []
            context,draft = service._unpack(package)
            left,right = st.columns(2)
            with left:
                st.markdown("#### Professor and applicant")
                st.write({"Professor":context.professor_name,"Institution":context.institution,
                    "Recipient":context.recipient,"Topics":context.research_topics,
                    "Research track":context.research_track_version_id,"Approved claim IDs":context.claim_ids})
                st.write("Stored papers", context.publication_titles)
                st.write("Evidence IDs", {"research":context.professor_evidence_ids,
                    "email":context.email_evidence_ids,"affiliation":context.affiliation_evidence_ids})
            with right:
                st.markdown("#### Message")
                st.write("Subject:",draft.subject)
                st.text_area("Frozen email body",draft.body,height=300,disabled=True)
                st.caption(f"{len(draft.body.split())} words · {draft.provider}/{draft.model} · {draft.prompt_version}")
            st.markdown("#### Exact attachments")
            st.dataframe([{"Version":a.document_version_id,"Type":a.document_type,"Filename":a.filename,
                "SHA-256":a.sha256,"Reason":a.reason} for a in context.attachments],hide_index=True)
            vault = DocumentVault(path)
            for a in context.attachments:
                version = vault.get_version(a.document_version_id)
                if version and vault.storage.verify_hash(version["storage_key"],a.sha256):
                    st.download_button(f"Review {a.filename}",vault.storage.get(version["storage_key"]),
                        file_name=a.filename,key=f"outreach_file_{package_id}_{a.document_version_id}")
            gate = json.loads(package["quality_json"])
            st.metric("Frozen quality gate",gate["status"])
            st.dataframe(gate["rules"],hide_index=True)
            if package["stale_at"] or stale_reasons:
                st.error("Package is stale: " + ", ".join(stale_reasons or ["previous review flagged a change"]))
            reviewer = st.text_input("Reviewer/operator",key=f"outreach_reviewer_{package_id}")
            if package["status"] in {"NEEDS_REVIEW","BLOCKED","APPROVED","SCHEDULED"}:
                with st.expander("Edit as a new immutable version"):
                    subject = st.text_input("New subject",value=draft.subject,key=f"outreach_subject_{package_id}")
                    body = st.text_area("New body",value=draft.body,height=250,key=f"outreach_body_{package_id}")
                    if st.button("Save new version",key=f"outreach_revise_{package_id}"):
                        _act(lambda: service.revise(package_id,subject,body,reviewer),"New outreach version created")
            if package["status"] == "NEEDS_REVIEW":
                a,b = st.columns(2)
                with a:
                    if st.button("Approve reviewed package"):
                        _act(lambda: service.approve(package_id,reviewer),"Outreach package approved")
                with b:
                    if st.button("Reject package"):
                        _act(lambda: service.reject(package_id,reviewer),"Outreach package cancelled")
            if package["status"] == "APPROVED":
                when = st.text_input("Schedule reminder for UTC time (ISO 8601)",placeholder="2026-10-01T10:00:00+00:00")
                if st.button("Schedule for later"):
                    _act(lambda: service.schedule(package_id,when,reviewer),"Package scheduled for operator review")
            if package["status"] in {"APPROVED","SCHEDULED"}:
                st.caption("Manual send requires rotated OAuth credentials, a valid token, Sent reconciliation, and a fresh gate. No automatic worker is active.")
                acknowledged = st.checkbox("I reviewed the exact recipient, email, and attachments",key=f"outreach_ack_{package_id}")
                if st.button("Send reviewed package via Gmail",disabled=not gateway.credentials_ready() or not acknowledged):
                    _act(lambda: service.send(package_id,gateway,reviewer),"Gmail attempt recorded; inspect the resulting queue state")
                if not gateway.credentials_ready():
                    st.warning("Gmail is unavailable until a rotated OAuth client and valid JSON token are configured locally.")
            if package["status"] in {"SENDING", "AMBIGUOUS_SEND"}:
                st.error("Delivery is uncertain. Do not send this package again. Reconcile Gmail Sent metadata to resolve it.")
                message = _rows(path,"SELECT id FROM outreach_messages WHERE package_id=? ORDER BY id DESC LIMIT 1",(package_id,))
                if message and st.button("Reconcile uncertain send"):
                    def recover():
                        if gateway.credentials_ready():
                            service.reconcile_sent(gateway.list_sent(),source="GMAIL")
                        return service.recover_ambiguous(message[0]["id"],reviewer)
                    _act(recover,"Uncertain send checked against known Sent metadata")

    with history:
        st.caption("Only recipient, subject, timestamp, and Gmail identifiers are imported. Potential links need confirmation.")
        if gateway.credentials_ready():
            if st.button("Import recent Gmail Sent metadata"):
                _act(lambda: service.reconcile_sent(gateway.list_sent(),source="GMAIL"),"Sent metadata imported for review")
        else:
            st.info("Live Gmail import is disabled until rotated credentials and a valid token are installed.")
        records = _rows(path, """SELECT g.*,f.name AS matched_professor FROM gmail_threads g
            LEFT JOIN faculty_profiles f ON f.id=g.faculty_profile_id ORDER BY g.id DESC LIMIT 100""")
        if records:
            st.dataframe([{"ID":r["id"],"Professor":r["matched_professor"],"Recipient":r["recipient"],
                "Subject":r["subject"],"When":r["message_at"],"Thread":r["gmail_thread_id"],
                "Confidence":r["match_confidence"],"Review":r["match_state"]} for r in records],hide_index=True)
            pending = [r for r in records if r["direction"]=="OUTBOUND" and r["match_state"]=="PENDING"]
            if pending:
                selected = st.selectbox("Sent candidate to review",[r["id"] for r in pending])
                item = next(r for r in pending if r["id"]==selected)
                matches = _rows(path,"SELECT id,name,email FROM faculty_profiles WHERE lower(email)=?",(item["recipient"],))
                if matches:
                    professor_id = st.selectbox("Confirm matching professor",[m["id"] for m in matches],
                        format_func=lambda i: next(m["name"] for m in matches if m["id"]==i))
                    if st.button("Confirm Sent link"):
                        _act(lambda: service.confirm_sent_link(item["gmail_message_id"],professor_id,"Local operator",accept=True),"Contact link confirmed")
                if st.button("Reject suggested link"):
                    _act(lambda: service.confirm_sent_link(item["gmail_message_id"],None,"Local operator",accept=False),"Suggested link rejected")
            if gateway.credentials_ready():
                confirmed = [r for r in records if r["direction"]=="OUTBOUND" and r["match_state"]=="CONFIRMED"]
                if confirmed:
                    selected_thread = st.selectbox("Confirmed thread for reply check",[r["id"] for r in confirmed],
                        format_func=lambda i: next(f"{r['subject']} · {r['gmail_thread_id']}" for r in confirmed if r["id"]==i))
                    item = next(r for r in confirmed if r["id"]==selected_thread)
                    if st.button("Check thread for incoming replies"):
                        def import_replies():
                            for message in gateway.list_thread(item["gmail_thread_id"]):
                                if message["message_id"] != item["gmail_message_id"] and "SENT" not in message["labels"]:
                                    service.record_reply(message)
                        _act(import_replies,"Thread metadata checked for replies")
        faculty = _rows(path,"SELECT id,name,email FROM faculty_profiles WHERE email IS NOT NULL ORDER BY name")
        if faculty:
            with st.expander("Record a known historical contact manually"):
                person = st.selectbox("Professor",[f["id"] for f in faculty],
                    format_func=lambda i: next(f["name"] for f in faculty if f["id"]==i))
                address = next(f["email"] for f in faculty if f["id"]==person)
                subject = st.text_input("Historical subject")
                when = st.text_input("Sent time (ISO 8601)")
                reviewer = st.text_input("Recorded by")
                if st.button("Record historical contact"):
                    _act(lambda: service.record_manual_contact(person,address,subject,when,reviewer),"Historical contact recorded")
        all_faculty = _rows(path,"SELECT id,name,institution FROM faculty_profiles ORDER BY name")
        if all_faculty:
            with st.expander("Record no-contact or rejection state"):
                person = st.selectbox("Professor for contact state",[f["id"] for f in all_faculty],
                    format_func=lambda i: next(f"{f['name']} — {f['institution']}" for f in all_faculty if f["id"]==i))
                current = _rows(path,"SELECT state,reason,reviewed_by FROM contact_restrictions WHERE faculty_profile_id=?",(person,))
                if current:
                    st.write("Current state",current[0])
                state = st.selectbox("Contact state",["DO_NOT_CONTACT","REJECTED","CLEAR"])
                reason = st.text_input("Reason and evidence for state")
                reviewer = st.text_input("Contact state reviewer")
                if st.button("Save reviewed contact state"):
                    _act(lambda: service.set_contact_restriction(person,state,reason,reviewer),"Contact state recorded")

    with campaigns:
        with st.expander("Create review-mode campaign"):
            name = st.text_input("Campaign name")
            cycle = st.text_input("Cycle",value="2026-27")
            timezone_name = st.text_input("Campaign target timezone (IANA; use one local zone per campaign)",placeholder="Europe/London")
            min_fit = st.number_input("Minimum Research Fit",0.0,10.0,0.0,0.5)
            min_coverage = st.number_input("Minimum evidence coverage",0.0,1.0,0.0,0.1)
            daily_cap = st.number_input("Daily send cap",1,20,3)
            if st.button("Create campaign (auto-send off)"):
                policy = CampaignPolicy(minimum_research_fit=min_fit,minimum_evidence_coverage=min_coverage,
                    timezone_name=timezone_name or None,daily_cap=daily_cap)
                _act(lambda: service.create_campaign(name,cycle,policy),"Campaign created with automatic sending off")
        campaign_rows = _rows(path,"SELECT * FROM campaigns ORDER BY id DESC")
        if campaign_rows:
            campaign_id = st.selectbox("Campaign",[c["id"] for c in campaign_rows],
                format_func=lambda i: next(f"{c['name']} · {c['status']}" for c in campaign_rows if c["id"]==i))
            campaign = next(c for c in campaign_rows if c["id"]==campaign_id)
            st.write({"Policy":json.loads(campaign["policy_json"]),"Auto-send enabled":bool(campaign["auto_send_enabled"]),
                      "Emergency stop":bool(campaign["emergency_stop"])})
            a,b = st.columns(2)
            with a:
                if st.button("Pause / resume campaign"):
                    _act(lambda: service.set_campaign_controls(campaign_id,"Local operator",
                        status="ACTIVE" if campaign["status"]=="PAUSED" else "PAUSED"),"Campaign control updated")
            with b:
                if st.button("Toggle emergency stop"):
                    _act(lambda: service.set_campaign_controls(campaign_id,"Local operator",
                        emergency_stop=not campaign["emergency_stop"]),"Emergency stop updated")
            if profiles and tracks:
                if st.button("Run and save campaign dry run"):
                    try:
                        report = service.dry_run(campaign_id,profiles[0]["id"],tracks[0]["id"])
                        st.session_state["slice4_dry_run"] = report
                    except Exception as error:
                        st.error(str(error))
                report = st.session_state.get("slice4_dry_run")
                if report and report["campaign_id"] == campaign_id:
                    st.dataframe(report["candidates"],hide_index=True)
