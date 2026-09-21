"""Reply triage, portal checklist, submission archive, and local backup."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from phd_agent.backup import BackupService
from phd_agent.db import connect, utc_now
from phd_agent.portal import ANSWER_KEYS, PAYMENT_STATES, PortalAssistance
from phd_agent.replies import CATEGORIES, ReplyIntelligence


def _act(action, success: str):
    try:
        result = action()
    except Exception as error:
        st.error(str(error))
        return None
    st.success(success)
    st.rerun()
    return result


def render_followthrough(path: Path):
    st.subheader("Follow-through")
    st.caption("Classify replies, prepare copy-ready portal answers, archive a submission you made yourself, and back up the local ledger. This tab never submits a portal form or processes payment.")
    intel = ReplyIntelligence(path)
    portal = PortalAssistance(path)
    replies, answers, archive, backup = st.tabs(["Replies", "Portal answers", "Submission archive", "Backup"])

    with replies:
        st.caption("Classification creates a next action. It never sends a reply.")
        if st.button("Mark follow-ups due"):
            _act(lambda: intel.mark_follow_ups_due(), "Follow-up reminders updated")
        inbox = intel.list_inbox()
        if not inbox:
            st.info("No reply metadata imported yet. Import Gmail thread metadata from Outreach Review.")
        else:
            for item in inbox:
                st.markdown(f"**#{item['id']}** {item.get('faculty_name') or 'Unmatched'} · {item['subject_raw']} · {item['detected_state']}")
                if item["category"]:
                    st.caption(f"Classified as {item['category']} ({item['confidence']}) · task #{item['task_id']}")
                    continue
                suggestion = ReplyIntelligence.suggest_category(item["subject_raw"])
                with st.form(f"classify_{item['id']}"):
                    category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(suggestion))
                    excerpt = st.text_area("Operator excerpt (optional; message bodies are not stored automatically)")
                    length = st.text_input("Requested length or template")
                    deadline = st.text_input("Stated deadline (ISO date)")
                    if st.form_submit_button("Classify and create task"):
                        _act(lambda: intel.classify(item["id"], category, "Operator", excerpt=excerpt,
                                                    requested_length=length or None,
                                                    requested_deadline=deadline or None),
                             "Reply classified; no email was sent")
        packages = []
        with connect(path) as db:
            packages = [dict(r) for r in db.execute(
                "SELECT id,stage,status FROM outreach_packages WHERE status IN ('SENT','FOLLOW_UP_DUE','REPLIED') ORDER BY id DESC")]
        if packages:
            package_id = st.selectbox("Create a reviewed follow-up draft from", [p["id"] for p in packages],
                                      format_func=lambda i: next(f"#{p['id']} {p['status']}" for p in packages if p["id"]==i))
            if st.button("Draft follow-up (does not send)"):
                _act(lambda: intel.draft_follow_up(package_id), "Follow-up draft added to Outreach Review")

    with answers:
        apps = []
        with connect(path) as db:
            apps = [dict(r) for r in db.execute("SELECT id,cycle FROM applications ORDER BY id DESC")]
        with st.form("new_answer"):
            field_key = st.selectbox("Answer field", ANSWER_KEYS)
            label = st.text_input("Label")
            value = st.text_area("Copy-ready value")
            if st.form_submit_button("Save draft answer"):
                _act(lambda: portal.save_answer(field_key, label, value), "Answer saved as draft")
        for answer in portal.list_answers():
            st.write(f"#{answer['id']} {answer['field_key']} v{answer['version_number']} · {answer['approval_state']}")
            st.code(answer["value_text"])
            if answer["approval_state"] == "DRAFT" and st.button("Approve answer", key=f"approve_answer_{answer['id']}"):
                _act(lambda: portal.approve_answer(answer["id"], "Operator"), "Answer approved")
        if apps:
            app_id = st.selectbox("Application checklist", [a["id"] for a in apps],
                                  format_func=lambda i: f"Application #{i}")
            with st.form("checklist_field"):
                key = st.selectbox("Field key", ANSWER_KEYS, key="checklist_key")
                portal_label = st.text_input("Portal field label")
                if st.form_submit_button("Add portal field"):
                    _act(lambda: portal.add_checklist_field(app_id, key, portal_label), "Portal field added")
            approved = [a for a in portal.list_answers() if a["approval_state"] == "APPROVED"]
            for field in portal.checklist(app_id):
                st.write(f"{field['portal_label']} · {field['status']}")
                if approved and field["status"] == "EMPTY":
                    answer_id = st.selectbox(f"Fill {field['portal_label']}", [a["id"] for a in approved],
                                             key=f"fill_{field['id']}",
                                             format_func=lambda i: next(a["label"] for a in approved if a["id"]==i))
                    if st.button("Fill from approved answer", key=f"fill_btn_{field['id']}"):
                        _act(lambda: portal.fill_field(field["id"], answer_id), "Field filled")
                if field["status"] == "FILLED" and st.button("Mark reviewed", key=f"review_{field['id']}"):
                    _act(lambda: portal.review_field(field["id"], "Operator"), "Field reviewed")
            st.markdown("#### Copy-ready values")
            st.json(portal.copy_ready(app_id))

    with archive:
        st.caption("Record a confirmation number after you submit and pay on the institution's site.")
        apps = []
        with connect(path) as db:
            apps = [dict(r) for r in db.execute("SELECT id,cycle,status FROM applications ORDER BY id DESC")]
            packages = [dict(r) for r in db.execute(
                "SELECT id,application_id,status FROM application_packages WHERE context='FORMAL_APPLICATION' AND status='READY'")]
        if not apps or not packages:
            st.info("Build and mark a FORMAL_APPLICATION package ready first.")
        else:
            app_id = st.selectbox("Submitted application", [a["id"] for a in apps], key="archive_app")
            matching = [p for p in packages if p["application_id"] == app_id]
            if not matching:
                st.warning("No READY formal package for this application.")
            else:
                package_id = st.selectbox("Frozen package", [p["id"] for p in matching], key="archive_package")
                confirmation = st.text_input("Portal confirmation number")
                submitted_at = st.text_input("Submission time (ISO)", value=utc_now())
                payment = st.selectbox("Payment (user-recorded)", PAYMENT_STATES)
                reference = st.text_input("Payment reference")
                if st.button("Freeze submission archive"):
                    _act(lambda: portal.record_submission(
                        app_id, package_id, confirmation, submitted_at,
                        "Operator", payment_state=payment, payment_reference=reference or None,
                        acknowledge_empty_checklist=True),
                        "Submission archived. The app did not contact the portal.")
        with connect(path) as db:
            archives = [dict(r) for r in db.execute("SELECT * FROM submission_archives ORDER BY id DESC")]
        for row in archives:
            st.write(f"Archive #{row['id']} · application {row['application_id']} · {row['confirmation_number']} · payment {row['payment_state']}")

    with backup:
        st.caption("Back up the SQLite ledger and Document Vault. Gmail credentials are excluded unless you explicitly include them.")
        dest = st.text_input("Backup directory", value=str(path.parent / "backups" / "manual-backup"))
        include = st.checkbox("Include Gmail credentials (not recommended)", value=False)
        if st.button("Create backup"):
            service = BackupService(path, path.parent)
            _act(lambda: service.create_backup(Path(dest), "Operator", include_credentials=include),
                 "Backup written")
        restore_from = st.text_input("Restore from directory")
        restore_to = st.text_input("Restore into directory", value=str(path.parent / "restore-copy"))
        replace = st.checkbox("Replace existing destination database")
        if st.button("Restore backup"):
            service = BackupService(path, path.parent)
            _act(lambda: service.restore(Path(restore_from), Path(restore_to), "Operator",
                                         replace_existing=replace),
                 "Backup restored into the destination directory")
