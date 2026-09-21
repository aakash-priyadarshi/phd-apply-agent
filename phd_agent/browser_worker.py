"""Local browser-worker contract and deterministic form planning."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from phd_agent.db import connect, migrate, transaction, utc_now


def companion_command(plan_id: int, platform_name: str | None = None) -> tuple[str, str]:
    import os
    if (platform_name or os.name) == "nt":
        return f".\\.venv\\Scripts\\python.exe -m scripts.browser_companion {plan_id}", "powershell"
    return f"./.venv/bin/python -m scripts.browser_companion {plan_id}", "bash"


def _css_escape(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("'", "\\'")


def _normalized_label(value: str) -> str:
    return re.sub(r"\W+", " ", (value or "").casefold()).strip()


def build_field_locator(*, tag: str, element_id: str | None, name: str | None,
                        aria_label: str | None, tag_index: int,
                        accessible_label: str | None = None) -> str:
    if element_id:
        return f"#{_css_escape(element_id)}"
    label = (accessible_label or "").strip()
    if label and not _normalized_label(label).startswith("field"):
        return f"get-by-label:{label}"
    if name:
        return f"{tag}[name='{_css_escape(name)}']"
    if aria_label:
        return f"{tag}[aria-label='{_css_escape(aria_label)}']"
    return f"xpath=(//{tag})[{max(1, tag_index)}]"


def playwright_locator(page, locator: str):
    if locator.startswith("get-by-label:"):
        return page.get_by_label(locator.split(":", 1)[1], exact=False)
    return page.locator(locator)


def _wrapping_label_text(node) -> str:
    wrapping = node.find_parent("label") if hasattr(node, "find_parent") else None
    if wrapping is None:
        return ""
    parts = []
    for child in wrapping.contents:
        name = getattr(child, "name", None)
        if name in {"input", "textarea", "select", "button"}:
            continue
        text = child.get_text(" ", strip=True) if hasattr(child, "get_text") else str(child).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def labels_match(expected: str, actual: str) -> bool:
    left, right = _normalized_label(expected), _normalized_label(actual)
    if not left or left.startswith("field "):
        return True
    return bool(right) and (left in right or right in left)


SENSITIVE_KEYS = {"PASSWORD", "MFA_CODE", "PAYMENT", "PASSPORT", "GOVERNMENT_ID"}
NARRATIVE_KEYS = {"RESEARCH_INTERESTS", "AWARDS", "PUBLICATIONS_SUMMARY", "FUNDING_STATEMENT", "OTHER"}
LABEL_MAP = (
    (re.compile(r"full\s*name|legal\s*name", re.I), "FULL_NAME"),
    (re.compile(r"e-?mail", re.I), "EMAIL"),
    (re.compile(r"phone|mobile", re.I), "PHONE"),
    (re.compile(r"nationality|citizenship", re.I), "NATIONALITY"),
    (re.compile(r"degree|education|qualification", re.I), "EDUCATION"),
    (re.compile(r"research\s*(interest|experience)|proposed\s*research", re.I), "RESEARCH_INTERESTS"),
    (re.compile(r"publication", re.I), "PUBLICATIONS_SUMMARY"),
    (re.compile(r"award|honou?r", re.I), "AWARDS"),
    (re.compile(r"referee|reference", re.I), "REFEREES"),
    (re.compile(r"english|ielts|toefl", re.I), "ENGLISH_TEST"),
    (re.compile(r"funding|scholarship", re.I), "FUNDING_STATEMENT"),
    (re.compile(r"password", re.I), "PASSWORD"),
    (re.compile(r"mfa|verification\s*code|one.time", re.I), "MFA_CODE"),
    (re.compile(r"payment|card\s*number|cvv", re.I), "PAYMENT"),
    (re.compile(r"passport", re.I), "PASSPORT"),
)


@dataclass(frozen=True)
class BrowserField:
    locator: str
    label: str
    input_type: str = "text"
    required: bool = False


@dataclass(frozen=True)
class AcquisitionResult:
    url: str
    status: str
    text: str = ""
    html: str = ""
    method: str = "PLAYWRIGHT"
    human_action: str | None = None


@dataclass(frozen=True)
class FillPlanItem:
    locator: str
    label: str
    field_key: str | None
    action: str
    value: str | None
    source: dict | None
    reason: str


class BrowserWorker(Protocol):
    def acquire(self, url: str) -> AcquisitionResult: ...
    def inspect_fields(self, url: str) -> Sequence[BrowserField]: ...
    def fill_safe_fields(self, plan: Sequence[FillPlanItem]) -> int: ...


class LocalPlaywrightWorker:
    """Optional local companion. It pauses for CAPTCHA, MFA, and unsupported widgets."""

    BLOCK_SIGNALS = ("captcha", "verify you are human", "access denied", "cloudflare")

    def _playwright(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise RuntimeError("Install Playwright in the local companion environment to use browser acquisition") from error
        return sync_playwright

    def acquire(self, url: str) -> AcquisitionResult:
        _validate_web_url(url)
        sync_playwright = self._playwright()
        with sync_playwright() as api:
            browser = api.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded")
            text = page.locator("body").inner_text()
            html = page.content()
            if any(signal in text.casefold() for signal in self.BLOCK_SIGNALS):
                browser.close()
                return AcquisitionResult(url, "HUMAN_INPUT_REQUIRED", method="PLAYWRIGHT",
                    human_action="Complete the CAPTCHA or access check in the local browser, then retry acquisition.")
            browser.close()
            return AcquisitionResult(url, "ACQUIRED", text=text, html=html)

    def inspect_fields(self, url: str) -> Sequence[BrowserField]:
        _validate_web_url(url)
        sync_playwright = self._playwright()
        fields: list[BrowserField] = []
        with sync_playwright() as api:
            browser = api.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded")
            tag_counts: dict[str, int] = {}
            for index, node in enumerate(page.locator("input, textarea, select").all()):
                tag_name = node.evaluate("el => el.tagName.toLowerCase()")
                input_type = node.get_attribute("type") or ("text" if tag_name == "input" else tag_name)
                if input_type in {"hidden", "submit", "button"}:
                    continue
                node_id = node.get_attribute("id")
                name = node.get_attribute("name")
                aria = node.get_attribute("aria-label")
                label = ""
                if node_id and page.locator(f"label[for='{node_id}']").count():
                    label = page.locator(f"label[for='{node_id}']").first.inner_text()
                if not label:
                    wrapping = node.evaluate("el => el.closest('label') && el.closest('label').innerText")
                    label = wrapping or aria or name or f"Field {index + 1}"
                tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1
                locator = build_field_locator(
                    tag=tag_name, element_id=node_id, name=name, aria_label=aria,
                    tag_index=tag_counts[tag_name], accessible_label=str(label).strip())
                fields.append(BrowserField(locator, str(label).strip(), input_type,
                                           node.get_attribute("required") is not None))
            browser.close()
        return fields

    def fill_safe_fields(self, plan: Sequence[FillPlanItem]) -> int:
        raise RuntimeError("Start filling from a supervised local companion session; the Railway CMS never controls portal sessions")


def _validate_web_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Use a complete HTTP or HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not accepted")


def fields_from_html(html: str) -> list[BrowserField]:
    soup = BeautifulSoup(html, "html.parser")
    fields = []
    tag_counts: dict[str, int] = {}
    for index, node in enumerate(soup.select("input, textarea, select")):
        input_type = node.get("type") or ("text" if node.name == "input" else node.name)
        if input_type in {"hidden", "submit", "button"}:
            continue
        node_id = node.get("id")
        name = node.get("name")
        aria = node.get("aria-label")
        label_node = soup.select_one(f"label[for='{node_id}']") if node_id else None
        wrapping_label = _wrapping_label_text(node)
        label = (label_node.get_text(" ", strip=True) if label_node else
                 wrapping_label or aria or name or node.get("placeholder") or f"Field {index + 1}")
        tag_counts[node.name] = tag_counts.get(node.name, 0) + 1
        locator = build_field_locator(
            tag=node.name, element_id=node_id, name=name, aria_label=aria,
            tag_index=tag_counts[node.name], accessible_label=label)
        fields.append(BrowserField(locator, label, input_type, node.has_attr("required")))
    return fields


class FormPlanService:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        migrate(self.db_path)

    @staticmethod
    def classify(label: str) -> str | None:
        for pattern, key in LABEL_MAP:
            if pattern.search(label):
                return key
        return None

    def create_plan(self, application_id: int, page_url: str, fields: Sequence[BrowserField], *,
                    context_id: int | None = None) -> dict:
        _validate_web_url(page_url)
        with connect(self.db_path) as db:
            if not db.execute("SELECT 1 FROM applications WHERE id=?", (application_id,)).fetchone():
                raise ValueError("Application not found")
            answers = [dict(row) for row in db.execute("""SELECT * FROM answer_library
                WHERE approval_state='APPROVED' ORDER BY version_number DESC""")]
        by_key = {}
        for answer in answers:
            by_key.setdefault(answer["field_key"], answer)
        items = []
        for field in fields:
            key = self.classify(field.label)
            answer = by_key.get(key)
            if key in SENSITIVE_KEYS:
                action, value, source, reason = "MANUAL_SENSITIVE", None, None, "Sensitive value stays local and outside model inputs"
            elif key in NARRATIVE_KEYS:
                action, value, source = "GENERATE_AND_REVIEW", None, None
                reason = "Narrative answer requires applicant-context retrieval and review"
                if context_id is not None:
                    from phd_agent.portal import PortalAssistance
                    try:
                        answer_id = PortalAssistance(self.db_path).draft_context_answer(
                            key, field.label, context_id, application_id=application_id)
                        with connect(self.db_path) as db:
                            draft = db.execute("SELECT * FROM answer_library WHERE id=?", (answer_id,)).fetchone()
                        value = draft["value_text"]
                        source = {"answer_id": answer_id, "claim_revision_id": draft["claim_revision_id"],
                                  "source_evidence_id": draft["source_evidence_id"]}
                        reason = "Grounded applicant-context draft; operator approval is required before use"
                    except ValueError as error:
                        reason = f"No grounded draft was created: {error}"
            elif field.input_type.casefold() not in {"text", "email", "tel", "url", "textarea"}:
                action, value, source, reason = "MANUAL", None, None, "This control type needs supervised operator input"
            elif answer:
                action, value = "SAFE_AUTOFILL", answer["value_text"]
                source = {"answer_id": answer["id"], "claim_revision_id": answer["claim_revision_id"],
                          "source_evidence_id": answer["source_evidence_id"]}
                reason = "Approved reusable Application Profile value"
            else:
                action, value, source, reason = "MANUAL", None, None, "No approved reusable value matches this field"
            items.append(FillPlanItem(field.locator, field.label, key, action, value, source, reason))
        safe = sum(item.action == "SAFE_AUTOFILL" for item in items)
        review = sum(item.action == "GENERATE_AND_REVIEW" for item in items)
        manual = len(items) - safe - review
        payload = [asdict(item) for item in items]
        with transaction(self.db_path) as db:
            plan_id = db.execute("""INSERT INTO browser_fill_plans
                (application_id,page_url,field_count,safe_count,review_count,manual_count,plan_json,created_at)
                VALUES(?,?,?,?,?,?,?,?)""", (
                application_id, page_url, len(items), safe, review, manual,
                json.dumps(payload, ensure_ascii=False, sort_keys=True), utc_now(),
            )).lastrowid
            db.execute("""INSERT INTO workload_events
                (event_type,count_value,entity_type,entity_id,metadata_json,created_at)
                VALUES('BROWSER_FIELDS_MAPPED',?,'BROWSER_FILL_PLAN',?,'{}',?)""", (len(items), plan_id, utc_now()))
            db.execute("""INSERT INTO workload_events
                (event_type,count_value,entity_type,entity_id,metadata_json,created_at)
                VALUES('FIELDS_REUSED_FROM_PROFILE',?,'BROWSER_FILL_PLAN',?,'{}',?)""", (safe, plan_id, utc_now()))
            db.execute("""INSERT INTO workload_events
                (event_type,count_value,entity_type,entity_id,metadata_json,created_at)
                VALUES('FIELDS_REQUIRING_MANUAL_ENTRY',?,'BROWSER_FILL_PLAN',?,'{}',?)""", (manual, plan_id, utc_now()))
        if context_id is not None:
            from phd_agent.applicant_context import ApplicantResearchContextService
            ApplicantResearchContextService(self.db_path).link_output(
                "BROWSER_FILL_PLAN", plan_id, context_id,
                provider="deterministic", model="form-mapper-v1", prompt_version="field-map-v1",
            )
        return {"id": plan_id, "field_count": len(items), "safe_count": safe,
                "review_count": review, "manual_count": manual, "items": payload}

    def approve(self, plan_id: int, reviewer: str) -> None:
        if not reviewer.strip():
            raise ValueError("Reviewer required")
        with transaction(self.db_path) as db:
            changed = db.execute("""UPDATE browser_fill_plans SET status='APPROVED',approved_at=?,approved_by=?
                WHERE id=? AND status='REVIEW_REQUIRED'""", (utc_now(), reviewer.strip(), plan_id)).rowcount
        if not changed:
            raise ValueError("Fill plan not found or already reviewed")

    def get(self, plan_id: int) -> dict:
        with connect(self.db_path) as db:
            row = db.execute("SELECT * FROM browser_fill_plans WHERE id=?", (plan_id,)).fetchone()
        if not row:
            raise ValueError("Fill plan not found")
        result = dict(row)
        result["items"] = json.loads(result.pop("plan_json"))
        return result


def execute_approved_plan(db_path: Path | str, plan_id: int, *,
                          browser_profile_dir: Path | str) -> int:
    """Open a local persistent browser, fill only approved safe fields, and stop before submit."""
    service = FormPlanService(db_path)
    plan = service.get(plan_id)
    if plan["status"] != "APPROVED":
        raise ValueError("Approve the fill plan in the CMS before local execution")
    safe_items = [FillPlanItem(**item) for item in plan["items"] if item["action"] == "SAFE_AUTOFILL"]
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError("Install Playwright and run `playwright install chromium` in the local companion environment") from error
    profile_dir = Path(browser_profile_dir).expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    filled = 0
    with sync_playwright() as api:
        context = api.chromium.launch_persistent_context(str(profile_dir), headless=False)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(plan["page_url"], wait_until="domcontentloaded")
        body = page.locator("body").inner_text().casefold()
        if any(signal in body for signal in LocalPlaywrightWorker.BLOCK_SIGNALS):
            print("Human action required: complete the access check in the browser, then press Enter here.")
            input()
        for item in safe_items:
            locator = playwright_locator(page, item.locator).first
            if not locator.count():
                print(f"Skipped missing field: {item.label}")
                continue
            current_label = (
                locator.get_attribute("aria-label")
                or locator.get_attribute("name")
                or locator.evaluate(
                    "el => { const id = el.id; if (id) { const node = document.querySelector(`label[for='${id}']`); if (node) return node.innerText; } const wrap = el.closest('label'); return wrap ? wrap.innerText : ''; }")
                or ""
            )
            if not labels_match(item.label, current_label):
                print(f"Skipped label mismatch for {item.label}: found {current_label!r}")
                continue
            locator.fill(item.value or "")
            filled += 1
        print(f"Filled {filled} approved safe fields. Review every value in the browser.")
        print("Submission, declarations, MFA, payment, file choosers, and sensitive fields remain manual.")
        input("Press Enter after you have reviewed the page. The companion will close without submitting. ")
        context.close()
    with transaction(db_path) as db:
        db.execute("UPDATE browser_fill_plans SET status='FILLED' WHERE id=? AND status='APPROVED'", (plan_id,))
        db.execute("""INSERT INTO workload_events
            (event_type,count_value,entity_type,entity_id,metadata_json,created_at)
            VALUES('BROWSER_FIELDS_AUTOFILLED',?,'BROWSER_FILL_PLAN',?,'{}',?)""", (filled, plan_id, utc_now()))
    return filled
