"""Offline smoke of the preserved legacy operator path; never calls Gmail."""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import openai
import requests
from PyPDF2 import PdfWriter
from streamlit.testing.v1 import AppTest


class FakeCompletions:
    def create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "Extract professor information" in prompt:
            content = json.dumps([{
                "name": "Dr Test", "email": "test@example.invalid",
                "research_interests": "AI evaluation", "profile_url": "https://example.invalid/faculty",
                "alignment_score": 8, "collaboration_potential": "Related evaluation methods",
            }])
        elif "Write a professional" in prompt:
            content = json.dumps({
                "subject": "Research enquiry",
                "body": "Dear Dr Test,\\n\\nI am interested in your work.\\n\\nBest regards,\\nTest Applicant",
                "key_points": [], "tone_analysis": "professional",
            })
        else:
            content = "I study AI evaluation and reliable systems."
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class FakeResponse:
    status_code = 200
    content = b"<html><body>Dr Test researches AI evaluation.</body></html>"

    def raise_for_status(self):
        return None


def test_legacy_upload_discover_draft_edit_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("PHD_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PHD_AGENT_ENV", "development")
    monkeypatch.setenv("PHD_AGENT_AUTH_DISABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setenv("USER_NAME", "Test Applicant")
    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setattr(requests.Session, "get", lambda *args, **kwargs: FakeResponse())
    (tmp_path / "PhD_Targets.csv").write_text(
        "University Name,Country,Departments to Search,Priority,Notes\n"
        "University of Oxford,UK,CS,High,Test\n", encoding="utf-8",
    )
    pdf = PdfWriter()
    pdf.add_blank_page(width=595, height=842)
    stream = io.BytesIO()
    pdf.write(stream)

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py", default_timeout=60).run()
    assert not app.exception
    app.toggle[0].set_value(True).run()
    next(radio for radio in app.radio if radio.label == "Advanced area").set_value("Legacy outreach").run()
    assert not app.exception
    assert not app.get("file_uploader")
    next(button for button in app.button if button.label == "Open legacy workspace").click().run()
    assert not app.exception
    app.get("file_uploader")[0].upload("test-cv.pdf", stream.getvalue(), "application/pdf").run()
    assert not app.exception
    app.button(key="analyze_cv_btn").click().run()
    assert not app.exception
    assert (tmp_path / "documents" / "uploaded_cv.pdf").read_bytes() == stream.getvalue()
    assert (tmp_path / "research_profile.txt").is_file()

    next(button for button in app.button if button.label == "🔍 Run Stage 1").click().run()
    assert not app.exception
    with sqlite3.connect(tmp_path / "phd_outreach.db") as db:
        row = db.execute("SELECT id, status FROM professors WHERE name = 'Dr Test'").fetchone()
    assert row and row[1] == "verified"

    app.button(key=f"gen_email_{row[0]}").click().run()
    assert not app.exception
    with sqlite3.connect(tmp_path / "phd_outreach.db") as db:
        assert db.execute("SELECT draft_email_subject FROM professors WHERE id = ?", (row[0],)).fetchone()[0] == "Research enquiry"

    app.button(key=f"preview_{row[0]}").click().run()
    assert not app.exception
    assert any(widget.key == f"modal_subject_{row[0]}" for widget in app.text_input)
    app.text_input(key=f"inline_subject_{row[0]}").set_value("Edited research enquiry")
    next(button for button in app.button if button.label == "Save draft changes").click().run()
    assert not app.exception
    with sqlite3.connect(tmp_path / "phd_outreach.db") as db:
        assert db.execute("SELECT draft_email_subject FROM professors WHERE id = ?", (row[0],)).fetchone()[0] == "Edited research enquiry"
        assert db.execute("SELECT COUNT(*) FROM professors WHERE status = 'email_sent'").fetchone()[0] == 0
