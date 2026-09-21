"""Opt-in Gmail gateway; never starts OAuth or retries a send implicitly."""

from __future__ import annotations

import base64
import json
import mimetypes
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from phd_agent.config import load_settings


class GmailUnavailable(RuntimeError):
    pass


class GmailGateway:
    def __init__(self, *, data_dir: Path | None = None, service=None):
        self.data_dir = Path(data_dir or load_settings().data_dir)
        self._service = service

    def credentials_ready(self) -> bool:
        if self._service is not None:
            return True
        client = self.data_dir / "credentials.json"
        token = self.data_dir / "gmail_token.json"
        legacy = self.data_dir / "backups" / "credentials-legacy.json"
        if not client.is_file() or not token.is_file():
            return False
        try:
            active_json = json.loads(client.read_text(encoding="utf-8"))
            token_json = json.loads(token.read_text(encoding="utf-8"))
            active_id = (active_json.get("installed") or active_json.get("web") or {})["client_id"]
            if token_json.get("client_id") != active_id:
                return False
            if legacy.is_file():
                old_json = json.loads(legacy.read_text(encoding="utf-8"))
                old_id = (old_json.get("installed") or old_json.get("web") or {}).get("client_id")
                if active_id == old_id:
                    return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return True

    @property
    def service(self):
        if self._service is not None:
            return self._service
        if not self.credentials_ready():
            raise GmailUnavailable("Rotated OAuth client and a valid JSON token are required")
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        scopes = ["https://www.googleapis.com/auth/gmail.send",
                  "https://www.googleapis.com/auth/gmail.readonly"]
        credentials = Credentials.from_authorized_user_file(str(self.data_dir / "gmail_token.json"), scopes)
        if not credentials.valid:
            raise GmailUnavailable("Gmail token is absent or expired; reauthorize with the rotated client")
        self._service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
        return self._service

    def send(self, recipient: str, subject: str, body: str,
             attachments: list[tuple[str, bytes]], package_identity: str) -> dict:
        service = self.service  # Fail before entering the network send request.
        message = EmailMessage()
        message["To"] = recipient
        message["Subject"] = subject
        message["X-PhDAgent-Package"] = package_identity
        message.set_content(body)
        for filename, data in attachments:
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            maintype, subtype = mime.split("/", 1)
            message.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        result = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        if not result.get("id") or not result.get("threadId"):
            raise RuntimeError("Gmail response lacked a message or thread ID; reconcile Sent before retrying")
        return {"message_id": result["id"], "thread_id": result["threadId"]}

    def list_sent(self, after: datetime | None = None) -> list[dict]:
        """Read Sent history. A checkpoint limits later scans to newer messages."""
        service = self.service
        found = []
        page_token = None
        query = "in:sent"
        if after is not None:
            if after.tzinfo is None:
                after = after.replace(tzinfo=timezone.utc)
            query += f" after:{int(after.timestamp())}"
        while True:
            response = service.users().messages().list(userId="me", q=query,
                maxResults=100, pageToken=page_token).execute()
            for item in response.get("messages", []):
                full = service.users().messages().get(userId="me", id=item["id"], format="metadata",
                    metadataHeaders=["To", "Subject", "Date", "X-PhDAgent-Package"]).execute()
                headers = {h["name"].casefold(): h["value"] for h in full.get("payload", {}).get("headers", [])}
                milliseconds = int(full.get("internalDate", "0"))
                sent_at = datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()
                found.append({"recipient": headers.get("to", ""), "subject": headers.get("subject", ""),
                    "message_at": sent_at, "message_id": full["id"],
                    "thread_id": full.get("threadId", ""),
                    "package_identity": headers.get("x-phdagent-package", "")})
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return found

    def list_thread(self, thread_id: str) -> list[dict]:
        full = self.service.users().threads().get(userId="me", id=thread_id, format="metadata",
            metadataHeaders=["From", "To", "Subject"]).execute()
        result = []
        for message in full.get("messages", []):
            headers = {h["name"].casefold(): h["value"] for h in message.get("payload", {}).get("headers", [])}
            result.append({"message_id": message["id"], "thread_id": thread_id,
                "sender": headers.get("from", ""), "recipient": headers.get("to", ""),
                "subject": headers.get("subject", ""),
                "message_at": datetime.fromtimestamp(int(message.get("internalDate", "0"))/1000, timezone.utc).isoformat(),
                "labels": message.get("labelIds", [])})
        return result
