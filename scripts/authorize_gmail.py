"""Explicit local OAuth setup for the rotated Gmail client; never runs on app startup."""

from __future__ import annotations

import json
import os

from google_auth_oauthlib.flow import InstalledAppFlow

from phd_agent.config import load_settings
from phd_agent.gmail_gateway import GmailGateway


SCOPES = ["https://www.googleapis.com/auth/gmail.send",
          "https://www.googleapis.com/auth/gmail.readonly"]


def main() -> None:
    data_dir = load_settings().data_dir
    client_path = data_dir / "credentials.json"
    old_path = data_dir / "backups" / "credentials-legacy.json"
    if not client_path.is_file():
        raise SystemExit("Place the rotated Desktop OAuth client at data/credentials.json first")
    current = json.loads(client_path.read_text(encoding="utf-8"))
    current_id = (current.get("installed") or current.get("web") or {}).get("client_id")
    if not current_id:
        raise SystemExit("OAuth client JSON has no client_id")
    if old_path.is_file():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        old_id = (old.get("installed") or old.get("web") or {}).get("client_id")
        if current_id == old_id:
            raise SystemExit("The active OAuth client is the previously committed client; rotate it first")
    flow = InstalledAppFlow.from_client_secrets_file(str(client_path), SCOPES)
    credentials = flow.run_local_server(port=0)
    token_path = data_dir / "gmail_token.json"
    temporary = token_path.with_suffix(".json.tmp")
    temporary.write_text(credentials.to_json(), encoding="utf-8")
    os.replace(temporary, token_path)
    if os.name != "nt":
        token_path.chmod(0o600)
    if not GmailGateway(data_dir=data_dir).credentials_ready():
        raise SystemExit("OAuth completed, but the rotated client/token pair did not validate")
    print("Gmail OAuth setup complete. Import and review Sent history before any manual send.")


if __name__ == "__main__":
    main()
