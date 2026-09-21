"""Optional real-send integration test; skipped in normal tests and CI."""

import os

import pytest

from gmail_manager import GmailManager


@pytest.mark.skipif(
    os.getenv("PHD_AGENT_SEND_REAL_EMAIL") != "1"
    or not os.getenv("PHD_AGENT_TEST_RECIPIENT"),
    reason="Real send needs both PHD_AGENT_SEND_REAL_EMAIL=1 and a test recipient",
)
def test_email_sending():
    manager = GmailManager()
    assert manager.service is not None
    result = manager.send_email(
        to_email=os.environ["PHD_AGENT_TEST_RECIPIENT"],
        subject="PhD agent integration test",
        body="This is an explicitly requested integration test.",
        from_name="PhD Agent Test",
    )
    assert result["success"]
