"""Optional manual Gmail authentication check; skipped in normal tests."""

import os

import pytest

from gmail_manager import GmailManager


@pytest.mark.skipif(
    os.getenv("PHD_AGENT_TEST_GMAIL_AUTH") != "1",
    reason="Set PHD_AGENT_TEST_GMAIL_AUTH=1 for an interactive Gmail check",
)
def test_gmail_auth():
    manager = GmailManager()
    assert manager.service is not None
