"""Run an approved browser fill plan locally; never submits a university form."""

from __future__ import annotations

import argparse
from pathlib import Path

from phd_agent.browser_worker import execute_approved_plan
from phd_agent.config import load_settings, refuse_hosted_scripts


def main() -> None:
    refuse_hosted_scripts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_id", type=int)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--browser-profile", type=Path)
    args = parser.parse_args()
    settings = load_settings()
    database = args.database or settings.database_path
    browser_profile = args.browser_profile or settings.data_dir / "browser-profile"
    execute_approved_plan(database, args.plan_id, browser_profile_dir=browser_profile)


if __name__ == "__main__":
    main()
