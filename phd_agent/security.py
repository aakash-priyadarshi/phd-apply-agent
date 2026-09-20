"""Name-only warning for private runtime files accidentally tracked by Git."""

from __future__ import annotations

import subprocess
from pathlib import Path

from phd_agent.paths import APP_ROOT


def tracked_private_paths(root: Path = APP_ROOT) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    names = [name.decode("utf-8", errors="replace") for name in result.stdout.split(b"\0") if name]
    private: list[str] = []
    for name in names:
        path = Path(name)
        lower = path.name.lower()
        if (
            path.parts[0].lower() == "data"
            or lower.startswith("credentials") and lower.endswith(".json")
            or lower in {
                "uploaded_cv.pdf", "research_profile.txt", ".env",
                "phd_targets.csv", "phd_results.csv", "webdriver_config.json",
            }
            or lower.endswith((".db", ".sqlite", ".pickle", ".pdf"))
            or "token" in lower and lower.endswith(".json")
        ):
            private.append(name)
    return private
