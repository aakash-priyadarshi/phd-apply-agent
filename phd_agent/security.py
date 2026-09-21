"""Name-only warning for private runtime files accidentally tracked by Git."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from phd_agent.paths import APP_ROOT


class GitInspectionFailed(RuntimeError):
    """Git could not be queried for tracked private files."""


def tracked_private_paths(root: Path = APP_ROOT) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GitInspectionFailed("Git could not be inspected for tracked private files") from error

    names = [name.decode("utf-8", errors="replace") for name in result.stdout.split(b"\0") if name]
    private: list[str] = []
    for name in names:
        path = Path(name)
        lower = path.name.lower()
        if (
            path.parts[0].lower() == "data"
            or "documents" in (part.lower() for part in path.parts)
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


def restrict_private_file(path: Path) -> None:
    path = Path(path)
    if os.name == "nt":
        user = os.environ.get("USERNAME") or os.getlogin()
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(R,W)"],
            check=False, capture_output=True,
        )
        return
    os.chmod(path, 0o600)


def write_restricted_file(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        # Restrict the temporary file before credential bytes are written.
        restrict_private_file(Path(tmp_name))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    restrict_private_file(path)
