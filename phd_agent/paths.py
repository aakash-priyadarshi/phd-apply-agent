"""Runtime paths and a non-destructive migration from the 2025 layout."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_data_layout(data_dir: Path, root: Path = APP_ROOT) -> list[str]:
    """Copy old runtime files into ignored storage without overwriting user data.

    Old files are left in place for manual cleanup. The database receives an
    untouched backup before the app uses its new active copy.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "documents").mkdir(exist_ok=True)
    (data_dir / "backups").mkdir(exist_ok=True)

    legacy = {
        "phd_outreach.db": "phd_outreach.db",
        "uploaded_cv.pdf": "documents/uploaded_cv.pdf",
        "research_profile.txt": "research_profile.txt",
        "PhD_Targets.csv": "PhD_Targets.csv",
        "PhD_Results.csv": "PhD_Results.csv",
    }
    copied: list[str] = []
    db_source = root / "phd_outreach.db"
    backup = data_dir / "backups" / "phd_outreach-legacy.db"
    if db_source.is_file() and not backup.exists():
        shutil.copy2(db_source, backup)
        if _digest(db_source) != _digest(backup):
            backup.unlink()
            raise OSError("Legacy database backup failed verification")
        copied.append("legacy database backup")

    for old_name, new_name in legacy.items():
        source = root / old_name
        destination = data_dir / new_name
        if not source.is_file() or destination.exists():
            continue
        shutil.copy2(source, destination)
        if _digest(source) != _digest(destination):
            destination.unlink()
            raise OSError(f"Migration failed verification for {old_name}")
        copied.append(old_name)
    return copied
