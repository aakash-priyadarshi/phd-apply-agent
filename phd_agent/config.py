"""Deterministic settings for the existing Streamlit process."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from phd_agent.paths import APP_ROOT


@dataclass(frozen=True)
class Settings:
    data_dir: Path

    @property
    def database_path(self) -> Path:
        return self.data_dir / "phd_outreach.db"

    @property
    def cv_path(self) -> Path:
        return self.data_dir / "documents" / "uploaded_cv.pdf"

    @property
    def profile_path(self) -> Path:
        return self.data_dir / "research_profile.txt"

    @property
    def credentials_path(self) -> Path:
        return self.data_dir / "credentials.json"

    @property
    def gmail_token_path(self) -> Path:
        return self.data_dir / "gmail_token.json"

    @property
    def targets_path(self) -> Path:
        return self.data_dir / "PhD_Targets.csv"

    @property
    def results_path(self) -> Path:
        return self.data_dir / "PhD_Results.csv"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "phd_outreach.log"

    @property
    def auto_send_enabled(self) -> bool:
        # Campaign automation needs the later quality gate and contact history.
        # Do not enable the legacy bulk send paths through an environment flag.
        return False


def load_settings(root: Path = APP_ROOT) -> Settings:
    load_dotenv(root / ".env", override=False)
    configured = os.environ.get("PHD_AGENT_DATA_DIR")
    data_dir = Path(configured).expanduser() if configured else root / "data"
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    resolved = data_dir.resolve()
    if resolved.is_relative_to(root.resolve()) and not resolved.is_relative_to((root / "data").resolve()):
        raise ValueError("PHD_AGENT_DATA_DIR inside the repository must be under ignored data/")
    return Settings(data_dir=resolved)
