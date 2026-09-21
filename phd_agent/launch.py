"""Prepare persistent runtime and OIDC secrets, then exec Streamlit."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from phd_agent.paths import APP_ROOT
from phd_agent.runtime import StartupError, prepare_runtime


def streamlit_command(port: str) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true",
    ]


def main(
    environ: Mapping[str, str] | None = None,
    *,
    root: Path = APP_ROOT,
    secrets_path: Path | None = None,
    exec_fn: Callable[[str, Sequence[str]], object] | None = None,
) -> None:
    env = os.environ if environ is None else environ
    try:
        prepare_runtime(root, env if environ is not None else None, secrets_path=secrets_path)
    except StartupError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
    port = (env.get("PORT") or "").strip()
    if not port.isdigit():
        raise SystemExit("PORT must be provided by the host")
    argv = streamlit_command(port)
    runner = exec_fn or os.execvp
    runner(argv[0], argv)


if __name__ == "__main__":
    main()
