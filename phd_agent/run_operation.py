"""Detached worker entry point. Arguments contain only the persistent database path and operation ID."""

from __future__ import annotations

import sys
from pathlib import Path

from phd_agent.operations import OperationService


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or not args[1].isdigit():
        raise SystemExit("Usage: python -m phd_agent.run_operation DATABASE OPERATION_ID")
    OperationService(Path(args[0])).run(int(args[1]))


if __name__ == "__main__":
    main()
