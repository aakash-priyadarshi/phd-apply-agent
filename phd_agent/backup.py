"""Local database and Document Vault backup/restore. Credentials stay out by default."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from phd_agent.db import migrate, transaction, utc_now
from phd_agent.documents import DocumentVault
from phd_agent.security import restrict_private_file


CREDENTIAL_NAMES = {"credentials.json", "gmail_token.json", "gmail_token.pickle", ".env"}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _private_mkdir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)


def _contained_path(root: Path, relative) -> Path:
    if not isinstance(relative, str) or not relative.strip() or relative.strip() != relative:
        raise ValueError("Backup path is missing or padded")
    candidate = Path(relative)
    if (candidate.is_absolute() or candidate.drive or candidate.root or not candidate.parts
            or any(part in {".", ".."} for part in candidate.parts)):
        raise ValueError("Backup path escapes the archive")
    root = root.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Backup path escapes the archive")
    return resolved


class BackupService:
    def __init__(self, db_path: Path | str, data_dir: Path | str | None = None):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.data_dir = Path(data_dir) if data_dir else self.db_path.parent
        self.vault = DocumentVault(self.db_path)

    def create_backup(self, destination: Path | str, actor: str, *,
                      include_credentials: bool = False) -> dict:
        if not actor.strip():
            raise ValueError("Backup operator required")
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError("Choose a new backup directory")
        _private_mkdir(destination)
        db_copy = destination / "phd_outreach.db"
        source = sqlite3.connect(str(self.db_path))
        try:
            dest = sqlite3.connect(str(db_copy))
            try:
                source.backup(dest)
            finally:
                dest.close()
        finally:
            source.close()
        restrict_private_file(db_copy)
        files = []
        vault_root = Path(self.vault.storage.root)
        if vault_root.is_dir():
            for path in sorted(p for p in vault_root.rglob("*") if p.is_file()):
                relative = path.relative_to(self.data_dir)
                target = _contained_path(destination, relative.as_posix())
                _private_mkdir(target.parent)
                shutil.copy2(path, target)
                files.append({"path": relative.as_posix(), "sha256": _digest(target)})
        excluded = []
        if include_credentials:
            for name in sorted(CREDENTIAL_NAMES):
                source_file = self.data_dir / name
                if source_file.is_file():
                    target = _contained_path(destination, name)
                    shutil.copy2(source_file, target)
                    restrict_private_file(target)
                    files.append({"path": name, "sha256": _digest(target)})
        else:
            excluded = sorted(name for name in CREDENTIAL_NAMES if (self.data_dir / name).is_file())
        manifest = {
            "created_at": utc_now(),
            "actor": actor.strip(),
            "db_sha256": _digest(db_copy),
            "vault_file_count": len(files),
            "include_credentials": include_credentials,
            "excluded_credentials": excluded,
            "files": files,
        }
        (destination / "manifest.json").write_text(_dump(manifest), encoding="utf-8")
        with transaction(self.db_path) as db:
            db.execute("""INSERT INTO backup_runs
                (kind,archive_path,db_sha256,vault_file_count,include_credentials,manifest_json,actor,created_at)
                VALUES('BACKUP',?,?,?,?,?,?,?)""",
                (str(destination.resolve()), manifest["db_sha256"], len(files),
                 int(include_credentials), _dump(manifest), actor.strip(), utc_now()))
        return manifest | {"path": str(destination.resolve())}

    def restore(self, archive: Path | str, destination_dir: Path | str, actor: str, *,
                replace_existing: bool = False) -> dict:
        if not actor.strip():
            raise ValueError("Restore operator required")
        archive = Path(archive)
        destination_dir = Path(destination_dir)
        manifest_path = archive / "manifest.json"
        db_copy = archive / "phd_outreach.db"
        if not manifest_path.is_file() or not db_copy.is_file():
            raise ValueError("Backup archive is missing the database or manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _digest(db_copy) != manifest["db_sha256"]:
            raise ValueError("Backup database hash does not match the manifest")
        verified = []
        for item in manifest.get("files", []):
            path = _contained_path(archive, item.get("path"))
            if not path.is_file() or _digest(path) != item["sha256"]:
                raise ValueError("Backup file hash does not match the manifest: " + str(item.get("path")))
            verified.append(item)
        target_db = destination_dir / "phd_outreach.db"
        dest_occupied = destination_dir.exists() and any(destination_dir.iterdir())
        if dest_occupied and not replace_existing:
            raise FileExistsError("Destination database exists; pass replace_existing to overwrite")
        if dest_occupied:
            shutil.rmtree(destination_dir)
        _private_mkdir(destination_dir)
        shutil.copy2(db_copy, target_db)
        restrict_private_file(target_db)
        restored = []
        for item in verified:
            if Path(item["path"]).name in CREDENTIAL_NAMES and not manifest.get("include_credentials"):
                continue
            source = _contained_path(archive, item["path"])
            target = _contained_path(destination_dir, item["path"])
            _private_mkdir(target.parent)
            shutil.copy2(source, target)
            if Path(item["path"]).name in CREDENTIAL_NAMES:
                restrict_private_file(target)
            restored.append(item["path"])
        record = {
            "restored_at": utc_now(),
            "actor": actor.strip(),
            "archive": str(archive.resolve()),
            "destination": str(destination_dir.resolve()),
            "db_sha256": manifest["db_sha256"],
            "files": restored,
        }
        if target_db.resolve() != self.db_path.resolve():
            migrate(target_db)
            with transaction(target_db) as db:
                db.execute("""INSERT INTO backup_runs
                    (kind,archive_path,db_sha256,vault_file_count,include_credentials,manifest_json,actor,created_at)
                    VALUES('RESTORE',?,?,?,?,?,?,?)""",
                    (str(archive.resolve()), manifest["db_sha256"], len(restored),
                     int(manifest.get("include_credentials", False)), _dump(record), actor.strip(), utc_now()))
        return record
