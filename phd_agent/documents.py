"""Local, immutable-by-version document storage and Vault operations."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
from datetime import date
from pathlib import Path
from typing import Protocol

from phd_agent.db import connect, migrate, transaction, utc_now


DOCUMENT_CLASSES = ("SOURCE", "MASTER", "GENERATED")
DOCUMENT_TYPES = (
    "CV", "SOP", "PERSONAL_STATEMENT", "RESEARCH_PROPOSAL", "RESEARCH_STATEMENT",
    "COVER_LETTER", "DEGREE_CERTIFICATE", "TRANSCRIPT", "MARKSHEET",
    "ENGLISH_TEST", "STANDARDIZED_TEST", "PASSPORT", "GOVERNMENT_ID", "PHOTO",
    "SIGNATURE", "PATENT_DOCUMENT", "PUBLICATION", "CERTIFICATE",
    "FUNDING_DOCUMENT", "APPLICATION_FORM", "PORTAL_CONFIRMATION",
    "PAYMENT_RECEIPT", "OTHER",
)
SENSITIVITIES = ("NORMAL", "CONFIDENTIAL", "HIGHLY_SENSITIVE")


class DocumentStorage(Protocol):
    def put(self, data: bytes) -> str: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def verify_hash(self, key: str, expected_sha256: str) -> bool: ...
    def export(self, key: str, destination: Path) -> Path: ...


# S3CompatibleDocumentStorage / Railway Bucket is a post-deployment enhancement,
# not a launch blocker. Keep this interface so document identities stay stable.


class LocalDocumentStorage:
    backend = "LOCAL"

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("Storage key escapes the local document store")
        return candidate

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        key = f"objects/{digest[:2]}/{digest}"
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not self.verify_hash(key, digest):
                raise OSError("Existing object does not match its hash")
            return key
        try:
            with path.open("xb") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
        except FileExistsError:
            if not self.verify_hash(key, digest):
                raise OSError("Concurrent object does not match its hash")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        if not self.verify_hash(key, digest):
            path.unlink(missing_ok=True)
            raise OSError("Stored object failed SHA-256 verification")
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def verify_hash(self, key: str, expected_sha256: str) -> bool:
        path = self._path(key)
        if not path.is_file():
            return False
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == expected_sha256

    def export(self, key: str, destination: Path) -> Path:
        source = self._path(key)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_file, destination.open("xb") as output:
            shutil.copyfileobj(input_file, output)
        return destination


def storage_policy(document_type: str, sensitivity: str) -> str:
    if document_type in {"PASSPORT", "GOVERNMENT_ID"} or sensitivity == "HIGHLY_SENSITIVE":
        return "LOCAL_ONLY"
    return "LOCAL_OR_CLOUD"


def _document_date(value: str | None, label: str) -> str | None:
    if value:
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{label} must use YYYY-MM-DD") from error
    return value


class DocumentVault:
    def __init__(self, db_path: Path | str, storage: DocumentStorage | None = None):
        self.db_path = Path(db_path)
        migrate(self.db_path)
        self.storage = storage or LocalDocumentStorage(self.db_path.parent / "documents")

    def find_duplicate(self, data: bytes) -> dict | None:
        digest = hashlib.sha256(data).hexdigest()
        with connect(self.db_path) as db:
            row = db.execute("""SELECT v.id AS version_id, v.document_id, v.version_number,
                d.title, d.document_class, d.document_type, v.storage_key, v.sha256
                FROM document_versions v JOIN documents d ON d.id = v.document_id
                WHERE v.sha256 = ? ORDER BY v.id LIMIT 1""", (digest,)).fetchone()
        if row and self.storage.verify_hash(row["storage_key"], digest):
            return dict(row)
        return None

    def upload(
        self, data: bytes, filename: str, document_class: str, document_type: str,
        title: str, *, document_id: int | None = None, issuer: str | None = None,
        degree_programme: str | None = None, issue_date: str | None = None,
        expiry_date: str | None = None, sensitivity: str | None = None,
        notes: str = "",
    ) -> dict:
        if not data:
            raise ValueError("Empty files cannot be uploaded")
        if document_class not in DOCUMENT_CLASSES or document_type not in DOCUMENT_TYPES:
            raise ValueError("Invalid document class or type")
        current = None
        if document_id is not None:
            with connect(self.db_path) as db:
                current = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            if not current:
                raise ValueError("Document does not exist")
            if (current["document_class"], current["document_type"]) != (document_class, document_type):
                raise ValueError("A new version must keep the same class and type")
            issuer = issuer if issuer is not None else current["issuer"]
            degree_programme = degree_programme if degree_programme is not None else current["degree_programme"]
            issue_date = issue_date if issue_date is not None else current["issue_date"]
            expiry_date = expiry_date if expiry_date is not None else current["expiry_date"]
        sensitivity = sensitivity or (current["sensitivity"] if current else (
            "HIGHLY_SENSITIVE" if document_type in {"PASSPORT", "GOVERNMENT_ID"} else "NORMAL"
        ))
        if document_type in {"PASSPORT", "GOVERNMENT_ID"}:
            sensitivity = "HIGHLY_SENSITIVE"
        if sensitivity not in SENSITIVITIES:
            raise ValueError("Invalid sensitivity")
        issue_date = _document_date(issue_date, "Issue date")
        expiry_date = _document_date(expiry_date, "Expiry date")
        duplicate = self.find_duplicate(data)
        if duplicate:
            return {"status": "duplicate", **duplicate}
        original_filename = Path(filename.replace("\\", "/")).name
        if not original_filename or original_filename in {".", ".."}:
            raise ValueError("A filename is required")
        title = title.strip() or (current["title"] if current else "")
        if not title:
            raise ValueError("A title is required")
        digest = hashlib.sha256(data).hexdigest()
        key = self.storage.put(data)
        now = utc_now()
        with transaction(self.db_path) as db:
            if document_id is None:
                cursor = db.execute("""INSERT INTO documents
                    (document_class, document_type, title, issuer, degree_programme,
                     issue_date, expiry_date, sensitivity, permitted_storage_policy,
                     notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                    document_class, document_type, title, issuer,
                    degree_programme, issue_date, expiry_date, sensitivity,
                    storage_policy(document_type, sensitivity), notes, now, now,
                ))
                document_id = cursor.lastrowid
                version_number = 1
                parent_version_id = None
            else:
                latest = db.execute("""SELECT id, version_number FROM document_versions
                    WHERE document_id = ? ORDER BY version_number DESC LIMIT 1""", (document_id,)).fetchone()
                version_number = latest["version_number"] + 1
                parent_version_id = latest["id"]
                db.execute("""UPDATE documents SET issuer = ?, degree_programme = ?,
                    issue_date = ?, expiry_date = ?, sensitivity = ?, permitted_storage_policy = ?,
                    updated_at = ? WHERE id = ?""", (
                    issuer, degree_programme, issue_date, expiry_date, sensitivity,
                    storage_policy(document_type, sensitivity), now, document_id,
                ))
            suffix = Path(original_filename).suffix.lower()
            canonical_filename = f"document-{document_id}-v{version_number}{suffix}"
            cursor = db.execute("""INSERT INTO document_versions
                (document_id, version_number, parent_version_id, original_filename,
                 canonical_filename, mime_type, byte_size, sha256, storage_backend,
                 storage_key, uploaded_at, notes, issuer, degree_programme,
                 issue_date, expiry_date, sensitivity, permitted_storage_policy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                document_id, version_number, parent_version_id, original_filename,
                canonical_filename, mimetypes.guess_type(original_filename)[0] or "application/octet-stream",
                len(data), digest, self.storage.backend, key, now, notes,
                issuer, degree_programme, issue_date, expiry_date, sensitivity,
                storage_policy(document_type, sensitivity),
            ))
            version_id = cursor.lastrowid
        return {
            "status": "created", "document_id": document_id,
            "version_id": version_id, "version_number": version_number,
            "sha256": digest, "storage_key": key,
        }

    def update_document(self, document_id: int, **changes) -> None:
        allowed = {
            "title", "issuer", "degree_programme", "issue_date", "expiry_date",
            "sensitivity", "notes",
        }
        if set(changes) - allowed:
            raise ValueError("Unsupported document metadata field")
        for field in ("issue_date", "expiry_date"):
            if field in changes:
                changes[field] = _document_date(changes[field], field.replace("_", " ").title())
        with transaction(self.db_path) as db:
            current = db.execute(
                "SELECT document_type, sensitivity FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if not current:
                raise ValueError("Document does not exist")
            sensitivity = changes.get("sensitivity", current["sensitivity"])
            if current["document_type"] in {"PASSPORT", "GOVERNMENT_ID"}:
                sensitivity = "HIGHLY_SENSITIVE"
                changes["sensitivity"] = sensitivity
            if sensitivity not in SENSITIVITIES:
                raise ValueError("Invalid sensitivity")
            changes["permitted_storage_policy"] = storage_policy(current["document_type"], sensitivity)
            changes["updated_at"] = utc_now()
            assignments = ", ".join(f"{field} = ?" for field in changes)
            db.execute(
                f"UPDATE documents SET {assignments} WHERE id = ?",
                (*changes.values(), document_id),
            )
            version_changes = {field: changes[field] for field in (
                "issuer", "degree_programme", "issue_date", "expiry_date",
                "sensitivity", "permitted_storage_policy",
            ) if field in changes}
            if version_changes:
                latest = db.execute("""SELECT id FROM document_versions WHERE document_id = ?
                    ORDER BY version_number DESC LIMIT 1""", (document_id,)).fetchone()
                version_assignments = ", ".join(f"{field} = ?" for field in version_changes)
                db.execute(f"UPDATE document_versions SET {version_assignments} WHERE id = ?",
                           (*version_changes.values(), latest["id"]))

    def set_approval(self, version_id: int, approved: bool, approved_by: str | None = None) -> None:
        version = self.get_version(version_id)
        if not version:
            raise ValueError("Version does not exist")
        if approved and version["document_type"] in {"TRANSCRIPT", "DEGREE_CERTIFICATE"} and version["verification_state"] == "NEEDS_REVIEW":
            raise ValueError("Review academic-document authenticity and type before approval")
        if approved and not self.storage.verify_hash(version["storage_key"], version["sha256"]):
            raise OSError("Cannot approve a missing or corrupt document version")
        with transaction(self.db_path) as db:
            cursor = db.execute("""UPDATE document_versions
                SET approval_state = ?, approved_by = ?, approved_at = ? WHERE id = ?""", (
                "APPROVED" if approved else "PENDING",
                approved_by if approved else None,
                utc_now() if approved else None,
                version_id,
            ))
            if cursor.rowcount != 1:
                raise ValueError("Version does not exist")

    def set_verification(self, version_id: int, state: str, reviewer: str) -> None:
        if state not in {"UNVERIFIED", "VERIFIED", "NEEDS_REVIEW"} or not reviewer.strip():
            raise ValueError("Verification state and reviewer required")
        with transaction(self.db_path) as db:
            version = db.execute("""SELECT v.notes,d.document_class FROM document_versions v
                JOIN documents d ON d.id=v.document_id WHERE v.id=?""", (version_id,)).fetchone()
            if not version:
                raise ValueError("Version does not exist")
            if version["document_class"] != "SOURCE":
                raise ValueError("Verification review applies to original source versions")
            note = (version["notes"] or "") + f"\nVerification review {utc_now()}: {state} by {reviewer.strip()}"
            db.execute("UPDATE document_versions SET verification_state=?,notes=? WHERE id=?",
                       (state,note.strip(),version_id))

    def list_documents(self) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute("""SELECT d.*, v.id AS latest_version_id,
                v.version_number, v.original_filename, v.sha256, v.approval_state,
                v.storage_backend, v.storage_key
                FROM documents d JOIN document_versions v ON v.document_id = d.id
                WHERE v.version_number = (SELECT MAX(v2.version_number)
                    FROM document_versions v2 WHERE v2.document_id = d.id)
                ORDER BY d.id DESC""")]

    def list_versions(self, document_id: int) -> list[dict]:
        with connect(self.db_path) as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM document_versions WHERE document_id = ? ORDER BY version_number DESC",
                (document_id,),
            )]

    def get_version(self, version_id: int) -> dict | None:
        with connect(self.db_path) as db:
            row = db.execute("""SELECT v.*, d.document_type, d.document_class, d.title
                FROM document_versions v JOIN documents d ON d.id = v.document_id
                WHERE v.id = ?""", (version_id,)).fetchone()
            return dict(row) if row else None

    def link_to_application(
        self, application_id: int, requirement_id: int, version_id: int,
        notes: str = "",
    ) -> None:
        version = self.get_version(version_id)
        if not version:
            raise ValueError("Document version does not exist")
        with transaction(self.db_path) as db:
            requirement = db.execute(
                "SELECT * FROM requirements WHERE id = ? AND application_id = ?",
                (requirement_id, application_id),
            ).fetchone()
            if not requirement:
                raise ValueError("Requirement does not belong to application")
            if not requirement["normalized_document_type"]:
                raise ValueError("Requirement does not request a document")
            if requirement["normalized_document_type"] != version["document_type"]:
                raise ValueError("Document type does not match requirement")
            if version["expiry_date"] and version["expiry_date"] < date.today().isoformat():
                state = "NEEDS_UPDATE"
            elif version["approval_state"] != "APPROVED":
                state = "NEEDS_APPROVAL"
            elif not self.storage.verify_hash(version["storage_key"], version["sha256"]):
                state = "NEEDS_UPDATE"
            else:
                state = "AVAILABLE"
            db.execute("""INSERT INTO application_documents
                (application_id, requirement_id, document_version_id, document_state,
                 source_evidence_id, notes, linked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(application_id, requirement_id) DO UPDATE SET
                document_version_id = excluded.document_version_id,
                document_state = excluded.document_state,
                source_evidence_id = excluded.source_evidence_id,
                notes = excluded.notes,
                linked_at = excluded.linked_at""", (
                application_id, requirement_id, version_id, state,
                requirement["source_evidence_id"], notes, utc_now(),
            ))

    def unlink_from_application(self, application_id: int, requirement_id: int) -> None:
        with transaction(self.db_path) as db:
            db.execute(
                "DELETE FROM application_documents WHERE application_id = ? AND requirement_id = ?",
                (application_id, requirement_id),
            )
