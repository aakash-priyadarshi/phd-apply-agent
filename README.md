# PhD application and outreach console

This is a local Streamlit application with a manual application ledger and Document Vault. The earlier CV analysis, professor discovery, email drafting/editing, and individual Gmail sender remain available under **Legacy outreach**. The [2026–27 implementation plan](docs/implementation-plan-2026-27.md) describes the remaining staged work.

## Run locally

Use Python 3.11 or newer. On Windows, run `setup.bat`, copy `.env.example` to `.env`, fill in your values, then run `start_phd_outreach.bat`.

On macOS or Linux:

```sh
sh setup.sh
cp .env.example .env
# Edit .env locally.
.venv/bin/python -m streamlit run streamlit_app.py
```

For tests, install `requirements-dev.txt` in the same environment and run `python -m pytest -q`. Normal tests do not send email. The two Gmail integration tests require explicit environment flags and a test recipient.

## Local data

All runtime files are kept outside Git under `data/` by default:

```text
data/
  phd_outreach.db
  PhD_Targets.csv
  PhD_Results.csv
  research_profile.txt
  credentials.json
  gmail_token.json
  documents/uploaded_cv.pdf
  documents/objects/<SHA-256 prefix>/<SHA-256>
  backups/phd_outreach-legacy*.db
  phd_outreach.log
```

The app copies non-credential files from the old repository-root layout into `data/` on first run without overwriting newer local files. Before this upgrade, the existing database was copied byte-for-byte to `data/backups/` and its active copy was verified. Back up `data/` regularly. `PHD_AGENT_DATA_DIR` can point to another local directory; relative values are resolved from the repository root. A custom directory inside the repository must be under the ignored `data/` tree.

The application ledger uses additive SQLite migrations recorded in `schema_migrations`. Slice 1 adds the ledger and Vault tables without changing legacy professor rows. The ignored Slice 0 database backup remains available for rollback: stop the app, copy `data/backups/phd_outreach-legacy-1bcb632.db` to a separate location, and replace `data/phd_outreach.db` with that copy only if you intend to discard all Slice 1 records. Do not restore over a running app. Vault files under `data/documents/` need their own backup alongside the database.

## Manual application workflow

Open **Application CMS** in the sidebar. You can use it without an OpenAI key or Gmail authorization.

1. Add an official source snapshot with its URL, excerpt, and verification state.
2. Add a programme or opportunity, then create an application and record its next action.
3. Add sourced application, funding, document, and referee deadlines. Add requirements with `FORMAL_APPLICATION`, `FACULTY_OUTREACH`, or `REPLY_REQUEST` context. Keep unresolved items `UNKNOWN`.
4. Upload original files in **Document Vault**. Matching SHA-256 bytes show the existing version; source bytes are never overwritten. Review and approve a version, then link it to a matching requirement from the application detail screen. The same version can be linked to multiple applications.
5. Add tasks and referees. The **Today** screen shows deadlines, missing required items, unknown requirements, overdue tasks, and document approval or expiry alerts.

Administrative readiness is shown as separate counts: required items completed, unknown requirements, and conditional requirements. It is a document/process checklist, not an admission assessment. The application ledger is manual in this slice; it does not crawl programme sites, choose packages, generate documents, or submit forms. `PASSPORT` and `GOVERNMENT_ID` default to `HIGHLY_SENSITIVE` and are local-only. No cloud backend is active.

The original root copies of the 2025 runtime files were removed from the working tree after verified local copies were made. Git still contains their earlier versions in history. This branch does **not** rewrite Git history.

## Gmail setup and credential rotation

The old OAuth client file was tracked in the public repository and has been preserved locally under `data/backups/`, not in the active credentials path. Create a new Desktop OAuth client in your Google Cloud project, disable or delete the old client, and place the new downloaded JSON at `data/credentials.json`. Review and revoke the old app grant in your Google account if it is no longer needed. Google describes [creating a Desktop client and using JSON tokens](https://developers.google.com/workspace/gmail/api/quickstart/python) and [credential and token handling](https://developers.google.com/identity/protocols/oauth2/policies).

The app never loads the old `gmail_token.pickle`. Gmail authorization creates `data/gmail_token.json`; treat it as a credential. The JSON token is not encrypted by this local release, so keep the data directory accessible only to your user account and protect the disk. If an old pickle token exists, remove it after you have confirmed the new authorization works. The app currently requests Gmail send and read access; read access supports connection status and the planned sent-history reconciliation.

Bulk send and generate-and-send controls are disabled in this release. Individual professor emails still require a deliberate click in the Streamlit UI. The reviewed outreach queue, factuality gate, and Gmail history protection are later slices.

## Current workflow

1. Upload or reuse a local CV, then generate the research profile.
2. Add target universities or reuse the copied target list.
3. Run Stage 1 discovery and inspect the results.
4. Generate and edit an email for one professor.
5. Send that individual email only after reviewing it and the CV attachment.

The 2025 discovery map is limited, and existing professor records need reverification for the new cycle. Do not treat an old `verified` status as current recruiting evidence.
