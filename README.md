# PhD application and outreach console

This is the existing local Streamlit application for CV analysis, professor discovery, email drafting and editing, and individual Gmail sends. The [2026–27 implementation plan](docs/implementation-plan-2026-27.md) describes its staged upgrade into an application and document CMS.

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
  backups/phd_outreach-legacy*.db
  phd_outreach.log
```

The app copies non-credential files from the old repository-root layout into `data/` on first run without overwriting newer local files. Before this upgrade, the existing database was copied byte-for-byte to `data/backups/` and its active copy was verified. Back up `data/` regularly. `PHD_AGENT_DATA_DIR` can point to another local directory; relative values are resolved from the repository root.

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
