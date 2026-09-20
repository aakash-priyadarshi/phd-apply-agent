# PhD application and outreach console

This is a local Streamlit application with an application ledger, Document Vault, versioned applicant truth library, research directions, and source-backed discovery. The earlier CV analysis, professor discovery, email drafting/editing, and individual Gmail sender remain available under **Legacy outreach**. The [2026–27 implementation plan](docs/implementation-plan-2026-27.md) describes the staged work.

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

The application ledger uses additive SQLite migrations recorded in `schema_migrations`. Slice 2 adds profile, claim, research track, catalogue, faculty verification, publication, and assessment tables without changing legacy professor rows. The approved Slice 1 baseline is backed up under `data/backups/phd_outreach-slice1-eefd5b8c.db`. Stop the app before any restore and keep a separate copy of the current database first; restoring the baseline discards later local work. Vault files under `data/documents/` need their own backup alongside the database.

## Manual application workflow

Open **Application CMS** in the sidebar. You can use it without an OpenAI key or Gmail authorization.

1. Add an official source snapshot with its URL, excerpt, and verification state.
2. Add a programme or opportunity, then create an application and record its next action.
3. Add sourced application, funding, document, and referee deadlines. Add requirements with `FORMAL_APPLICATION`, `FACULTY_OUTREACH`, or `REPLY_REQUEST` context. Keep unresolved items `UNKNOWN`.
4. Upload original files in **Document Vault**. Matching SHA-256 bytes show the existing version; source bytes are never overwritten. Review and approve a version, then link it to a matching requirement from the application detail screen. The same version can be linked to multiple applications.
5. Add tasks and referees. The **Today** screen shows deadlines, missing required items, unknown requirements, overdue tasks, and document approval or expiry alerts.

Administrative readiness is shown as separate counts: required items completed, unknown requirements, and conditional requirements. It is a document/process checklist, not an admission assessment. `PASSPORT` and `GOVERNMENT_ID` default to `HIGHLY_SENSITIVE` and are local-only. No cloud backend is active.

## Applicant truth and discovery workflow

The **Applicant Truth** tab imports approved local CV/PDF/DOCX source versions, reads every PDF page or DOCX paragraph/table, and optionally calls OpenAI typed structured output to create *pending* claim candidates. Model extraction requires `OPENAI_API_KEY`; manual claims work without it. The existing local CV was imported into the Vault as a **pending** source, and its manually seeded claim candidates remain pending. Review the CV source version, inspect each claim and its page/location, correct wording and classification, then approve it for application and outreach independently. FACT and INFERENCE claims need linked evidence; facts need a verified review state. Approved profile snapshots preserve exact claim revisions and cannot be edited.

The **Research Directions** tab holds draft or approved versions of proposed PhD directions. An approved version is immutable; editing makes a new draft version. Attach approved claims where a direction relies on past experience. Draft directions do not prove applicant expertise.

The **Discovery** tab manages target institution states, official source URLs, evidence snapshots, opportunities, professors, and OpenAlex publications. Historical targets start as `CONSIDERING`, and all 156 historical professor rows were copied into separate `NEEDS_REVERIFICATION` faculty profiles. An old LLM score never upgrades a verification state. Static source pages are fetched and hashed when possible; manual snapshots retain a reviewed excerpt when a site blocks simple HTTP fetching. Source evidence and verification events are append-only. Evidence freshness is a review interval: 45 days for faculty affiliation/email, 3 for openings, 7 for deadlines, 14 for requirements/contact policy, and 30 for publication context.

Search or enter an official programme, faculty, lab, or vacancy page; snapshot it; then review any candidate before adding a faculty profile or opportunity. `OPEN`, `CLOSED`, and `UNKNOWN` opportunity states are separate from publication activity. OpenAlex author search is available after institutional identity review; an operator must choose the author after comparing name, affiliation, subject area, and works. Independent publication pages can be linked when OpenAlex affiliation metadata is noisy. A paper never establishes a current opening. Discoveries can link to the application ledger, and conflicting programme, opportunity, deadline, or requirement data creates a review task instead of overwriting entered values. Research Fit and Application Readiness remain separate nullable fields with explicit unknowns.

The ten-record pilot is reproducible with `python -m scripts.slice2_pilot --apply` after reviewing the linked official pages. The command is idempotent for records already reviewed. It creates draft research directions and one Stanford programme application opportunity, but no application, email, or submission. The remaining historical professor records require controlled review. This slice does not generate tailored CVs, SOPs, proposals, emails, or portal submissions.

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
