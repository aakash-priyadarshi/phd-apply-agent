# PhD application and outreach console

This is a local Streamlit application with an application ledger, Document Vault, versioned applicant truth library, research directions, source-backed discovery, and a reviewed faculty outreach queue. The earlier CV analysis, professor discovery, and email drafting/editing remain available under **Legacy outreach**; its sending controls are disabled. The [2026–27 implementation plan](docs/implementation-plan-2026-27.md) describes the staged work.

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

Academic certificates and records marked `NEEDS_REVIEW` require a separate document type/authenticity review before approval. The original University of Liverpool MSc award certificate and Higher Education Achievement Report are in the active local Vault as confidential source versions. The HEAR contains transcript-like module results; whether a target institution accepts it as an unofficial transcript must be checked against that institution's rules. Neither is automatically linked to an application requirement.

## Applicant truth and discovery workflow

The **Applicant Truth** tab imports approved local CV/PDF/DOCX source versions, reads every PDF page or DOCX paragraph/table, and optionally calls OpenAI typed structured output to create *pending* claim candidates. Model extraction requires `OPENAI_API_KEY`; manual claims work without it. The existing local CV was imported into the Vault as a **pending** source, and its manually seeded claim candidates remain pending. Review the CV source version, inspect each claim and its page/location, correct wording and classification, then approve it for application and outreach independently. FACT and INFERENCE claims need linked evidence; facts need a verified review state. Approved profile snapshots preserve exact claim revisions and cannot be edited.

The **Research Directions** tab holds draft or approved versions of proposed PhD directions. An approved version is immutable; editing makes a new draft version. Attach approved claims where a direction relies on past experience. Draft directions do not prove applicant expertise.

The **Discovery** tab manages target institution states, official source URLs, evidence snapshots, opportunities, professors, and OpenAlex publications. Historical targets start as `CONSIDERING`, and all 156 historical professor rows were copied into separate `NEEDS_REVERIFICATION` faculty profiles. An old LLM score never upgrades a verification state. Static source pages are fetched and hashed when possible; manual snapshots retain a reviewed excerpt when a site blocks simple HTTP fetching. Source evidence and verification events are append-only. Evidence freshness is a review interval: 45 days for faculty affiliation/email, 3 for openings, 7 for deadlines, 14 for requirements/contact policy, and 30 for publication context.

Search or enter an official programme, faculty, lab, or vacancy page; snapshot it; then review any candidate before adding a faculty profile or opportunity. `OPEN`, `CLOSED`, and `UNKNOWN` opportunity states are separate from publication activity. OpenAlex author search is available after institutional identity review; an operator must choose the author after comparing name, affiliation, subject area, and works. Independent publication pages can be linked when OpenAlex affiliation metadata is noisy. A paper never establishes a current opening. Discoveries can link to the application ledger, and conflicting programme, opportunity, deadline, or requirement data creates a review task instead of overwriting entered values. Research Fit and Application Readiness remain separate nullable fields with explicit unknowns.

The ten-record pilot is reproducible with `python -m scripts.slice2_pilot --apply` after reviewing the linked official pages. The command is idempotent for records already reviewed. It creates draft research directions and one Stanford programme application opportunity, but no application, email, or submission. The remaining historical professor records require controlled review.

## Reviewed matching, materials, and packages

Slice 3 adds **Match Review**, **Materials**, and **Packages / Preflight** to Application CMS. Start in **Applicant Truth**: review source evidence for each claim, choose its application and outreach permissions, and create an approved profile snapshot. In **Research Directions**, review supporting claim IDs and approve a track version. The existing nine pending claims and three draft directions are not automatically approved. Returning an approved direction to draft creates a new version.

Match Review computes a 0–10 Research Fit from five configurable, evidence-backed components and displays evidence coverage and unknown components. Application Readiness is a separate set of factual states, never an admission probability. Overrides and annotations create append-only review versions. A publication is research evidence, not evidence of an opening. Before faculty contact, record an explicit official policy such as `ALLOWED`, `CONTACT_ALLOWED`, or `DO_NOT_CONTACT`; unrecognized free text stays unknown.

Materials supports approved, versioned Master CV sections and story modules. Each reusable bullet/module names approved claim revision IDs. Tailored CVs select or reorder reviewed sections and show the master-to-variant diff. Statements use approved story modules plus an exact sourced requirement. Proposals preserve an approved research direction and cite only stored, verified publications. A cover letter needs a required/optional sourced requirement or an intentional manual selection. Drafts keep editable text, PDF, generation context, claim/evidence IDs, model/template identifiers, and quality checks. Review each PDF and its evidence before approval; a new edit creates a new version. The local renderer and manual workflows work without an OpenAI key. Short content is visibly warned and needs human expansion/review for a competitive application.

Packages use a deterministic rule for each requirement: `INCLUDE`, `EXCLUDE`, or `REVIEW`. `UNKNOWN` is always `REVIEW`. A required document must be linked to an approved Vault version before a package can be built. The builder freezes requirement/evidence snapshots, selected version IDs, SHA-256 hashes, decisions, and a manifest under `data/exports/<application>/package-vN/`. It can create a ZIP and an explicitly selected combined PDF while preserving the originals. The export is a convenience copy; the Vault remains canonical.

Run preflight before marking a package ready. It checks the context, current requirements and evidence, files and hashes, limits, generated material lineage, citations, and relevant application or faculty conditions. Every rule has `PASS`, `WARNING`, or `BLOCK`, an affected record, and an operator action. A `BLOCK` prevents `READY`; `WARNING` remains visible for reviewer judgment. Formal packages also require the application deadline, route, eligibility, and referees. Outreach packages require current faculty affiliation, research evidence, verified email, and an explicitly allowed contact policy. **READY means reviewed materials only**: Slice 3 sends no email, submits no portal form, and stores no cloud documents. The legacy single-send controls are disabled until the later outreach safety workflow exists.

The isolated demonstration can be reproduced from the approved local Slice 2 database with:

```powershell
.\.venv\Scripts\python.exe -m scripts.slice3_demo --source-db data/phd_outreach.db --output-dir data/slice3-demo-new --liverpool-results "C:\Users\aakas\OneDrive\Desktop\PHD documents\Aakash-Priyadarshi-liverpool-marksheet.pdf"
```

Choose a fresh output directory each run. The script copies the database and Vault into that ignored directory, then makes sandbox-only review decisions for the Stanford programme and Diyi Yang research scenario. It writes `demo-report.json` and a local package export. It never approves claims or adds applications to the active database. The Liverpool assessment-results PDF is marked `PROVISIONAL_TRANSCRIPT`; a missing Galgotias transcript is replaced only in the sandbox with a prominent `DEMO_ONLY` placeholder. Both force preflight `BLOCK` until genuine accepted transcripts are supplied. The demo also leaves eligibility, English applicability, referee submissions, and the fee unresolved. Review and replace these in the real CMS; do not submit the sandbox package.

The original root copies of the 2025 runtime files were removed from the working tree after verified local copies were made. Git still contains their earlier versions in history. This branch does **not** rewrite Git history.

## Gmail setup and credential rotation

The old OAuth client file was tracked in the public repository and has been preserved locally under `data/backups/`, not in the active credentials path. Create a new Desktop OAuth client in your Google Cloud project, disable or delete the old client, and place the new downloaded JSON at `data/credentials.json`. Review and revoke the old app grant in your Google account if it is no longer needed. Google describes [creating a Desktop client and using JSON tokens](https://developers.google.com/workspace/gmail/api/quickstart/python) and [credential and token handling](https://developers.google.com/identity/protocols/oauth2/policies).

The app never loads the old `gmail_token.pickle`. After placing the rotated client JSON, run `.\.venv\Scripts\python.exe -m scripts.authorize_gmail` on Windows, or `.venv/bin/python -m scripts.authorize_gmail` elsewhere, to open an explicit local browser consent flow and create `data/gmail_token.json`. Treat the JSON token as a credential. It is not encrypted by this local release, so keep the data directory accessible only to your user account and protect the disk. If an old pickle token exists, remove it after you have confirmed the new authorization works. The new gateway requests Gmail send and read access; it will not start authorization by itself.

## Reviewed outreach (Slice 4)

In **Application CMS → Outreach Review**, prepare an email only after approving a real applicant profile, an evidence-backed research direction, current professor identity/email/topic evidence, an explicitly allowed faculty contact route, and a ready `FACULTY_OUTREACH` document package. Sandbox approvals never populate the active applicant record. The draft uses a typed context and deterministic wording. Its snapshot freezes claim/evidence/paper IDs, exact email text, selected attachment versions and hashes, and quality-gate results. Editing creates another version; approval never sends immediately.

Review the professor, relevant publication, applicant claim, full email, exact files, and each quality rule. Import the **complete** Gmail Sent metadata history in **Gmail Sent memory** and confirm possible matches. You can record a known historical contact or a reviewed `DO_NOT_CONTACT`/`REJECTED` state. Only an approved current package with reconciled Sent history and rotated Gmail credentials can be manually sent. The send action reads the complete Sent history again, reserves a unique local outreach key, and attaches only the approved hashed versions. If the Gmail result is uncertain, the package becomes `AMBIGUOUS_SEND`; use **Reconcile uncertain send** and inspect Sent before taking any further action. The app never retries that attempt automatically. Sent and reply metadata are stored locally without message bodies.

Campaigns provide review-mode policies, pause/emergency controls, and saved dry runs that say `WOULD_GENERATE`, `WOULD_SEND`, or `BLOCKED` with reasons. `auto_send_enabled` defaults to false, and this slice has no automatic sending worker. Configure one target timezone per campaign; keep candidates in the same local zone when using send windows. Windows and daily caps apply to reviewed manual campaign sends. The isolated synthetic demonstration runs with `.\.venv\Scripts\python.exe -m scripts.slice4_demo --output-dir data/slice4-demo-fresh` (choose a fresh ignored directory). It does not contact Gmail or modify the active database.

Legacy bulk send, generate-and-send, and individual send controls remain disabled. No portal application is submitted by this slice.

## Current workflow

1. Upload or reuse a local CV, then generate the research profile.
2. Add target universities or reuse the copied target list.
3. Run Stage 1 discovery and inspect the results.
4. Generate and edit an email for one professor.
5. Review the draft locally; use only the approved Outreach Review package workflow for Gmail sending after credential rotation and Sent reconciliation.

The 2025 discovery map is limited, and existing professor records need reverification for the new cycle. Do not treat an old `verified` status as current recruiting evidence.
