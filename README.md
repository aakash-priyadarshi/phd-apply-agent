# PhD application and outreach console

This is a Streamlit PhD application agent with an application ledger, Document Vault, versioned applicant truth, CV-grounded research context, source-backed discovery, reviewed faculty outreach, portal assistance, and immutable submission records. The normal workspace is intent first. Detailed database forms are under **Advanced / Legacy → Data & audit**; the 2025 outreach interface is retained only under **Advanced / Legacy → Legacy outreach**, with its sending controls disabled. The [2026–27 implementation plan](docs/implementation-plan-2026-27.md) and [intent-first architecture](docs/intent-first-orchestration.md) describe the system.

## Run locally

Use Python 3.11 or newer. On Windows, run `setup.bat`, copy `.env.example` to `.env`, fill in your values, then run `start_phd_outreach.bat`.

On macOS or Linux:

```sh
sh setup.sh
cp .env.example .env
# Edit .env locally.
.venv/bin/python -m streamlit run streamlit_app.py
```

Local unauthenticated use requires `PHD_AGENT_AUTH_DISABLED=true` in `.env`. That bypass is ignored when `PHD_AGENT_ENV=production` or the process is running on Railway.

Hosted start command (Railway injects `PORT`; do not hard-code it):

```sh
python -m phd_agent.launch
```

That wrapper writes OIDC secrets and validates the allowlist, then execs:

```sh
streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

Full Railway steps, OIDC, volume backups, Gmail bootstrap, and rollback are in [docs/railway-deployment.md](docs/railway-deployment.md). Production uses `PHD_AGENT_DATA_DIR=/data` on one persistent volume. The CMS starts without OpenAI or Gmail. Auto-send stays disabled.

For tests, install `requirements-dev.txt` in the same environment, run `python -m phd_agent.auth_dependencies`, `python -m pip check`, and `python -m pytest -q`. Normal tests do not send email. The two Gmail integration tests require explicit environment flags and a test recipient.

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

The application ledger uses additive SQLite migrations recorded in `schema_migrations`. Slice 2 adds profile, claim, research track, catalogue, faculty verification, publication, and assessment tables without changing legacy professor rows. The approved Slice 1 baseline is backed up under `data/backups/phd_outreach-slice1-eefd5b8c.db`. Stop the app before any restore and keep a separate copy of the current database first; restoring the baseline discards later local work. Vault files under `data/documents/` need their own backup alongside the database. On Railway, put SQLite and the Vault on the `/data` volume and enable Railway volume backups; do not treat in-app backups on the same volume as disaster recovery.

## Intent-first application workflow

The application now opens directly into a simple personal workspace. You can build the applicant profile, reuse the Vault, analyse URLs, manage applications and review matching without learning the underlying database screens. OpenAI improves structured extraction and current web discovery when configured; local parsing and the core workspace work without it.

1. Add a CV and supporting documents under **Build my profile**. One action stores the originals, extracts source-backed facts, creates the internal profile/Master CV/research context, and writes a readable `data/profile_summaries/applicant-profile-vN.md` file. Previously uploaded Vault documents are detected and reused.
2. Enter a research intent or analyse one official programme URL. Static retrieval is attempted first; blocked or weak pages request pasted text or saved HTML/PDF.
3. Review extracted fields, evidence, unknowns, CV-grounded fit, and shortlist/reject/accept the candidate. Accepting once creates the programme, opportunity, application, source snapshot, deadline, and document requirements that the evidence supports.
4. Use **Find Relevant Supervisors** for a shortlisted application. Cards keep demonstrated applicant experience, proposed direction, Research Fit, contact policy, evidence, and unknowns distinct.
5. Reuse approved common documents and Application Profile values. **Prepare Application** reports missing, unknown, package, and preflight blockers without hiding them in a percentage.
6. Analyse saved portal HTML in **Browser Assistant**, approve the fill plan, and run the optional local Playwright companion. It fills approved safe fields only and stops before submission.

The detailed ledger, source, claim, matching, material, package, outreach, reply, archive, and backup screens remain available after enabling **Advanced tools → Data & audit**. The retired 2025 workspace is not loaded until **Open legacy workspace** is pressed.

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

## Follow-through (Slice 5)

**Application CMS → Follow-through** classifies imported reply metadata, creates the next document/meeting/portal task, and can draft a reviewed follow-up. Classification never sends a reply. A proposal request can generate a draft variant from the approved research direction; that draft still needs review before it can enter a package.

Portal answers are an approved copy-ready library with visible sources. Map them onto a per-application field checklist, review each value, then paste into the institution form yourself. After you submit and pay on the university site, freeze a **submission archive** with the confirmation number, user-recorded payment state, frozen package hashes, and the reviewed answers. The app does not submit the portal form and does not process payment.

**Backup** copies the SQLite ledger and Document Vault into a new directory and verifies hashes on restore. `credentials.json` and `gmail_token.json` are excluded unless you explicitly include them. Restore into a separate directory first; replacing an existing database requires an explicit flag. Stop the app before restoring over the active database.

The isolated demonstration runs with `.\.venv\Scripts\python.exe -m scripts.slice5_demo --output-dir data/slice5-demo-fresh` (choose a fresh ignored directory). It does not contact Gmail or a university portal.

OAuth client rotation remains mandatory before any real Gmail use.

## Local browser companion

Portal browser sessions remain on the applicant's computer. Playwright is deliberately not installed in the Railway production requirements. In the local virtual environment, install it when needed:

```powershell
.\.venv\Scripts\python.exe -m pip install playwright
.\.venv\Scripts\python.exe -m playwright install chromium
```

Create and approve a fill plan in **Browser Assistant**, then run the command shown by the app, for example:

```powershell
.\.venv\Scripts\python.exe -m scripts.browser_companion 12
```

The companion uses a persistent browser profile under the ignored data directory, pauses for access checks, fills only `SAFE_AUTOFILL` fields, and closes without submitting. Passwords, MFA, payment, identity values, legal declarations, file chooser actions, and final submission remain manual.

## Model routing

`phd_agent.model_router` keeps GPT-5.6 family names and escalation policy outside business logic. Luna is the default for low risk extraction and classification, Terra for programme/faculty reasoning and first drafts, and Sol for high stakes research fit and final document work. Low confidence, conflicting evidence, unsupported output, or an operator request can escalate a task; an override cannot lower its reviewed risk tier. Deterministic safety gates do not use a model.

The 2025 discovery map remains historical data, and existing professor records still require current evidence. Do not treat an old score or publication activity as proof of a current opening.
