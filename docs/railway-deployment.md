# Railway deployment

This is a single-user Streamlit CMS. The first hosted deployment uses one Streamlit service and one Railway volume mounted at `/data`. SQLite and the Document Vault stay on that volume. Do not add Postgres, replicas, or a Railway Storage Bucket for launch.

`S3CompatibleDocumentStorage` / Railway Bucket remains a post-deployment enhancement. Document identities stay on the existing `DocumentStorage` interface and `LocalDocumentStorage` for this release.

## Start command

Railway should start the service with:

```sh
python -m phd_agent.launch
```

The wrapper runs `prepare_runtime()` first so `.streamlit/secrets.toml` and the operator allowlist exist before Streamlit loads OIDC. It then execs the equivalent of:

```sh
streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

Do not hard-code a port. Railway injects `PORT`. The wrapper is the `startCommand` in `railway.toml`.

Startup also imports and validates the reviewed authentication runtime before touching the database. Production requirements install `streamlit[auth]==1.64.0` and `Authlib==1.8.0`; a missing or different authentication runtime stops the process with a safe configuration error instead of exposing a broken login button.

Health checks use Streamlit’s built-in route `/_stcore/health`, also set in `railway.toml`. Railway only probes that path at deploy time. A volume-backed service still has brief downtime on redeploy because two deployments cannot mount the same volume.

## 1. Merge reviewed code to main

Merge the reviewed deployment-readiness branch into `main` after GitHub checks pass. Deploy from `main`, not from a laptop copy of `data/`.

## 2. Create the Railway project and service from GitHub

1. Create a Railway project.
2. Add one service from this GitHub repository, production branch `main`.
3. Leave it as a single replica. Volumes cannot be used with replicas.

## 3. Attach a volume at `/data`

Add one volume to the Streamlit service with mount path `/data`. Volumes are available at runtime, not during image build. Do not create `phd_outreach.db` in a Dockerfile `RUN` step.

## 4. Set `PHD_AGENT_DATA_DIR=/data`

The hosted app refuses to start unless this value is an existing absolute directory.

## 5. Configure the start command

Confirm the start command from the section above. `railway.toml` supplies it; the dashboard custom start command should match if you override config-as-code.

## 6. Configure application variables

Set at least:

| Variable | Production value |
| --- | --- |
| `PHD_AGENT_ENV` | `production` |
| `PHD_AGENT_DATA_DIR` | `/data` |
| `PHD_AGENT_ALLOWED_EMAILS` | Your Google account, comma-separated if needed |
| `AUTO_SEND_ENABLED` | `false` (ignored even if set true) |

Do **not** set `PHD_AGENT_AUTH_DISABLED=true` on Railway. Production and any Railway environment ignore that bypass.

Optional later: `OPENAI_API_KEY`, `OPENALEX_API_KEY`, `USER_NAME`, `USER_EMAIL`. The CMS starts in manual mode without OpenAI or Gmail.

Never put `credentials.json`, `gmail_token.json`, OIDC secrets, or applicant files in Git.

## 7. Configure OIDC authentication

Create a Google Cloud **Web** OAuth client (this is separate from the Desktop client used for Gmail). Authorized redirect URI:

`https://<your-public-domain>/oauth2callback`

Set:

| Variable | Purpose |
| --- | --- |
| `PHD_AGENT_OIDC_CLIENT_ID` | Web client ID |
| `PHD_AGENT_OIDC_CLIENT_SECRET` | Web client secret |
| `PHD_AGENT_OIDC_COOKIE_SECRET` | Long random string; keep it stable across deploys |
| `PHD_AGENT_OIDC_REDIRECT_URI` | The `https://…/oauth2callback` URL |
| `PHD_AGENT_OIDC_SERVER_METADATA_URL` | Defaults to Google’s OpenID configuration |

The redirect URI must be an absolute HTTPS URL ending exactly in `/oauth2callback`. The metadata URL is fixed to `https://accounts.google.com/.well-known/openid-configuration`; a different override is rejected. At startup the app writes `.streamlit/secrets.toml` with restricted file permissions from these variables. It does not expose ID or access tokens. That file is gitignored. Unauthenticated visitors and accounts outside `PHD_AGENT_ALLOWED_EMAILS` never see profile, documents, professor data, Gmail metadata, applications, backups, or archives.

Streamlit CORS and XSRF protection remain enabled. The repository allowlists `https://phd-agent-production.up.railway.app` as the production origin and `phd-agent-production.up.railway.app` as the production WebSocket host. Localhost remains accepted by Streamlit 1.64 for local development. If the public domain changes, update `.streamlit/config.toml` and the Google redirect URI together before deploying.

Give the service a public HTTPS domain. Do not expose it without this login gate.

## 8. Deploy without Gmail first

The first deploy should omit Gmail files and Gmail env vars. Confirm the app starts, login works, and Application CMS loads an empty or restored ledger.

## 9. Verify migrations and data persistence

After login:

- `phd_outreach.db` exists under `/data`
- `schema_migrations` includes versions 1–7
- `documents/` exists on the volume
- creating a source or Vault upload writes under `/data/documents`

## 10. Restart or redeploy and verify data survives

Redeploy the service. Login again and confirm the same database rows and Vault files are present. A volume-backed redeploy has short downtime; that is expected.

## 11. Enable scheduled volume backups

In the service **Backups** tab, enable a Railway volume schedule (daily is the default recommendation; weekly/monthly can be added). This is the disaster-recovery copy. The in-app Backup tab still copies the ledger and Vault into a directory, defaults to excluding credentials, and is not enough if it only writes onto the same volume.

Do not wipe the volume: wiping deletes Railway backups.

## 12. Rotate and configure Gmail OAuth separately

Keep using a **rotated Desktop** OAuth client. Do not start `InstalledAppFlow` on Railway.

Choose one:

1. Copy rotated `credentials.json` and `gmail_token.json` onto the volume with `railway ssh` / `railway volume files` after creating the token on a trusted local machine with `python -m scripts.authorize_gmail`.
2. Or set `PHD_AGENT_GMAIL_CREDENTIALS_JSON` and `PHD_AGENT_GMAIL_TOKEN_JSON` once. Startup writes them with `write_restricted_file()` only if the files are not already on the volume. Existing volume files win.

Never print these values in logs or the UI. Do not run demo scripts (`scripts/slice*_demo.py`, `scripts/slice2_pilot.py`) against production.

## 13. Import Sent history before any email

In **Gmail Sent memory**, import the complete Sent metadata and review matches before any manual send. Rotated credentials plus Sent reconciliation are still required.

## 14. Keep auto-send disabled

`auto_send_enabled` is always false. There is no automatic sending worker.

## Rollback and restore

**Railway volume backup:** open the service Backups tab, restore the chosen snapshot, review the staged volume swap, then deploy. The previous volume stays in the project unmounted. Confirm `/data` still contains `phd_outreach.db` and `documents/` after the service comes back.

**Application backup:** restore a verified archive into a separate directory first. Replacing `/data` requires stopping the service and passing `replace_existing` only after you have a Railway volume backup. Credential files stay excluded unless you explicitly included them when the archive was made.

If authentication is misconfigured, the process refuses to render applicant data. Fix OIDC variables and redeploy; do not enable `PHD_AGENT_AUTH_DISABLED` on Railway.

## Local development

Copy `.env.example` to `.env` and keep `PHD_AGENT_ENV=development` plus `PHD_AGENT_AUTH_DISABLED=true`. That bypass is explicit and does not apply when `PHD_AGENT_ENV=production` or Railway environment variables are present.
