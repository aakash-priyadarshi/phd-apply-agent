# Simple workspace redesign

## Goal

The default product should feel like an assistant for one applicant, not a database administration tool. The operator supplies documents, a research direction, programme intent or URL, and decisions. The application maintains the ledger and provenance underneath.

## Primary journey

1. **Build my profile** — add a CV and any useful supporting documents in one upload. Existing Vault documents are reused automatically.
2. **Review the readable profile** — the system extracts source-backed applicant facts and creates a versioned Markdown summary.
3. **Find programmes** — describe the target or analyse one official URL.
4. **Choose** — shortlist or add a programme from a compact card.
5. **Prepare** — reuse stored documents and applicant context; show only missing or uncertain items.

## Interaction rules

- No approval vocabulary in the normal workspace. A single deliberate “Build/Update my profile” action records the internal reviewed versions needed by the existing backend.
- Never send passwords, MFA, payment or identity-document values to a model.
- Show source names beside profile facts, but keep IDs, hashes and database forms in Advanced.
- Advanced and Legacy code remain available only after the operator explicitly opens them. Merely navigating to Advanced must not mount a document uploader or trigger a browser file dialog.
- The normal navigation is Home, Find programmes, Applications, People, and My documents.

## Document processing

- Accept PDF, DOCX, TXT and Markdown up to 25 MB per file, with a separate expanded-size guard for DOCX archives.
- Store source bytes once in the existing Vault.
- Extract readable text and source-backed facts. Use configured structured extraction when available and deterministic local parsing as a fallback.
- Create or refresh the approved profile snapshot, research direction, Master CV and ApplicantResearchContext in one operation.
- Write `data/profile_summaries/applicant-profile-pN-tN.md` and keep a generated copy in the Vault. Including both profile and research-direction versions prevents a focus-only update from replacing an earlier summary.
- Store and parse transcripts and degree certificates immediately, but do not use them as approved profile evidence until their authenticity check is complete.
- Adding documents creates a new context version only when applicant facts change; previous generated outputs remain immutable and receive the existing staleness marker.

## Delivery sequence

1. Profile document ingestion and Markdown summary service.
2. Setup recovery for previously uploaded CVs.
3. New task-based Streamlit workspace.
4. Lazy Advanced/Legacy entry.
5. Regression tests for profile activation, summary persistence and legacy isolation.
