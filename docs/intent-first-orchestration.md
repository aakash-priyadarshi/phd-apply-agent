# Intent-first orchestration

The normal product surface is organised around applicant actions rather than tables:

`INTENT → DISCOVER → SHORTLIST → PROFESSORS → CONTACT → REPLY → DOCUMENTS → APPLICATION → SUBMISSION`

The existing ledger, Vault, approved claims, research tracks, matching, outreach, packages, preflight, reply history, submission archives, and backups remain authoritative.

## ApplicantResearchContext

`ApplicantResearchContextService` creates immutable snapshots from one approved profile version, approved Master CV version, and active approved research-track version. Every CV bullet resolves to approved claim revisions. Approved non-CV claims and the proposed research direction remain separately classified.

Task retrieval stores:

- context snapshot and hash
- task and query
- selected CV sections and context items
- claim revision IDs
- document version and source evidence IDs
- provider, model, prompt/policy version, and creation time

Output links record which context and retrieval influenced a match, generated artifact, outreach package, programme candidate, application answer, or browser fill plan. A new applicant context adds a separate staleness record for linked prior outputs. It does not edit an approved artifact, package snapshot, email, or archive.

## Programme ingestion

One official URL follows this order:

1. bounded static HTTP or PDF extraction from a public host
2. optional local `BrowserWorker` acquisition when a worker is supplied
3. paste text or upload saved HTML/PDF

Automated and operator-supplied acquisitions are labelled separately. Extracted candidates preserve field evidence, confidence, unknown fields, the applicant-context retrieval, and review state. Acceptance creates verified ledger records in one operator action. Unsupported fields remain unknown.

Intent-only discovery searches the reviewed programme and opportunity catalogue already in the ledger. When `OPENAI_API_KEY` is configured, the official-source provider routes discovery planning through Terra with OpenAI web search. Its output can only propose HTTPS official URLs; aggregator and social hosts are rejected, and each proposed page must still pass bounded acquisition, evidence capture, extraction, and operator review. OpenAI remains optional at startup.

Supervisor discovery uses the selected programme plus task-specific ApplicantResearchContext retrieval. Verified local faculty are ranked into evidence-backed cards. Optional web discovery acquires proposed official faculty/lab pages and queues unverified candidates; it never promotes a model result directly to a verified professor, current affiliation, email, supervision opening, or contact permission.

## BrowserWorker boundary

Railway stores orchestration state and fill plans. The local Playwright companion owns portal cookies and supervised browser sessions. The deterministic form planner classifies approved reusable values as safe autofill, narrative fields as generate-and-review, and credentials, MFA, payments, passport/identity values, and unmatched controls as manual.

An approved plan can fill safe fields locally. No code path clicks submit, accepts a legal declaration, completes a payment, supplies MFA, or sends sensitive values to a model.

## Workload measurements

Append-only workload events currently cover pages ingested, fields extracted, Application Profile values reused, form fields mapped, manual fields, and fields autofilled. These are operational counts, not quality claims.
