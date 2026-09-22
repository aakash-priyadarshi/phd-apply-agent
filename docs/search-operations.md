# Search operations and record controls

This release adds a persistent search workspace over the existing applicant context and application ledger. It does not change applicant trust, preflight, outreach approval, or the Document Vault.

## Applicant flow

1. In **Find programmes** or **Searches**, name a search, describe the research target, and optionally choose countries, programme types, funding, QS preference, research-fit floor, and official-page budget.
2. A background operation checks saved programme records, then proposes official URLs through the configured OpenAI provider. With no API key, the saved-record pass still works. Country and programme-type choices are included in the provider query and applied as explicit result filters. Unknown country or funding is not presented as a match to a hard filter.
3. **Operations** and the search page show the current stage, source page, count, result count, and event history. The cards and results refresh in the browser. **Stop** is cooperative: the current official page may finish before the worker stops. Completed records are retained. **Resume** reuses the saved URL list and skips completed pages; **Retry** starts its discovery pass again. An interrupted run after a service restart is marked `INTERRUPTED` and may be resumed. Up to two operations can be active at once.
4. Search results can be grouped by country or university and paged. Several unaccepted results can be removed at once and later restored. Accepted results are managed through their application.
5. **Applications** exposes corrections to cycle, status, portal, funding/eligibility/contact status, next action, notes, programme name, institution, country, department, degree, URLs, and sourced deadlines. Official-source confirmation is explicit; unknown details stay unknown. Archive shows a linked-record impact count and preserves historical links and files for restoration. **Universities** groups active applications by country or institution.

## Persistence and boundaries

Migration 10 adds `search_sessions`, `operations`, append-only `operation_events`, `record_change_events`, `record_field_reviews`, and reversible `archived_at` fields on applications and programme candidates. Existing records are unchanged, active by default, and migrations are additive. The detached worker uses the existing SQLite path under `PHD_AGENT_DATA_DIR`, including `/data` on Railway. It never sends email. Status changes and checkpoints are SQLite transactions; source analysis itself remains an existing orchestrator step. Neither stopping nor archiving deletes historical evidence or packages.

The programme and faculty searches are the first supported operation types. Other long-running document, refresh, browser, and outreach flows remain synchronous and are future work. This slice does not implement comparison, watchlists, calendar, dynamic QS lookup, missing-detail refresh, or portal handoff from the broader roadmap. Existing advanced review screens remain available where detailed faculty verification is necessary.
