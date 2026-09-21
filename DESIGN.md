# Interface design

The application uses a **calm personal workspace** visual system. It should feel like a focused assistant for one applicant, with evidence and records handled behind the interface.

## Mode

Operate. The applicant should always see the next useful action, the evidence behind it, and the decision they control.

## Composition

- A narrow navigation rail names five plain-language areas: Home, Find programmes, Applications, People, and My documents.
- The main canvas begins with one action or decision, followed by current state and evidence.
- Candidate programmes, professors, applications, and fill plans use bordered records with stable metadata positions.
- Setup is one guided document action. Existing Vault files are detected and can finish setup without re-uploading.
- Detailed tables, approvals, IDs, migration-era forms, and audit correction tools live under Advanced.
- Status summaries use counts and plain words. Percentages never replace unknown or conditional counts.

## Visual system

- Warm paper background `#F7F6F2`, white working surfaces, deep ink `#17202A`.
- Cobalt `#2457D6` marks primary actions and selected navigation.
- Forest `#147A5B`, amber `#A86513`, and red `#B33A3A` communicate verified, review, and blocked states with text labels as well as color.
- System sans serif carries interface copy. Monospace is reserved for IDs, hashes, source state, and provenance.
- Corners are restrained. Cards use a one pixel rule and a small shadow; primary controls remain rectangular and easy to scan.
- No gradients, decorative dashboards, floating glass surfaces, or animation that competes with review work.

## Interaction

- Intent is the primary first action after the applicant profile exists.
- Document uploads immediately create a readable Markdown profile summary and the internal reviewed context required by matching and generation.
- URL ingestion escalates visibly from static retrieval to local browser assistance to paste/upload.
- Candidate acceptance is one review form and one action.
- Consequential actions name what will be created; sending and submission remain separate explicit actions.
- Unknown, uncertain, stale, and blocked states remain visible until reviewed.
- Legacy outreach is lazy loaded only after a separate explicit open action, so entering Advanced cannot trigger a document chooser.

## Adaptation

- Metric groups wrap instead of shrinking unreadably.
- Cards become one column on narrow screens.
- Data tables remain horizontally scrollable and are secondary to summary cards.
- Reduced motion is the default; no workflow depends on hover.
