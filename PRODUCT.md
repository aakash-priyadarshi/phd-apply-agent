# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The primary user is an individual PhD applicant managing a high stakes, multi university application cycle. The operator needs to discover suitable programmes and supervisors, prepare grounded outreach and documents, complete portal applications, and preserve an auditable record without repeatedly entering the same facts.

## Product Purpose

The PhD Application Agent reduces applicant workload while preserving evidence, correctness, and human control. A normal workflow starts with an intent, programme URL, supplied page, or review decision. The system then populates the underlying CMS and asks the operator to handle only uncertain, missing, sensitive, or approval required work.

Success means a programme can move from intent or URL through evidence backed review and shortlisting; outreach can move from professor selection through a CV grounded draft and review; and applications can reuse approved facts, context, and documents without repeated entry.

## Positioning

The product combines an evidence and provenance ledger with a reusable, versioned ApplicantResearchContext. It connects the applicant's demonstrated experience and proposed research direction to programme discovery, supervisor fit, outreach, generated documents, and application answers while keeping research fit separate from readiness and admission probability.

## Operating Context

The canonical workflow is `INTENT → DISCOVER → SHORTLIST → PROFESSORS → CONTACT → REPLY → DOCUMENTS → APPLICATION → SUBMISSION`. Professor contact remains conditional on verified policy. The Railway hosted Streamlit CMS is the orchestration and data system; a future local Playwright worker may assist supervised portal sessions that involve local cookies, MFA, CAPTCHA, file choosers, or sensitive values.

The application uses SQLite and a local Document Vault, with production state under `/data`. It retains historical outreach data and compatibility code. Gmail and OpenAI are optional at startup, and no workflow may automatically submit a university application or reply to a professor.

## Capabilities and Constraints

- Preserve the application ledger, Document Vault, versioned applicant truth, requirements, research matching, outreach safety, preflight, submission archive, and append only audit history.
- Treat the approved Master CV, approved profile snapshot and claims, approved research direction, and verified research records as the trusted applicant context. Unapproved CV statements remain review candidates.
- Retrieve task relevant context instead of sending the full CV to every model call, and record the applicant and opportunity evidence that influenced each result.
- Keep demonstrated experience, transferable experience, proposed direction, unknowns, Research Fit, and Application Readiness distinct.
- Preserve immutable approved artifacts, frozen packages, and submission archives. Context changes mark downstream work for review and produce new versions on request.
- Prefer official university evidence. Unknown facts remain `UNKNOWN`; publication activity alone never proves an opening or supervision availability.
- Use deterministic logic for permissions, authentication, hashes, requirement states, package selection, duplicate and send safety, submissions, and preflight.
- Passwords, MFA codes, payment values, and highly sensitive identity values stay out of language model inputs. Final submit, legal declarations, and payments require explicit human action.
- Keep model identifiers and escalation policy outside business logic and record generation provenance.

## Evidence on Hand

The repository contains the approved Slice 0–5 implementation, additive SQLite migrations, source backed discovery, 156 historical professor rows, applicant claim review, research directions, generated materials, packages and preflight, reviewed outreach, reply classification, portal answer assistance, submission archives, backup and restore, deployment tests, and the implementation brief supplied for this change.

## Product Principles

1. Ask for intent and decisions; derive database records from evidence.
2. Reuse approved applicant truth and documents before asking for input.
3. Show provenance, uncertainty, and the next useful action at the point of decision.
4. Preserve human approval at consequential boundaries.
5. Measure success by manual work avoided without weakening correctness.

## Accessibility & Inclusion

The operator interface must remain keyboard usable, readable without color alone, explicit about states and blocking reasons, and functional at common laptop and narrow browser widths.
