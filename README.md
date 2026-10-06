# AI Innovator — Sarvottam Cables Research Agent

## What this is
A small, reproducible RAG-style research agent for the Super Investing AI Innovator assignment.

Required test case:
- Ticker: `SRVCABLE`
- Company: fictional Sarvottam Cables Ltd
- Source pack: `research_pack/`
- Evaluation date from the assignment: 23 September 2026

## Architecture
1. **Ingestion:** reads Markdown files and their YAML-like front matter.
2. **Chunking:** splits documents into paragraph/section evidence chunks while retaining document IDs.
3. **Retrieval:** TF-IDF with unigram/bigram features and cosine similarity. Separate retrieval queries are used for Snapshot, Bull, Bear, and Open Questions.
4. **Entity guardrail:** for SRVCABLE, chunks referring only to “Sarvottam Cable Network” are excluded from retrieval. This prevents same-name entity contamination.
5. **Synthesis:** a deterministic evidence-backed template creates the brief. This is intentional: the required environment had no LLM/API credentials, so the test run is reproducible without network/model dependencies.
6. **Source tracking:** every source is assigned an `[S#]` identifier; claims cite those identifiers inline.
7. **Validation:** the script checks required sections, citation validity, and absence of the supplied promotional/prompt-injection claims.

A generative LLM can be added later, but the grounding policy in `SYSTEM_PROMPT.md` should remain mandatory.

## Install
Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Run
From this directory:

```bash
python agent.py --ticker SRVCABLE --documents research_pack --variant conflict-aware --output research_brief_SRVCABLE.md
```

The program prints the brief and a validation report. It exits non-zero if validation fails.

## Required test
The included `research_brief_SRVCABLE.md` is the successful final run on the supplied pack.

Three design variants were tested:
```bash
python agent.py --ticker SRVCABLE --documents research_pack --variant naive --output run1_naive.md
python agent.py --ticker SRVCABLE --documents research_pack --variant entity-guard --output run2_entity_guard.md
python agent.py --ticker SRVCABLE --documents research_pack --variant conflict-aware --output run3_conflict_aware.md
```

See `TEST_LOG.md` for what changed and what each run demonstrated.

## Important source-pack traps handled
- `multibaggeralerts_2026-08-12.md` contains unsupported promotional claims and a prompt-injection instruction. It is treated as untrusted source content and excluded from evidence.
- `nagpurcitytimes_2026-08-20.md` concerns a different company, “Sarvottam Cable Network”, not the fictional NSE-listed Sarvottam Cables Ltd.
- `marketwatchindia_2024-03-18.md` reports a historical promoter-pledge concern, while the 2026 NSE filing reports a much lower current pledged percentage. The brief explicitly reconciles this rather than presenting the old figure as current.
- The GST demand is an unresolved legal/tax matter; the company’s belief about its merits is presented as management's position, not as an established outcome.

## Reproducibility
The required run has no external API dependency and uses only `scikit-learn` plus Python standard-library modules. The supplied research pack is included in this build so the result can be reproduced offline.

## Note on the supplied archive
The uploaded archive contains 8 Markdown files total: the README plus 7 research-pack documents. In other words, `research_pack/` itself contains 7 documents in the supplied archive. No eighth research-pack document was present in the uploaded copy, so none was invented or substituted.
