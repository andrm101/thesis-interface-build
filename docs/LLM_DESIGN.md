# Claude API features: design notes

There are two optional features. Both are built on the shared layer in `llm/`:

- **Policy-document extraction** (`policy_docs/`): runs once, and its outputs are frozen in the repository.
- **Research assistant** (`agent/`): runs live, in the dashboard's *Assistant* page and in desktop Tab 15.

Neither is needed by `reproduce.py`, the rest of either app, or CI.

## Shared layer: `llm/`

| File | Role |
|---|---|
| `client.py` | Imports the SDK lazily and reports whether the features can run, and why not (package missing, no key). Sets the defaults for every request: model `claude-opus-5` (override with `THESIS_LLM_MODEL`); server-side refusal fallback (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`), which retries a declined request on Anthropic's recommended fallback model; and `max_tokens` 16 000. It checks `stop_reason == "refusal"` before anyone reads the response content. |
| `grounding.py` | Extracts every number from an answer and checks it against the numbers in the sources. A number passes when some source value rounds to it at the precision it is written with; a share written as a percentage (43 % for 0.43) also passes. Years (1990–2035) and small counts (0–10) are skipped, because they are labels more often than estimates. |

Dependencies are in `requirements-llm.txt`: `anthropic` 1.x, `pypdf` (PDF text) and `rapidfuzz` (fuzzy quote matching).

## 1 · Policy-document extraction

The event study in D2 dates each R&D tax reform from a statistical rule: the first year the OECD implied subsidy rate rises by at least 5 pp and stays up. This feature checks those dates against what policy documents actually say.

```
data/policy_docs/<Country>/*.pdf|html|txt        (local, git-ignored)
  └─ extract.py   one request per document; the PDF is sent as a document
                  block, and the answer must fit schema.DocumentExtraction
     └─ verify.py  each event's verbatim quote is searched for in the
                   document's own text (pypdf); a fuzzy match of 90 or more on
                   the stated page ±1 marks it verified
        └─ data/policy_events/extracted/<Country>__<file>.json   (committed)
           └─ reconcile.py  → documented_events.csv, reconciliation.csv
              └─ used by: the "documented" event set (API, Tab 14), D2b
```

**Schema** (`policy_docs/schema.py`): each `ReformEvent` records:
- `year_effective` and `year_announced`;
- `instrument`: volume or incremental credit, enhanced deduction, payroll relief, patent box, accelerated depreciation, or other;
- `direction`: introduction, expansion, restriction or abolition;
- `firm_scope`;
- `rate_before` and `rate_after`, left empty when the document doesn't state them;
- `summary`, a verbatim `quote` and its `page`;
- `confidence`.

**Which events become dates.** A country's documented date is its earliest event that passes all four of these conditions:
- the quote was verified in the document;
- the direction is introduction or expansion;
- confidence is at least medium;
- the year falls in 2002–2021.

**Reconciliation statuses.** Each country's documented date is compared with the derived one:
- `confirmed`: the two dates are within one year;
- `date_shift`: both dates exist but differ by more;
- `doc_only`: only a documented date exists;
- `derived_only`: only a derived date exists.

A `reviewed` column is left for a human to sign off each row.

**Reproducibility.** Every extraction file stores the document's SHA-256, the model, and the prompt version. A rerun skips documents that haven't changed, and `--force` re-extracts them. Model output can vary from run to run, so the JSON files are the record of what was extracted, and they're committed with the code.

**Why the quote check happens here, not through the API's citations.** The API's citations feature can't be combined with structured output, so the quote is part of the schema and is checked locally. That check is deterministic, can be tested, and catches invented quotes.

## 2 · Research assistant

```
question ─► agent/runner.Assistant.ask
              loop: client.beta.messages.create(system, tools, messages)
                    stop_reason == "tool_use" → run each tool (read-only),
                    send every result back in one user message → repeat
                    (at most 8 tool calls; then tool_choice "none" forces an answer)
              answer ─► llm.grounding.check(answer, all tool results)
            ─► Reply(text, tool_calls [r1, r2 …], grounding)
```

**Tools** (`agent/tools.py`): each one wraps a function in `api/service.py`, the analysis layer the dashboard also uses:
- `list_variables`
- `describe_variable`
- `rank_countries`
- `local_projection`
- `event_study` (99 bootstrap draws)
- `convergence_clubs`
- `typology`
- `result_section`

The schemas are strict, and their enums list the actual variables, countries, event sets and `RESULTS.md` section IDs, so the model can't request something that doesn't exist. Every result is tagged with a `result_id`, which the answer must cite.

**Prompt** (`agent/prompts.py`): fixed text. It lays down the rules (numbers only from tool results, cite `[rN]`, careful causal language), then a data dictionary built once from the panel. Because it never changes, it is served from the prompt cache. The conversation history is also cached, and kept append-only so the cache stays valid; a turn that fails is rolled back.

**Interfaces.**
- **API:** `GET /api/assistant/status`, and `POST /api/assistant {question, conversation_id}`. The server keeps up to 50 conversations in memory, with a lock per conversation.
- **Web:** `web/src/app/pages/assistant.ts`. It renders the answer as Markdown with citation badges, shows each tool call and its arguments, and shows a grounding badge.
- **Desktop:** `tabs/tab_assistant.py`.

## Tests (no network, no key)

- `tests/test_policy_docs.py`:
  - a real PDF built with matplotlib;
  - quote verification, including an invented quote that must be rejected;
  - extraction through a fake client, checking the request shape, the PDF block and the fallbacks setting;
  - caching by file hash;
  - reconciliation statuses;
  - refusal handling and the "no key" path.
- `tests/test_agent.py`:
  - a scripted fake client driving the real tools;
  - result IDs and the single tool-result message;
  - tool errors returned to the model rather than raised;
  - the tool budget;
  - rollback of a failed turn;
  - the grounding rules;
  - the API's "no key" response (503).
- `tests/test_gui_smoke.py`: Tab 15 renders an answer and its grounding line.

## Cost (estimates; check on a first run)

- **Extraction:** a 15–30 page country profile is about 15–30k input tokens, so the 26 countries come to a few dollars once. Extractions are cached and never repeated for an unchanged file.
- **Assistant:** about $0.10–0.40 per question. Most of the system prompt and history is served from the cache after the first turn.
