# IG Calisthenics Corpus Builder

Build a research-ready corpus of public Instagram posts about calisthenics/bodyweight training using:
- **Apify** for post collection (hashtags + keyword-style discovery).
- **OpenAI Structured Outputs** for per-post eligibility decisions and narrative pre-tagging.
- A **feedback loop** that expands query terms from harvested hashtags and escalates when progress stagnates.
- A **resume-capable SQLite state store** to make long runs robust and auditable.
- Exports to a multi-sheet **Excel workbook** and a **PDF codebook**.

## Overview

This project collects candidate Instagram posts, then filters and labels them to produce:
- A final, deterministic sample of **N posts** (default 500).
- An eligible pool (default capped at 650) used for sampling and summary statistics.
- An audit trail of:
  - raw scraped items,
  - model decisions (including confidence and token counts when available),
  - actor run metadata,
  - reproducibility metadata (config hash, sampling seed, pool hash, selected keys).

### Outputs

Running the full pipeline writes the following into your chosen output directory:
- `state.sqlite` — resume-capable state database (raw items, decisions, runs, samples)
- `run.log` — JSONL log of major events and errors
- `corpus.xlsx` — workbook containing final sample, eligible pool, rejected posts, metadata, and tag summaries
- `codebook.pdf` — methods overview + operational rules + summary statistics + actor run listing

## Installation / Setup

### Requirements
- Python 3.10+ recommended
- An Apify API token (`APIFY_TOKEN`)
- An OpenAI API key (`OPENAI_API_KEY`)

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure environment variables

macOS/Linux:

```bash
export APIFY_TOKEN="your_apify_token"
export OPENAI_API_KEY="your_openai_key"
```

Windows (PowerShell):

```powershell
setx APIFY_TOKEN "your_apify_token"
setx OPENAI_API_KEY "your_openai_key"
```

## Usage

### 1) Dry-run (small end-to-end check)

Online dry-run (uses Apify + OpenAI):

```bash
python -m ig_corpus dry-run --config config.yaml
```

Offline dry-run (no network calls; uses a deterministic stub):

```bash
python -m ig_corpus dry-run --config config.yaml --offline
```

Notes:

* The CLI validates required environment variables even in offline mode; values can be dummy strings when running offline.
* Dry-run scrapes a tiny batch and processes up to 3 posts through the classifier, printing counts and one example decision.

### 2) Full run (feedback loop + exports)

```bash
python -m ig_corpus run --config config.yaml --out output/
```

This command runs until either:

* the eligible pool target is met and the final sample is recorded, or
* the run hits a configured cap (iterations or raw item limit), in which case exports are still generated and a failure report is printed.

### Resume behavior

Resume the most recent unfinished run in the output directory’s `state.sqlite`:

```bash
python -m ig_corpus run --config config.yaml --out output/ --resume
```

Run with an explicit run identifier (create or resume that run id):

```bash
python -m ig_corpus run --config config.yaml --out output/ --run-id my_run_001
```

## Configuration

Configuration is provided via YAML and validated into a strict schema.

### Minimal example

```yaml
targets:
  final_n: 500
  pool_n: 650
  sampling_seed: 1337

apify:
  token_env: APIFY_TOKEN
  primary_actor: apify/instagram-hashtag-scraper
  fallback_actor: apify/instagram-scraper
  results_type: posts
  results_limit_per_query: 150
  keyword_search: true
  run_batch_queries: 4

openai:
  api_key_env: OPENAI_API_KEY
  model_primary: gpt-5-nano
  model_escalation: gpt-5-mini
  escalation_confidence_threshold: 0.70
  max_output_tokens: 16000
  max_concurrent_requests: 4

filters:
  min_caption_chars: 40
  max_posts_per_user: 10
  allow_reels: true
  reject_if_sponsored_true: false

loop:
  max_iterations: 200
  stagnation_window: 10
  stagnation_min_new_eligible: 15
  max_raw_items: 20000
  backoff_seconds: 10

querying:
  seed_terms:
    - calisthenics
    - streetworkout
    - bodyweighttraining
    - bodyweightworkout
  expansion:
    enabled: true
    max_new_terms_per_iter: 15
    min_hashtag_freq_in_eligible: 4
    blocklist_terms:
      - bodybuilding
      - powerlifting
      - crossfit
      - yoga
      - bouldering
      - parkour
      - calisthenicscompetition
```

### Key sections

#### `targets`

* `final_n`: number of posts in the final exported sample
* `pool_n`: eligible pool size cap used for sampling and stats (must be ≥ `final_n`)
* `sampling_seed`: seed for deterministic sampling

#### `apify`

* `token_env`: env var name containing the Apify token
* `primary_actor`: actor id for broad collection (hashtag + keyword-style discovery)
* `fallback_actor`: actor id used for stagnation recovery (hashtag search + scraping discovered URLs)
* `results_limit_per_query`: requested results per term
* `keyword_search`: enable keyword-style discovery in the primary collector
* `run_batch_queries`: number of query terms per actor run

#### `openai`

* `api_key_env`: env var name containing the OpenAI key
* `model_primary`: main model for one-post labeling
* `model_escalation`: used when confidence is low or parsing fails
* `escalation_confidence_threshold`: escalation cutoff for `overall_confidence`
* `max_output_tokens`: cap for model thinking + output tokens
* `max_concurrent_requests`: maximum number of concurrent OpenAI labeling requests (must be ≥ 1)

#### `filters`

* `min_caption_chars`: fast precheck to avoid wasting model calls on too-short captions
* `max_posts_per_user`: dominance guard (0 disables)
* `allow_reels`: if false, non-feed content is rejected in prechecks
* `reject_if_sponsored_true`: if true, posts with `isSponsored=true` are rejected in prechecks

#### `loop`

* `max_iterations`: hard cap on feedback loop cycles
* `stagnation_window`: rolling window for stagnation detection
* `stagnation_min_new_eligible`: minimum new eligible posts required within the window
* `max_raw_items`: hard cap on total stored raw posts
* `backoff_seconds`: sleep between iterations

#### `querying`

* `seed_terms`: initial query terms used to bootstrap discovery
* `expansion`: controls hashtag harvesting and queue expansion

  * `enabled`: toggle expansion
  * `max_new_terms_per_iter`: enqueue cap per iteration
  * `min_hashtag_freq_in_eligible`: frequency threshold in eligible pool
  * `blocklist_terms`: terms never enqueued via expansion

## Methodology

This section describes the technical approach used to collect candidate posts, filter them deterministically, label them using Structured Outputs, and produce reproducible exports suitable for downstream analysis.

### System architecture

The pipeline is organized as an iterative feedback loop with four major subsystems:

1. **Acquisition** (Apify Actors): fetch public post metadata using controlled query terms.
2. **Normalization + deduplication**: map heterogeneous Actor fields into a stable internal post record and dedupe across runs.
3. **Filtering + labeling**:

   * fast, deterministic prechecks to avoid unnecessary model calls,
   * one-post-per-call LLM labeling with strict JSON schema enforcement,
   * deterministic eligibility enforcement derived from structured fields.
4. **Persistence + exports**:

   * resume-capable SQLite store for raw items, decisions, and run metadata,
   * reproducibility metadata for deterministic sampling,
   * Excel + PDF exports for analysis and auditability.

### Data acquisition (Apify)

Two Apify Actors are integrated via thin wrappers:

* **Primary collector** (`apify/instagram-hashtag-scraper`): used for most scraping, driven by batches of normalized query terms. The Actor is configured with:

  * `hashtags`: normalized query terms (leading `#` removed, whitespace trimmed, case-insensitive dedupe),
  * `resultsType`: currently fixed to `posts`,
  * `resultsLimit`: per-term cap (`apify.results_limit_per_query`),
  * `keywordSearch`: optional keyword-style discovery (`apify.keyword_search`).

* **Fallback collector** (`apify/instagram-scraper`): used during stagnation recovery for:

  * hashtag search (`searchType="hashtag"`) to discover additional hashtag URLs,
  * scraping discovered hashtag URLs via `directUrls`.

Dataset items are retrieved from the Actor’s default dataset and ingested into the pipeline in-memory for processing. The wrappers disable client-level retries when possible and apply a unified retry policy at the application layer.

### Query scheduling and adaptive discovery

The feedback loop uses a FIFO **term queue** with “present-only” deduplication:

* A term is deduped only while it is currently queued, which allows re-adding a term later (e.g., after it has been popped).
* Terms are normalized by trimming whitespace and stripping a leading `#`.

**Expansion** is driven by harvested hashtags from *eligible* posts:

* The loop maintains a hashtag frequency counter over eligible posts.
* Each iteration, it selects up to `querying.expansion.max_new_terms_per_iter` new terms whose eligible frequency meets `querying.expansion.min_hashtag_freq_in_eligible`.
* A blocklist is enforced using casefolded keys (`querying.expansion.blocklist_terms`) to prevent the introduction of known off-topic terms.

This strategy biases discovery toward terms empirically associated with accepted content, rather than expanding indiscriminately.

### Stagnation detection and fallback recovery

Stagnation is detected using a sliding window (`loop.stagnation_window`) over the count of newly eligible posts per iteration:

* If the sum of new eligible posts in the window is strictly below `loop.stagnation_min_new_eligible`, the loop triggers a fallback recovery path.
* On stagnation:

  * the per-query `results_limit_per_query` is increased (capped in code) to broaden the scrape yield,
  * the fallback Actor is invoked to discover additional hashtag URLs, which are then scraped via `directUrls`,
  * discovered hashtags are extracted from known Instagram hashtag URL patterns and enqueued if they are not blocklisted and not already attempted.

This dual approach (raising breadth + discovering new entry points) is intended to mitigate local maxima in term expansion.

### Normalization of heterogeneous Actor outputs

Apify dataset schemas may vary across Actor versions and settings. To keep downstream logic stable, dataset items are mapped into a single internal record (`NormalizedPost`) via best-effort extraction:

* **URL** is required; items lacking a usable URL are skipped.
* Fields are coerced conservatively:

  * IDs accept `str` or `int` → string normalized.
  * Lists accept either a string or list of strings.
  * Hashtags and mentions are normalized by stripping `#` / `@` prefixes and case-insensitive dedupe.
* Optional metadata such as `type`, `productType`, `isSponsored`, and `timestamp` is captured when present.

### Deduplication strategy

Post deduplication uses a stable dedupe key with a clear precedence order:

1. `id:<post_id>` when a post id is available,
2. `shortcode:<short_code>` when shortcode exists,
3. `url:<canonical_url>` otherwise.

URL canonicalization includes:

* lowering scheme and hostname,
* removing leading `www.`,
* stripping query strings and fragments,
* normalizing trailing slashes.

This avoids duplicating the same post encountered via multiple queries or Actor runs.

### Deterministic prechecks (LLM call reduction)

Before sending a post to the LLM, fast deterministic checks reduce cost and error surface area:

* Caption must exist and satisfy `filters.min_caption_chars`, unless the threshold is `0`.
* If `filters.allow_reels` is `false`, then non-feed content is rejected using `productType` when available, with a conservative fallback on `type`.
* If `filters.reject_if_sponsored_true` is `true`, posts with `isSponsored=true` are rejected early.

Prechecks are intentionally conservative: they reject only when a post clearly cannot be used, regardless of semantic topic.

### LLM labeling with Structured Outputs

For each candidate post passing prechecks, the system performs one-post-per-call labeling using OpenAI **Structured Outputs**:

* The request is configured with a strict JSON Schema (`DECISION_JSON_SCHEMA`) and `strict=true` to enforce a deterministic output shape.
* The model receives **only** extracted metadata:

  * `url`, `caption`, `hashtags`, `mentions`, `alt`, `type`, `productType`, `isSponsored`, `timestamp`.
* The prompt explicitly prohibits assuming video/image content beyond provided fields.

The decision schema includes:

* boolean eligibility + human-readable `eligibility_reasons`,
* structured results for:

  * language (`is_english`, confidence),
  * topic (`is_bodyweight_calisthenics`, confidence, notes),
  * commercial signal (`is_exclusively_commercial`, signals),
  * caption quality (`is_analyzable`, issues),
* tagging fields (`genre` enum + label lists),
* overall confidence in `[0, 1]`.

### Deterministic eligibility enforcement

Eligibility is computed deterministically from structured fields to avoid inconsistent model outputs:

A post is eligible only if **all** of the following hold:

* `language.is_english == True`
* `topic.is_bodyweight_calisthenics == True`
* `caption_quality.is_analyzable == True`
* `commercial.is_exclusively_commercial == False`

If the model’s `eligible` value conflicts with the computed result, the system overrides `eligible` and appends machine-readable markers into `eligibility_reasons`:

* `eligibility_overridden_accept` / `eligibility_overridden_reject`
* `eligibility_rule:<failure_key>` for each failure.

This ensures the eligibility gate remains stable across model behavior and prompt drift.

### Escalation policy and concurrency

Labeling supports escalation and parallelization:

* **Escalation**: If the primary model output is unparseable or `overall_confidence` falls below `openai.escalation_confidence_threshold`, the system retries the same post with `openai.model_escalation` (when configured and distinct).
* **Concurrency**: When `openai.max_concurrent_requests > 1`, the pipeline uses a thread pool and “forked” client instances to avoid sharing a single underlying HTTP client across threads.

Token counts are captured when available and persisted to support cost accounting and auditability.

### Retry, backoff, and transient failure handling

Network calls to both Apify and OpenAI are wrapped using an exponential backoff retry mechanism:

* Retries include the initial attempt (`max_attempts` counts total tries).
* Delay grows exponentially from `base_delay_seconds`, capped by `max_delay_seconds`, with optional jitter.
* When available, `Retry-After` hints are respected (with an optional cap).
* Retryability is determined via lightweight heuristics:

  * Apify: HTTP 429, HTTP 5xx, and network/timeout-like errors.
  * OpenAI: connection/timeout errors, HTTP 408/409/429, and HTTP 5xx.

A configurable per-iteration `loop.backoff_seconds` sleep provides additional pacing.

### Dominance guard and pool cap

Two deterministic controls keep the corpus balanced and bounded:

* **Dominance guard** (`filters.max_posts_per_user`):

  * For eligible posts, the system tracks per-author counts using `owner_username` when available, else `owner_id`.
  * Once a user reaches the configured cap, subsequent posts from that user are forced to ineligible with reason `dominance_guard`.
  * Setting the cap to `0` disables the mechanism.

* **Pool cap** (`targets.pool_n`):

  * When the eligible pool reaches the target size, newly eligible decisions can be forced to ineligible with reason `pool_cap_reached`.
  * This preserves a stable pool definition for sampling and statistics.

### Persistence model (SQLite) and resume semantics

State is persisted in SQLite to support resumable long runs:

* **Raw posts** (`raw_posts`):

  * keyed by `post_key` (the dedupe key),
  * stores `url`, `actor_source`, `raw_json`, and `fetched_at`,
  * upsert behavior updates the record while retaining `actor_source` when a new insert lacks it.

* **Decisions** (`llm_decisions`):

  * append-only decision records keyed to `post_key`,
  * include model name, eligibility, confidence, structured decision JSON, tokens, and timestamps.

* **Views**:

  * `latest_llm_decisions`: one most-recent decision per `post_key`,
  * `eligible_posts`: join of raw posts with latest decisions where `eligible=1`,
  * `final_sample_posts`: join of final sample keys to raw posts and latest decisions (for export/debug).

* **Runs** (`runs`):

  * per-run metadata: `run_id`, start/end timestamps, config hash, sampling seed, and dependency versions.

* **Final sampling** (`final_sample_runs`, `final_samples`):

  * stores sampling configuration and the selected post keys for reproducibility.

Resume behavior checks that:

* the run is not already finished,
* the config hash and sampling seed match the persisted run,
* resuming without a specified id selects the most recent unfinished run.

### Logging and audit trail

The CLI writes a JSONL run log (`run.log`) with:

* structured events (start, config loaded, batch lifecycle, exports),
* recorded exceptions including truncated tracebacks,
* optional URL context for scraping and LLM calls,
* consistent `session_id` and `run_id` tagging for correlation.

This supports post-hoc audits and debugging without requiring a separate observability stack.

### Reproducibility and deterministic sampling

Sampling is designed to be reproducible given:

* the ordered eligible pool keys used for sampling,
* a fixed integer `targets.sampling_seed`,
* stable hashing of config values.

Key reproducibility mechanisms:

* `config_sha256`: stable SHA-256 of config values (JSON-serialized with stable ordering).
* Eligible pool keys are collected in a stable order (decision timestamp then key).
* A SHA-256 hash is computed over the ordered pool key list (`pool_keys_sha256`).
* Final sampling uses `random.Random(seed).sample(...)` over the ordered pool and persists:

  * sampling seed,
  * pool/final sizes,
  * pool hash,
  * selected keys.

Exports include the pool hash and selected keys to allow downstream verification.

### Export methodology (Excel + PDF)

**Excel export** generates a multi-sheet workbook:

* `eligible_pool`: up to `targets.pool_n` eligible posts (stable early-first ordering),
* `final500`: subset of eligible pool selected into the final sample,
* `rejected`: recent rejected posts (capped),
* `run_metadata`: run and config snapshot plus reproducibility metadata,
* `tag_summary`: aggregate counts across tags, genres, and hashtags.

To mitigate spreadsheet formula injection, text fields are sanitized:

* values starting with `=`, `+`, `-`, or `@` are prefixed with `'`.

**PDF export** generates a codebook-style document with:

* run summary and configuration snapshot,
* methods overview synthesized from config values and recorded actor runs,
* operational inclusion/exclusion rules,
* summary statistics computed from the eligible pool used for export,
* a listing of recorded actor runs.

The PDF uses standard ReportLab flowables and escapes inserted text for safe rendering.

### Extensibility notes

The implementation favors clear seams for extension:

* scrapers and classifier can be injected (used by offline mode and unit tests),
* strict schemas and deterministic gates isolate “policy” from model variability,
* the SQLite store is intentionally minimal but contains enough structure to add:

  * alternative label schemas,
  * additional exports (e.g., Parquet/JSONL),
  * richer provenance fields (e.g., term → post attribution) without redesigning core flow.
