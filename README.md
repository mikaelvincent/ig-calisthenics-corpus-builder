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

### End-to-end pipeline

1. **Initialize / resume state**

   * A run record is created (or resumed) in SQLite.
   * Previously seen post keys (id/shortcode/url) are loaded to prevent duplicates.

2. **Scrape candidate posts**

   * The primary Apify actor is invoked on a batch of query terms.
   * Actor run metadata (actor id, run id, dataset id) is recorded for auditability.

3. **Normalize and deduplicate**

   * Each dataset item is normalized into a stable post record.
   * Deduplication key preference:

     1. `id:<post_id>`
     2. `shortcode:<short_code>`
     3. `url:<canonicalized_url>`

4. **Deterministic prechecks (LLM-sparing)**

   * Reject before LLM if:

     * caption missing, or
     * caption shorter than `filters.min_caption_chars`, or
     * reels disallowed but content is non-feed, or
     * sponsored pre-rejection enabled and `isSponsored=true`.

5. **One-post-per-call LLM labeling (Structured Outputs)**

   * Inputs include URL, caption, hashtags, mentions, alt/accessibility caption, type/productType, isSponsored, and timestamp (when present).
   * The model returns strict JSON matching a fixed schema:

     * language fields
     * topic relevance fields
     * caption quality fields
     * commercial-only detection fields
     * narrative/discourse tags
     * `overall_confidence`

6. **Eligibility is enforced deterministically**

   * Final acceptance requires all of:

     * `language.is_english == true`
     * `topic.is_bodyweight_calisthenics == true`
     * `caption_quality.is_analyzable == true`
     * `commercial.is_exclusively_commercial == false`
   * If the model’s `eligible` conflicts, the system overrides it and appends machine-readable reasons.

7. **Dominance guard**

   * If enabled, caps eligible posts per user (`filters.max_posts_per_user`) using username or owner id when available.

8. **Query expansion**

   * Hashtags from eligible posts are counted.
   * High-frequency hashtags (respecting thresholds and blocklist) are enqueued for future scraping.

9. **Stagnation handling**

   * If eligible growth stays below the configured threshold over the rolling window:

     * the per-query results limit is increased (bounded),
     * the fallback actor is invoked to search for additional hashtags and scrape discovered hashtag URLs,
     * discovered hashtags are added back into the queue.

10. **Deterministic sampling and persistence**

* When the eligible pool meets `targets.pool_n`, the final sample is selected using:

  * `targets.sampling_seed`
  * the ordered pool key list (order is part of reproducibility)
* The sample is persisted (run meta + selected keys) in SQLite.

11. **Exports**

* `corpus.xlsx` includes:

  * `final500` — final sampled posts
  * `eligible_pool` — eligible posts up to pool cap
  * `rejected` — a capped set of rejected decisions for audit
  * `run_metadata` — config dump, reproducibility fields, run and actor metadata
  * `tag_summary` — counts for genres, labels, moves, signals, hashtags, model usage
* `codebook.pdf` includes:

  * run summary and configuration snapshot
  * operational inclusion/exclusion rules
  * tag field definitions and allowed genre values
  * summary statistics computed from the eligible pool
  * actor run listing

### Storage and reproducibility notes

* SQLite stores:

  * `runs`, `raw_posts`, `llm_decisions`, `apify_actor_runs`
  * views for latest decisions and eligible posts
  * final sample tables (`final_sample_runs`, `final_samples`)
* Reproducibility metadata includes:

  * config hash
  * sampling seed
  * pool key list hash (`pool_keys_sha256`)
  * persisted selected post keys for the run
