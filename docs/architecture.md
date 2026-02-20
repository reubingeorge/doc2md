# doc2md — Architecture & Implementation Specification

Complete system design for the doc2md agentic document-to-markdown converter. This is the canonical reference for all implementation work.

---

## System Overview

doc2md is a production-grade Python package that converts documents (PDFs, JPEGs, PNGs) to structured Markdown using composable VLM agents orchestrated through YAML-defined pipelines with a shared blackboard for inter-agent communication.

It is an **agentic pipeline system** with three core abstractions:

1. **Agents** — Single-purpose VLM tasks defined in YAML. Each agent does ONE thing (extract tables, read handwriting, validate output, extract metadata). An agent knows nothing about other agents.

2. **Pipelines** — YAML-defined DAGs (directed acyclic graphs) of agents. A pipeline defines how agents compose: sequentially, in parallel, or routed per-page. The pipeline is the unit of document processing.

3. **Blackboard** — A typed, region-based shared memory that agents read from and write to during pipeline execution. The blackboard carries structured observations (language, layout, cross-page continuations, uncertain regions) so downstream agents can adapt.

---

## THE BLACKBOARD — DESIGN SPECIFICATION

The blackboard is the inter-agent communication backbone. It replaces a naive "pass markdown string to next step" approach with a structured, observable, typed shared memory.

### Why a blackboard (not just passing markdown between steps)

Without a blackboard, these scenarios fail:

| Scenario | Failure |
|---|---|
| `metadata_extract` discovers the document is in French | `text_extract` prompt is English-hardcoded, can't adapt |
| `table_extract` on page 3 finds a table that continues to page 4 | Pages are processed independently, table is split |
| First agent detects 2-column layout | Later agents use single-column prompts, garble the output |
| `text_extract` flags a blurry region | `validator` reviews everything equally instead of focusing there |
| Classification detects handwriting on some pages | `handwriting` agent runs on ALL pages, wasting cost |

### Blackboard architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        BLACKBOARD                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  REGION: document_metadata        (written once, read-many) │  │
│  │    language: "fr"                                           │  │
│  │    date_format: "DD/MM/YYYY"                                │  │
│  │    layout: "two_column"                                     │  │
│  │    page_count: 15                                           │  │
│  │    content_types: ["prose", "tables", "handwriting"]        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  REGION: page_observations        (per-page, append-only)  │  │
│  │    page_3:                                                  │  │
│  │      content_type: "table"                                  │  │
│  │      continues_on_next: true                                │  │
│  │      quality_score: 0.4                                     │  │
│  │      uncertain_regions: [{area: "bottom_right", ...}]       │  │
│  │    page_4:                                                  │  │
│  │      content_type: "table"                                  │  │
│  │      continues_from_previous: true                          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  REGION: step_outputs             (per-step markdown)       │  │
│  │    metadata_extract: "---\ntitle: ...\n---"                 │  │
│  │    text_extract: "## Chapter 1\n..."                        │  │
│  │    table_extract: "| Col A | Col B |\n..."                  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  REGION: agent_notes              (freeform, namespaced)    │  │
│  │    metadata_extract:                                        │  │
│  │      found_watermark: true                                  │  │
│  │      watermark_text: "DRAFT"                                │  │
│  │    text_extract:                                            │  │
│  │      unusual_formatting: "Roman numeral headers"            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  REGION: confidence_signals       (per-page, per-step)      │  │
│  │    page_3.table_extract:                                    │  │
│  │      self_assessment: 0.72                                  │  │
│  │      validation_pass_rate: 0.80                             │  │
│  │      image_quality: 0.40                                    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  EVENT LOG                        (append-only, ordered)    │  │
│  │    [t=0] WRITE document_metadata.language = "fr"            │  │
│  │           by: metadata_extract                              │  │
│  │    [t=1] WRITE page_observations.page_3.continues = true    │  │
│  │           by: table_extract                                 │  │
│  │    [t=2] READ  document_metadata.language                   │  │
│  │           by: text_extract                                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Blackboard optimization principles

This is a document processing pipeline, not an open-ended reasoning system. Optimize accordingly:

1. **Typed regions, not free-form.** The blackboard has 5 fixed regions with Pydantic schemas. Agents can't write arbitrary keys to arbitrary places. This prevents bloat and makes the blackboard self-documenting.

2. **No polling loop.** Classic blackboard systems have a control loop where agents check the board each cycle. We don't need that — our pipeline DAG already defines execution order. The blackboard is passive shared state, not an active coordination mechanism.

3. **Subscription-based reads.** Agents declare which blackboard regions they read in their YAML. The prompt builder only injects the regions they subscribed to, keeping prompts small. An agent that only needs `document_metadata.language` doesn't get the entire `page_observations` region dumped into its context.

4. **Write-on-complete, not streaming.** Agents write to the blackboard after their VLM call completes, not during. This avoids partial-write issues and simplifies the execution model.

5. **Append-only for collections.** `page_observations` and `agent_notes` are append-only. Agents can add entries but never delete or overwrite entries from other agents. `document_metadata` allows overwrites (last-write-wins with logged warning if values conflict).

6. **Event log for observability.** Every read and write is logged with timestamp, agent name, region, and key. This makes debugging multi-agent pipelines trivial — you can replay exactly what each agent saw and wrote.

7. **Lazy serialization for prompts.** When building a prompt, the blackboard serializes only the subscribed regions into the Jinja2 context. Large regions (like `step_outputs` with many pages of markdown) are never serialized unless explicitly subscribed to.

8. **Cache-key participation.** The blackboard state that an agent reads is included in that agent's cache key hash. If upstream agents produce different blackboard writes, downstream cache keys change automatically.

### Blackboard Python interface

```python
class Blackboard:
    """Typed, region-based shared memory for pipeline execution."""

    # ─── Regions (Pydantic models) ───
    document_metadata: DocumentMetadata     # language, layout, date_format, etc.
    page_observations: Dict[int, PageObs]   # per-page observations
    step_outputs: Dict[str, str]            # step_name → markdown output
    agent_notes: Dict[str, Dict[str, Any]]  # step_name → freeform notes
    confidence_signals: Dict[str, Dict]     # "page_N.step_name" → signals

    # ─── Event log ───
    event_log: List[BlackboardEvent]        # append-only audit trail

    # ─── Core operations ───
    def read(self, region: str, key: str, reader: str) -> Any:
        """Read a value. Logs the read event."""

    def write(self, region: str, key: str, value: Any, writer: str) -> None:
        """Write a value. Validates against region schema. Logs event."""

    def query(self, region: str, filter: Callable) -> List[Any]:
        """Query a region with a filter function."""

    def subscribe(self, regions: List[str]) -> BlackboardView:
        """Return a read-only view of specific regions for prompt injection."""

    def snapshot(self) -> dict:
        """Frozen snapshot for cache key computation."""

    def merge_parallel(self, boards: List['Blackboard']) -> None:
        """Merge blackboard writes from parallel step execution."""

    # ─── Prompt integration ───
    def to_jinja_context(self, subscriptions: List[str]) -> dict:
        """Serialize subscribed regions into a dict for Jinja2 prompt rendering.
        Only includes subscribed regions. Respects size limits."""
```

### How agents interact with the blackboard

**In agent YAML — declare subscriptions and outputs:**

```yaml
agent:
  name: text_extract
  blackboard:
    reads:                              # What this agent needs
      - document_metadata.language
      - document_metadata.layout
      - page_observations.*.quality_score
    writes:                             # What this agent produces
      - page_observations.*.uncertain_regions
      - agent_notes.text_extract
    writes_via: hybrid                  # prompt_elicited | code_computed | hybrid
```

**In prompt templates — read from blackboard via Jinja2:**

```yaml
prompt:
  system: |
    You are a text extraction specialist.
    {% if bb.document_metadata.language %}
    Document language: {{ bb.document_metadata.language }}.
    {% endif %}
    {% if bb.document_metadata.layout == "two_column" %}
    IMPORTANT: Two-column layout. Read columns left-to-right.
    {% endif %}
    {% for page_num, obs in bb.page_observations.items() %}
    {% if obs.quality_score and obs.quality_score < 0.5 %}
    WARNING: Page {{ page_num }} has low image quality ({{ obs.quality_score }}).
    {% endif %}
    {% endfor %}
```

**Writing context — VLM-elicited (agent output includes a `<blackboard>` block):**

```
The VLM outputs:

## Chapter 1: Introduction
The quick brown fox...

<blackboard>
page_observations:
  3:
    uncertain_regions:
      - area: bottom_right
        reason: blurry text
        confidence: low
agent_notes:
  text_extract:
    unusual_formatting: "Roman numeral section headers"
</blackboard>
```

The response parser extracts `<blackboard>`, validates against the agent's declared `writes` schema, writes to the blackboard, and strips the block from the markdown output. The user never sees blackboard data.

**Writing context — code-computed (deterministic, no VLM cost):**

```python
# Built-in blackboard writers (run after VLM call)
@blackboard_writer("page_observations.*.continues_on_next_page")
def detect_continuations(markdown: str, page_num: int) -> bool:
    """Heuristic: does this page's markdown end mid-table or mid-sentence?"""
    return markdown.rstrip().endswith("|") or not markdown.rstrip().endswith(".")

@blackboard_writer("document_metadata.language")
def detect_language(markdown: str) -> str:
    """Detect language from extracted text."""
    from langdetect import detect
    return detect(markdown)
```

### Blackboard merge for parallel steps

When steps run in parallel, each gets a COPY of the blackboard. After all complete, the pipeline engine merges their writes:

```
Pre-parallel blackboard:
  document_metadata: {language: "fr"}

Parallel step A writes: page_observations.3.uncertain_regions = [...]
Parallel step B writes: page_observations.3.table_count = 2
Parallel step C writes: agent_notes.handwriting = {found_signature: true}

Merge rules:
  document_metadata: last-write-wins (warn on conflict)
  page_observations: deep merge (per-page, per-field)
  step_outputs: keyed by step name (no conflict possible)
  agent_notes: keyed by step name (no conflict possible)
  confidence_signals: keyed by step+page (no conflict possible)

Post-merge blackboard:
  document_metadata: {language: "fr"}
  page_observations.3: {uncertain_regions: [...], table_count: 2}
  agent_notes: {handwriting: {found_signature: true}}
```

### Conditional step execution via blackboard

```yaml
steps:
  - name: handwriting_extract
    agent: handwriting
    condition: "'handwriting' in bb.document_metadata.content_types"
    # Only runs if classifier or metadata agent detected handwriting

  - name: expensive_validation
    agent: validator
    condition: "any(obs.quality_score < 0.5 for obs in bb.page_observations.values())"
    # Only runs if any page has low quality — saves cost on clean documents
```

### Cross-page grouping via blackboard

The page router reads `page_observations.*.continues_on_next_page` from the blackboard. When it finds consecutive pages marked as continuing, it groups them and sends both images to the same agent in a single VLM call:

```
Page 3: table_extract → bb.page_observations.3.continues_on_next_page = true
Page router sees this → groups page 3+4 → sends both to table_extract
Result: agent sees the FULL table across both pages
```

---

## PIPELINE EXECUTION ENGINE

### Step types

| Type | Description |
|---|---|
| `agent` | Single VLM call using an agent YAML. Default step type. |
| `parallel` | Fan-out: run multiple sub-steps concurrently. Fan-in: merge blackboard writes + outputs. |
| `page_route` | Classify each page (VLM, rules, or hybrid) and route to different agents. |
| `code` | Run a deterministic Python function. No VLM call. |

### Pipeline YAML structure

```yaml
pipeline:
  name: legal_contract
  version: "1.0"
  description: "Multi-section legal contracts with mixed content types"

  steps:
    - name: metadata
      agent: metadata_extract
      input: image
      pages: [1]                        # Only page 1

    - name: body_extraction
      type: page_route
      depends_on: [metadata]
      pages: [2:]
      cross_page_aware: true            # Read bb.page_observations for continuations
      router:
        strategy: hybrid
        rules:
          - pages: [1, 2]
            agent: cover_page
          - pages: [-1]
            agent: signature_page
        vlm_fallback:
          model: gpt-4.1-nano
          batch_size: 8
          categories:
            legal_prose: { agent: legal_text_extract }
            tables: { agent: table_extract }
            signatures: { agent: handwriting }
        default_agent: generic

    - name: enrichment
      type: parallel
      depends_on: [metadata, body_extraction]
      steps:
        - name: summary
          agent: summarize
        - name: key_terms
          agent: legal_key_terms
      merge:
        agent: enrichment_merger

    - name: final_validate
      agent: validator
      input: image_and_previous
      depends_on: [enrichment]

  page_merge:
    agent: document_merger

  confidence:
    strategy: weighted_average
    step_weights:
      metadata: 0.1
      body_extraction: 0.5
      enrichment: 0.1
      final_validate: 0.3
```

### Step graph resolution

Parse pipeline YAML into a DAG. Topological sort determines execution order. Steps with no `depends_on` implicitly depend on the previous step in the list. Steps with `depends_on: []` (empty) run immediately.

### Data flow between steps

Each step receives:
- **Image bytes** (original page images, or subset via `pages:` selector)
- **Blackboard view** (only the regions declared in agent's `blackboard.reads`)
- **Previous step output** (markdown from the step(s) in `depends_on`)

Each step produces:
- **Markdown** (the extracted content — stored in `bb.step_outputs`)
- **Blackboard writes** (observations, notes, confidence signals)
- **Confidence score** (per-page, per-step)

---

## AGENT YAML SPECIFICATION

An agent is a single-purpose VLM task. Full schema:

```yaml
agent:
  name: string                          # Unique identifier
  version: string                       # Semver for cache invalidation
  description: string                   # Used by classifier to describe this agent

  model:
    preferred: string                   # e.g. "gpt-4.1-mini"
    fallback: [string]                  # Fallback chain
    max_tokens: int                     # Max output tokens
    temperature: float                  # 0.0 for extraction, 0.3 for creative

  input: enum                           # What this agent receives:
    # image              — original page image(s) only
    # previous_output    — markdown from previous step
    # image_and_previous — both image + prior markdown
    # previous_outputs   — dict of all named prior step outputs
    # previous_output_only — text only, no image (e.g. summarize)

  preprocessing:                        # Image transforms before VLM call
    - name: string
      params: dict

  prompt:
    system: string                      # Jinja2 template (has access to {{ bb.* }})
    user: string                        # Jinja2 template

  blackboard:
    reads: [string]                     # Dot-paths to blackboard regions/keys
    writes: [string]                    # Dot-paths this agent may write to
    writes_via: enum                    # prompt_elicited | code_computed | hybrid
    write_schema:                       # Optional: Pydantic-style schema for writes
      key: {type: string, ...}
    code_writers:                        # For code_computed / hybrid
      - function: string                # e.g. "detect_language"
        input: enum                     # "markdown" | "image" | "both"
        output_key: string              # Blackboard key to write to

  confidence:
    signals: [string]                   # Which signals to use
    weights: {string: float}            # Signal weights
    expected_fields: [string]           # For completeness check
    calibration:
      method: enum                      # platt_scaling | isotonic | manual
      manual_curve: [[float, float]]    # [raw, calibrated] pairs

  validation:
    - rule: string
      params: dict

  retry:
    max_attempts: int
    strategy: enum                      # exponential | linear | fixed
    retry_on: [string]                  # error types to retry

  output_format: string                 # Optional: "frontmatter", "json"
  postprocessing: [string]              # Post-VLM cleanup functions
```

---

## AUTO-CLASSIFICATION

A lightweight VLM call on page 1 that selects the best **pipeline** (not agent).

- **Model:** `gpt-4.1-nano` (~$0.001/call)
- **Input:** Page 1 image only
- **Output:** `{pipeline_name, confidence, reasoning, content_types_detected}`
- **Dynamic prompt:** Built from pipeline registry — adding a custom pipeline YAML auto-registers it
- **Override:** `pipeline="legal_contract"` or `agent="receipt"` skips classification
- **Fallback:** confidence < 0.7 → `generic` pipeline
- **Content types detected** are written to `bb.document_metadata.content_types` for downstream conditional execution

---

## CACHING

Content-addressed, step-level caching with blackboard-aware invalidation.

### Cache key

```python
key = SHA256(
    image_hash +              # What image
    pipeline_name +           # Which pipeline
    step_name +               # Which step
    agent_name +              # Which agent
    agent_version +           # Agent YAML version
    model_id +                # Which model
    prompt_hash +             # Prompt content
    blackboard_snapshot_hash  # Hash of blackboard state THIS AGENT READ
)
```

The `blackboard_snapshot_hash` is critical: it includes ONLY the regions this agent subscribed to (from `blackboard.reads`). If an upstream agent writes different values to a region this agent reads, the cache key changes. If an upstream agent writes to a region this agent DOESN'T read, the cache key is unaffected.

### Two tiers

| Tier | Backend | Lifetime | Default Size |
|---|---|---|---|
| L1: Memory | LRU dict | Session | 500 MB |
| L2: Disk | SQLite | Persistent | 5 GB |

- Lookup: L1 → L2 → MISS
- Store: L1 + async L2
- Low-confidence results: shorter TTL (1 hour vs 7 days)
- Batch deduplication: hash all pages first, deduplicate, only process unique pages

---

## CONFIDENCE SCORING

Multi-signal weighted scoring. 0.0–1.0 per page, per step, per document.

### Signals

| Signal | Always Available | Cost | Description |
|---|---|---|---|
| VLM self-assessment | Yes | Zero (parsed from existing output) | VLM rates its own confidence per-field |
| Logprobs analysis | Only on GPT-4.1/4o family | Zero (from same API call) | Geometric mean of content token probabilities |
| Validation pass rate | Yes | Zero (runs rules from YAML) | % of validation rules passed |
| Completeness check | Yes | Zero (checks expected fields) | Expected fields present |
| Consistency check | Optional | 2× cost (double extraction) | Compare two extractions |
| Image quality | Yes | Zero (pre-VLM image analysis) | Blur, contrast, DPI, noise |

### Adaptive weight redistribution

When a signal is unavailable (e.g., logprobs on GPT-5+), redistribute its weight proportionally to remaining signals.

### Calibration

VLMs are systematically overconfident. Apply manual calibration curves (shipped per agent) or Platt scaling (user-trained with ground truth).

### Decision thresholds

- ≥ 0.8 → HIGH (accept)
- ≥ 0.6 → MEDIUM (accept with note)
- ≥ 0.3 → LOW (flag for human review, retry if budget remains)
- < 0.3 → FAILED (mark failed, continue with other pages)

### Low-confidence retry

Confidence < threshold → enhance image (upscale 2×, increase contrast) → re-extract → if still low, accept with human-review flag.

### Pipeline-level confidence

Aggregate step-level scores via configurable strategy: `minimum`, `weighted_average`, or `last_step` (use validator's score).

---

## CONCURRENCY

Two-tier async parallelism:

```
Batch Dispatcher (asyncio) → File Workers → Page Workers
                                            ↓
                                    Semaphore + Rate Limiter
```

- Files processed concurrently (default: 5)
- Pages within files concurrent (default: 10)
- Rate limiter: token-bucket for RPM + TPM against OpenAI limits
- Cache-aware: check cache before dispatching, only send misses to workers
- Backpressure: semaphore blocks without memory growth

---

## ERROR HANDLING

```
TRANSIENT (retry with backoff):  429, 500, 502, 503, timeout, connection
RECOVERABLE (retry modified):   validation failure, low confidence, content filter, token limit
TERMINAL (fail fast):           401, 402, 404 (try fallback model first), bad input
```

Partial result recovery: if pages 1-8 succeed but page 9 fails, return partial result with page 9 marked FAILED.

---

## MODEL DISCOVERY

OpenAI's API returns no capability metadata. Solution:

1. **Curated allowlist** (`models.yaml`): model_id, tier, priority, logprobs support
2. **Live validation**: intersect allowlist with user's API access
3. **User override**: any model string with warning
4. **Default**: `gpt-4.1-mini` (best cost/quality, supports logprobs)
5. **Classifier**: `gpt-4.1-nano` (cheapest)

---

## BUILT-IN AGENTS

| Agent | Purpose | Model |
|---|---|---|
| `_classifier` | Classifies documents into pipeline types | gpt-4.1-nano |
| `_page_classifier` | Classifies individual pages for page_route | gpt-4.1-nano |
| `generic` | General-purpose extraction | gpt-4.1-mini |
| `text_extract` | Prose/paragraph text only | gpt-4.1-mini |
| `table_extract` | Tabular data as Markdown tables | gpt-4.1-mini |
| `handwriting` | Handwritten text and annotations | gpt-4.1 |
| `metadata_extract` | Document metadata (dates, IDs, parties) | gpt-4.1-nano |
| `summarize` | Summary from extracted content (no image) | gpt-4.1-mini |
| `validator` | Cross-checks extraction against source image | gpt-4.1 |
| `document_merger` | Merges multi-page/multi-step outputs | gpt-4.1-mini |

## BUILT-IN PIPELINES

| Pipeline | Steps | Use Case |
|---|---|---|
| `generic` | `generic` (single step) | Default fallback |
| `receipt` | `receipt_extract` → `validator` | Receipts, invoices |
| `structured_pdf` | `metadata` ∥ `text` ∥ `tables` → `merger` → `validator` | Forms |
| `academic` | `metadata` → `text_extract` → `summarize` | Papers |
| `legal_contract` | `metadata` → `page_route` → `enrichment` → `validator` | Contracts |
| `handwritten` | `handwriting` → `validator` | Notes |
| `mixed_document` | `page_route`(text, tables, handwriting) → `merger` | Unknown |

---

## BACKWARD COMPATIBILITY

A standalone agent YAML (no pipeline) is wrapped in an implicit single-step pipeline. The `agent=` parameter still works. All v2-style configs work unchanged.

---

## PUBLIC API

### Python

```python
import doc2md

# Simple
result = doc2md.convert("receipt.pdf")
result.markdown                      # Final output
result.classified_as                 # Pipeline name
result.confidence.overall            # 0.85
result.confidence.needs_human_review # False

# Specify pipeline
result = doc2md.convert("contract.pdf", pipeline="legal_contract")
result.steps["metadata"].markdown    # Step output
result.steps["body_extraction"].page_assignments  # {3: "legal_text", ...}

# Access blackboard (for debugging/inspection)
result.blackboard.document_metadata.language  # "en"
result.blackboard.event_log                   # Full audit trail

# Batch
results = doc2md.convert_batch(["a.pdf", "b.png"], max_workers=5)

# Full control
converter = Doc2Md(api_key="sk-...", config_dir="./agents/", pipeline_dir="./pipelines/")
converter.cache.stats()
converter.cache.clear()
```

### CLI

```bash
doc2md convert file.pdf -o output.md                    # Auto-classify
doc2md convert file.pdf --pipeline legal_contract       # Explicit pipeline
doc2md convert file.pdf --agent receipt                  # Single agent
doc2md convert ./docs/ --output-dir ./md/ --workers 5   # Batch
doc2md convert file.pdf -vv                             # Show step-by-step + blackboard
doc2md pipelines                                        # List available
doc2md cache stats / clear                              # Cache management
doc2md validate-pipeline my_pipeline.yaml               # Validate config
```

---

## CONFIGURATION HIERARCHY

```
Package defaults → Global (~/.doc2md/config.yaml) → Project (./doc2md.yaml)
→ Environment variables → Pipeline YAML → Agent YAML → Runtime arguments
```

---

## DEPENDENCIES

**Core (no GPU):** `openai`, `httpx`, `pydantic`, `PyYAML`, `Jinja2`, `Pillow`, `pdf2image`, `tenacity`, `rich`, `click`, `numpy`, `thefuzz`

**Optional:** `opencv-python-headless` (advanced preprocessing), `PyMuPDF` (fast PDF), `scikit-learn` (Platt scaling), `langdetect` (language detection)

---

## SECURITY

- API keys: never log/serialize. Accept via env var, config (warn permissions), or param.
- YAML: `yaml.safe_load()` only. Never `yaml.load()`.
- Jinja2: `SandboxedEnvironment` always. Blackboard values flow into templates — must be safe.
- Blackboard writes: validated against declared schema. Unexpected keys dropped. Types enforced.
- Code steps: only registered functions. No arbitrary execution from YAML.
- Cache: warn about permissions. Provide encrypt option.
- File paths: sanitize, no traversal, no symlinks.
- Image size: 50MB max default.

---

## PACKAGE STRUCTURE

```
doc2md/
├── src/doc2md/
│   ├── __init__.py                    # Public API
│   ├── core.py                        # Doc2Md class
│   ├── cli.py                         # Click CLI
│   ├── config/                        # schema.py, loader.py, defaults.py, hierarchy.py
│   ├── models/                        # discovery.py, allowlist.py, models.yaml
│   ├── blackboard/                    # THE BLACKBOARD MODULE
│   │   ├── board.py                   # Blackboard class
│   │   ├── regions.py                 # Pydantic models for typed regions
│   │   ├── events.py                  # Event log types
│   │   ├── merge.py                   # Parallel merge logic
│   │   ├── serializer.py             # Jinja2 context serialization
│   │   └── writers.py                 # Built-in code-computed writers
│   ├── pipeline/                      # engine.py, graph.py, step_executor.py,
│   │                                  # page_router.py, parallel_executor.py,
│   │                                  # merger.py, data_flow.py,
│   │                                  # preprocessor.py, postprocessor.py
│   ├── agents/                        # engine.py, registry.py, classifier.py,
│   │   ├── builtin/agents/            # YAML agent definitions
│   │   └── builtin/pipelines/         # YAML pipeline definitions
│   ├── cache/                         # manager.py, keys.py, memory.py, disk.py, stats.py
│   ├── confidence/                    # engine.py, signals/, calibration.py,
│   │                                  # combiner.py, report.py
│   ├── vlm/                           # client.py, prompt_builder.py, response_parser.py
│   ├── concurrency/                   # pool.py, rate_limiter.py
│   ├── errors/                        # exceptions.py, retry.py, fallback.py
│   ├── transforms/                    # Code step functions
│   ├── validation/                    # rules.py, runner.py
│   ├── types.py                       # Shared Pydantic models
│   └── utils/                         # image.py, pdf.py, markdown.py
├── agents/                            # User custom agents
├── pipelines/                         # User custom pipelines
└── tests/                             # Comprehensive test suite
```

---

## OBSERVABILITY

Every conversion emits structured events covering: classification, pipeline start/end, step start/end, blackboard reads/writes, cache hits/misses, VLM calls, retries, confidence scores, and final completion with cost breakdown. The blackboard event log provides a complete audit trail of every read and write.

---

## KEY DESIGN DECISIONS (for reference)

| Decision | Chosen | Why |
|---|---|---|
| VLM-first vs OCR-first | VLM-first | VLMs do layout + extraction in one call |
| Agent = unit of work, not unit of processing | Agents do one thing; pipelines compose them | Real documents need multiple specialists |
| Blackboard vs context dict | Blackboard | Typed regions, subscriptions, event log, parallel merge, cache participation |
| Blackboard: active vs passive | Passive (no polling loop) | Pipeline DAG defines execution order; blackboard is shared state, not coordinator |
| Blackboard: schema | Typed regions with Pydantic | Prevents bloat, self-documenting, validates writes |
| Pipeline format | YAML with step graph | Consistent with agents; DAG covers all patterns |
| Page routing | 3 modes: vlm, rules, hybrid | Rules are free, VLM handles unknowns, hybrid is best |
| Cache key | SHA256 of (image + pipeline + step + agent + model + prompt + blackboard_snapshot) | Deterministic, blackboard-aware invalidation |
| Confidence | Multi-signal weighted | Logprobs deprecated on GPT-5+; multi-signal is resilient |
| Concurrency | asyncio (not threads) | I/O-bound API calls |
| Backward compatibility | Standalone agent YAML wrapped in implicit pipeline | Simple cases stay simple |

---

