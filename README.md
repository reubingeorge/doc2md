# doc2md

[![CI](https://github.com/reubingeorge/doc2md/actions/workflows/ci.yml/badge.svg)](https://github.com/reubingeorge/doc2md/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Agentic document-to-markdown converter using composable VLM agents orchestrated through YAML-defined pipelines.

### Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-3776AB?logo=python&logoColor=white)
![YAML](https://img.shields.io/badge/YAML-CB171E?logo=yaml&logoColor=white)
![Jinja2](https://img.shields.io/badge/Jinja2-B41717?logo=jinja&logoColor=white)
![Click](https://img.shields.io/badge/Click_CLI-000000?logo=gnubash&logoColor=white)
![Rich](https://img.shields.io/badge/Rich-000000?logo=gnubash&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=githubactions&logoColor=white)

## Features

- **Multi-agent pipelines** — Chain specialized VLM agents (text extraction, table parsing, handwriting recognition) via YAML-defined DAGs
- **Flexible output** — Save as a single merged markdown file or per-page files for granular control
- **Auto-classification** — Detects document type and selects the optimal pipeline automatically
- **Blackboard architecture** — Typed, region-based shared memory enables inter-agent communication
- **Confidence scoring** — 6-signal confidence engine with adaptive weight redistribution and calibration
- **Two-tier caching** — L1 in-memory + L2 SQLite on-disk with content-addressed keys
- **Preprocessing & postprocessing** — Image transforms (deskew, denoise, contrast) and markdown cleanup (heading normalization, table alignment, deduplication)
- **Concurrency** — Async batch processing with rate limiting and semaphore-bounded parallelism
- **Error recovery** — Retry with exponential/linear/fixed backoff, model fallback chains
- **Extensible** — Drop YAML files into `agents/` and `pipelines/` directories to add custom agents and workflows

## Installation

```bash
pip install doc2md
```

With optional dependencies:

```bash
pip install doc2md[cv]          # OpenCV for advanced image preprocessing
pip install doc2md[calibration] # scikit-learn for confidence calibration
pip install doc2md[all]         # Everything including dev tools
```

## Authentication

Provide your OpenAI API key in one of two ways:

**Option 1 — Environment variable (recommended):**

```bash
export OPENAI_API_KEY="sk-..."
```

**Option 2 — Pass directly in code:**

```python
import doc2md

result = doc2md.convert("receipt.pdf", api_key="sk-...")

# Or with the full client
converter = doc2md.Doc2Md(api_key="sk-...")
```

## Quick Start

```python
import doc2md

# Auto-classify and convert
result = doc2md.convert("receipt.pdf")
print(result.markdown)
print(f"Confidence: {result.confidence:.2f}")

# Use a specific pipeline
result = doc2md.convert("contract.pdf", pipeline="legal_contract")

# Use a specific model
result = doc2md.convert("notes.png", model="gpt-4o")

# Save to file
result = doc2md.convert("report.pdf", output="report.md")

# Save per-page (creates page_001.md, page_002.md, ... in output dir)
result = doc2md.convert("report.pdf", output="output/", per_page=True)

# Access per-page results programmatically
result = doc2md.convert("report.pdf")
for i, page_md in enumerate(result.page_markdowns, 1):
    print(f"--- Page {i} ---")
    print(page_md)

# Or save via the result object
result.save("report.md")                  # Single file
result.save("output/", per_page=True)     # Per-page files

# Batch convert
results = doc2md.convert_batch(["a.pdf", "b.png"], max_workers=5)

# Full control
converter = doc2md.Doc2Md(api_key="sk-...", no_cache=True)
result = await converter.convert_async("document.pdf")
await converter.close()
```

## CLI

```bash
# Single file
doc2md convert document.pdf -o output.md

# Per-page output (saves page_001.md, page_002.md, ... in output dir)
doc2md convert document.pdf -o output/ --per-page

# Explicit pipeline
doc2md convert document.pdf --pipeline receipt

# Batch (all supported files in a directory)
doc2md convert ./docs/ --output-dir ./md/ --workers 5

# Verbose (show steps, tokens, confidence)
doc2md convert document.pdf -o out.md -vv

# Management
doc2md pipelines                          # List available pipelines
doc2md cache stats                        # Show cache statistics
doc2md cache clear                        # Clear all caches
doc2md validate-pipeline my_pipeline.yaml # Validate a YAML config
```

## Architecture

### Three Core Abstractions

| Concept | Description |
|---------|-------------|
| **Agents** | Single-purpose VLM tasks defined in YAML — one agent = one focused job |
| **Pipelines** | YAML-defined DAGs of agents with 4 step types: `agent`, `parallel`, `page_route`, `code` |
| **Blackboard** | Typed shared memory with 5 regions that agents read/write during execution |

### Built-in Pipelines

| Pipeline | Use Case |
|----------|----------|
| `generic` | Default fallback — single-step extraction |
| `receipt` | Receipts and invoices |
| `structured_pdf` | Forms with parallel metadata / text / table extraction |
| `academic` | Research papers with validation |
| `legal_contract` | Contracts with page-level routing |
| `handwritten` | Handwritten notes |
| `mixed_document` | Unknown documents with page-type routing |

### Pipeline Processing Flow

```
Input (PDF/Image)
  -> Auto-classify document type
  -> Select pipeline
  -> For each step in DAG:
       -> Preprocess image (deskew, contrast, denoise)
       -> Build prompt from agent YAML + blackboard context
       -> Call VLM (with cache check)
       -> Parse response, write to blackboard
       -> Postprocess markdown (normalize, dedup, clean)
       -> Score confidence (6 signals, weighted)
  -> Final merge + pipeline-level postprocessing
  -> ConversionResult
```

## Configuration

Configuration merges from multiple sources (later overrides earlier):

1. Package defaults
2. Global config (`~/.doc2md/config.yaml`)
3. Project config (`./doc2md.yaml`)
4. Environment variables (`OPENAI_API_KEY`, `DOC2MD_*`)
5. Runtime arguments

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI |
| `DOC2MD_MODEL` | Override default model |
| `DOC2MD_CACHE_DISABLED` | Set to `true` to disable caching |
| `DOC2MD_LOG_LEVEL` | Log level (`DEBUG`, `INFO`, `WARNING`) |
| `DOC2MD_MAX_WORKERS` | Concurrent workers for batch processing |

## Custom Agents & Pipelines

Drop YAML files into `./agents/` or `./pipelines/` to extend:

```yaml
# agents/my_extractor.yaml
agent:
  name: my_extractor
  version: "1.0"
  description: "Custom extraction agent"
  model:
    preferred: gpt-4.1-mini
    fallback: [gpt-4o-mini]
  input: image
  preprocessing:
    - name: enhance_contrast
      params: { factor: 1.5 }
  prompt:
    system: "Extract structured data from this document image."
    user: "Convert this image to well-formatted markdown."
  postprocessing:
    - normalize_headings
    - strip_artifacts
```

```yaml
# pipelines/my_pipeline.yaml
pipeline:
  name: my_pipeline
  version: "1.0"
  steps:
    - name: extract
      type: agent
      agent: my_extractor
    - name: cleanup
      type: code
      function: deduplicate_content
      depends_on: [extract]
  postprocessing:
    - fix_table_alignment
```

## Development

```bash
git clone https://github.com/reubingeorge/doc2md.git
cd doc2md
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT License — see [LICENSE](LICENSE) for details.

