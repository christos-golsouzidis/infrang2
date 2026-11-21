# Infrang (INFormation Retrieval and ANswer Generation)

## Overview

Infrang turns a folder of documents (or a handful of URLs) into a searchable vector knowledge base that lives entirely on your disk — then answers questions about it with an LLM. No servers, no cloud vector DB, no data leaving your machine except the final generation call.

---

## Features

- **Local-first storage** — embedded Qdrant (on-disk) + SQLite, all under `.infrang_data/`.
- **Hybrid retrieval** — dense (`BAAI/bge-small-en-v1.5`) + sparse (`Qdrant/bm42-...`) vectors fused with Reciprocal Rank Fusion (RRF).
- **Broad format support** — PDF, HTML, DOCX, XLSX, PPTX, Markdown, plain text, plus remote URLs.
- **Content-addressed deduplication** — BLAKE3 payload hashes mean re-ingesting the same file is a no-op.
- **Crash-safe ingestion** — an `open_status` flag in SQLite detects interrupted ingests and cleans up before retrying.
- **Token-aware chunking** — chunks are cut on tokenizer offsets (default 504 tokens, 16 overlap), never mid-token.
- **Parallel upserts** — configurable parallel workers for Qdrant writes.
- **Optional generation** — Groq is only touched for `generate()` / `answer()`; ingestion and search work fully offline.
- **Typed error hierarchy** — every failure mode maps to a specific exception subclass.
- **Context-manager friendly** — `with Infrang(...) as kb:` closes Qdrant, Groq and the HTTP session for you.

---

## How it works

```
                 ┌────────────┐
  files / URLs ─▶│  extract   │  pdfminer · trafilatura · python-docx
                 └─────┬──────┘  openpyxl · python-pptx · utf-8
                       ▼
                 ┌────────────┐
                 │   chunk    │  tokenizer offsets, 504 tok / 16 overlap
                 └─────┬──────┘
                       ▼
          ┌────────────────────────┐
          │  embed (dense+sparse)  │  fastembed models, cached locally
          └───────────┬────────────┘
                      ▼
          ┌────────────────────────┐        ┌───────────────┐
          │   Qdrant (on disk)     │◀──────▶│ SQLite sources│
          │   RRF hybrid search    │        │ hash/filename │
          └───────────┬────────────┘        └───────────────┘
                      ▼
                 ┌────────────┐
   question ────▶│  Groq LLM  │────▶ grounded answer + usage stats
                 └────────────┘
```

---

## Requirements

- **Python 3.12+** (the module uses the PEP 695 `type` alias statement)
- ~500 MB of disk for the default embedding models (downloaded on first init to `.infrang_data/models/`)
- A [Groq API key](https://console.groq.com/) — only needed for `generate()` / `answer()`

---

## Installation

```bash
pip install qdrant-client fastembed tokenizers groq requests \
            pdfminer.six trafilatura openpyxl python-pptx python-docx blake3
```

Or, from a checkout of this repository:

```bash
pip install -r requirements.txt
```

---

## Quick start

```python
from infrang import Infrang

kb = Infrang(
    collection="handbook",
    groq_api_key="gsk_...",          # optional; can be set later
    infrang_data_dir=".infrang_data",
)

# 1. Create the collection (required once per collection name)
kb.create(kb.read_path("./company_handbook"))

# 2. Hybrid search — returns metadata + RRF score
for hit in kb.search("parental leave policy", limit=5):
    print(hit["score"], hit["metadata"]["source"])

# 3. Grounded answer (search + generation in one call)
result = kb.answer("How many days of parental leave do I get?")
print(result["answer"])
print(result["usage"]["total_tokens"])

kb.uninit()
```

The same thing, safely scoped:

```python
with Infrang(collection="handbook") as kb:
    kb.create(kb.read_path("./company_handbook"), overwrite=True)
    print(kb.answer("How many days of parental leave do I get?", groq_api_key="gsk_...")["answer"])
```

---

## Usage

### Ingest a directory

```python
kb.create(kb.read_path("./docs"))          # (bytes, filename, extension) tuples
```

`read_path` yields *every* regular file in the directory (non-recursive) and rejects unsupported extensions at conversion time. Files that are empty or larger than `max_payload_size` (8 MB default) are skipped with a logged warning instead of aborting the run.

### Ingest URLs

```python
content, final_url, fmt = kb.fetch_url("https://example.com/report.pdf")
kb.update([(content, final_url, fmt)])
```

`fetch_url` only accepts responses whose `Content-Type` is HTML or PDF; anything else raises `InfrangDocumentError`. Requests are retried 3× with backoff on 429/5xx.

### Add documents incrementally

```python
kb.update([
    (b"# Notes\nsome text", "notes.md", "md"),
    (open("spec.docx", "rb").read(), "spec.docx", "docx"),
])
```

`update()` skips failures per-file and keeps going — check the logs for anything that didn't make it in.

### Search only (no LLM)

```python
hits = kb.search("vector quantization", limit=10)
contexts = [h["metadata"]["text"] for h in hits]
```

### Generate with your own context

```python
resp = kb.generate(
    query="Summarize the abstract",
    context=["The paper proposes ...", "Experiments show ..."],
    groq_api_key="gsk_...",
)
```

`generate()` is strict by design: the system prompt forbids outside knowledge and instructs the model to reply *"I don't know the answer."* when the context is insufficient. An empty context short-circuits to `{"answer": "Empty context found.", "usage": {...zeros...}}`.

### Inspect and maintain

```python
kb.get_collections()                     # list of collection names on disk
kb.get_sources()                         # [{'hash', 'filename', 'open_status'}, ...]
kb.get_chunk_number()                    # total chunks in this collection
kb.get_chunk_number(source="spec.docx")  # chunks for one document
kb.remove(filename="spec.docx")          # or reference_id=<blake3 hash>
kb.delete()                              # drop the whole collection
```

### Empty collection

```python
kb.create(None)      # set up vectors + SQLite without ingesting anything
kb.update(...)       # add later
```

---

## API reference

### `Infrang(...)`

| Parameter | Default | Description |
|---|---|---|
| `collection` | *(required)* | Collection name. Only `[A-Za-z0-9_-]`, optional leading `.` — no spaces or slashes. |
| `dense_model` | `BAAI/bge-small-en-v1.5` | FastEmbed dense model. Also supplies the tokenizer. |
| `sparse_model` | `Qdrant/bm42-all-minilm-l6-v2-attentions` | FastEmbed sparse model (IDF modifier). |
| `generative_model` | `openai/gpt-oss-120b` | Groq model ID used by `generate()` / `answer()`. |
| `groq_api_key` | `None` | If omitted, a warning is logged; set later via `set_groq_client()`. |
| `parallel` | `4` | Parallel workers for `upload_collection`. Must be a positive int. |
| `infrang_data_dir` | `.infrang_data` | Root for Qdrant, SQLite and the model cache. |
| `max_payload_size` | `8 * 1024 * 1024` | Files above this are skipped during ingestion. |
| `max_token_chunk` | `504` | Maximum tokens per chunk. |
| `overlap_token_chunk` | `16` | Token overlap between chunks. Must be `< max_token_chunk`. |

Initialization creates the data directory, opens the embedded Qdrant client, verifies SQLite, builds a retrying HTTP session, and **preloads/warms all three models** (dense, sparse, tokenizer) so the first query isn't slow.

### Methods

| Method | Returns | Notes |
|---|---|---|
| `create(sources, overwrite=False)` | `None` | Creates the Qdrant collection + SQLite table. `overwrite=True` deletes the existing collection first. Required before any ingest or search. |
| `update(sources)` | `None` | Ingest an iterable of `(bytes, filename, format)`. Per-file failures are logged, not raised. |
| `read_path(path)` | iterable of tuples | Lazy file reader; raises if the path is missing, not a directory, or empty. |
| `fetch_url(url)` | `(bytes, str, FileFormat)` | HTML/PDF only; retries on transient errors. |
| `search(query, limit=8)` | `list[dict]` | Dense+sparse prefetch with RRF fusion. Each item: `{'metadata': {...}, 'score': float}`. |
| `generate(query, context, groq_api_key=None)` | `dict` | `{'answer': str, 'usage': {...}}`. Requires a Groq client or key. |
| `answer(query, groq_api_key=None, num_of_results=8)` | `dict` | `search()` → `generate()` in one call. |
| `set_groq_client(api_key)` | `None` | Initializes the Groq client after construction. |
| `get_sources()` | `list[dict]` | Rows from the `sources` table. |
| `get_collections()` | `list[str]` | Directories under `documents/`. |
| `get_chunk_number(reference_id=None, source=None)` | `int` | Total, per-document, or per-filename chunk counts. |
| `remove(reference_id=..., filename=...)` | `None` | Deletes vectors *and* the SQLite row. Exactly one argument required. |
| `delete()` | `None` | Removes the collection directory and the Qdrant collection. |
| `uninit()` | `None` | Closes Qdrant, Groq, and the HTTP session. Called automatically on `__exit__`. |

### Exceptions

```
InfrangError
├── InfrangInitializationError   # Qdrant/SQLite/model/HTTP setup failures
├── InfrangDatabaseError         # SQLite + Qdrant write/delete failures
├── InfrangCollectionError       # missing collection during count/queries
├── InfrangDocumentError         # extraction failures, unsupported types, empty text
├── InfrangSearchError           # (reserved)
├── InfrangGenerationError       # missing Groq client / key
└── InfrangValidationError       # bad collection name, bad `parallel`, etc.
```

---

## On-disk layout

```
.infrang_data/
├── models/                          # FastEmbed model cache (shared across collections)
└── documents/
    └── <collection>/
        ├── sources.sqlite           # hash · filename · open_status
        └── ...                      # embedded Qdrant segment files
```

- `hash` is the BLAKE3 digest of the raw payload — the document's `reference_id` in every chunk.
- `open_status = True` means an ingest started; `False` means it completed. A `True` row found on re-ingest triggers `remove()` + retry, so partial ingests never linger.

---

## Things worth knowing

- **One process at a time.** Embedded Qdrant takes a file lock on the data directory. Don't point two running processes at the same `.infrang_data`; run separate `infrang_data_dir`s or a Qdrant server instead.
- **`create()` is mandatory.** Calling `update()`, `search()` or `get_chunk_number()` before `create()` fails — usually as `InfrangCollectionError` or `InfrangDatabaseError`.
- **First run is slow.** Models are downloaded and warmed during `__init__`. Subsequent inits use the cache but still warm up (~1 s).
- **Dedup is content-based, not path-based.** Renaming a file doesn't re-ingest it; editing one byte does.
- **Collection names are directory names.** They're validated against `^\.?[A-Za-z0-9_\-]+$` to prevent path traversal.
- **HTML extraction is best-effort.** `trafilatura.extract()` returns `None` for pages it can't parse; that surfaces as an "empty text" `InfrangDocumentError`.
- **Groq only sees retrieved chunks.** Nothing else from your corpus is sent, and only when you call `generate()` / `answer()`.

---

## Roadmap ideas

- Recursive directory ingestion and glob filters
- Server-mode Qdrant support for multi-process use
- Pluggable LLM backends behind the generation interface
- Cross-encoder reranking stage
- `pyproject.toml` + published package

---

## License

No license file is included in this repository yet. Add one (MIT/Apache-2.0 are common choices) before distributing.
