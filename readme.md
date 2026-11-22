# Infrang

Infrang is a small, local-first RAG (retrieval-augmented generation) library. It ingests documents, stores them in an embedded [Qdrant](https://qdrant.tech/) database, retrieves relevant chunks with hybrid (dense + sparse) search, and generates answers strictly from the retrieved context using a model hosted on [Groq](https://groq.com/).

## Features

- **Multiple formats:** PDF, HTML, DOCX, XLSX, PPTX, Markdown and plain text
- **Hybrid search:** dense (`BAAI/bge-small-en-v1.5`) and sparse (`Qdrant/bm42-all-minilm-l6-v2-attentions`) embeddings, fused with Reciprocal Rank Fusion (RRF)
- **Local storage:** embedded Qdrant plus a SQLite source registry, with no server to run
- **Deduplication:** every document is hashed with BLAKE3, so the same content is never ingested twice
- **Crash recovery:** an interrupted ingest is detected on the next run, cleaned up and redone
- **Token-aware chunking:** chunks are sized with the embedding model's own tokenizer, with configurable overlap
- **Grounded answers:** the generative model is instructed to answer only from the provided context
- **Web ingestion:** fetch HTML or PDF pages directly from a URL, with automatic retries

## Requirements

- Python 3.12+ (the code uses the `type` statement)
- A [Groq API key](https://console.groq.com/) for answer generation (search works without one)

## Installation

```bash
pip install pdfminer.six trafilatura openpyxl python-pptx python-docx blake3 \
    qdrant-client fastembed groq requests urllib3 tokenizers
```

The embedding models are downloaded on first run and cached in `<infrang_data_dir>/models`.

## Quick start

```python
from infrang import Infrang

with Infrang(collection='my_docs', groq_api_key='YOUR_GROQ_KEY') as rag:
    # Create the collection and ingest every file in a directory
    rag.create(rag.read_path('./knowledge_base'))

    # Retrieve relevant chunks
    for hit in rag.search('What is the refund policy?', limit=5):
        print(hit['score'], hit['metadata']['source'])

    # Or search and generate an answer in one step
    result = rag.answer('What is the refund policy?')
    print(result['answer'])
    print(result['usage'])
```

### Reopening an existing collection

`create()` makes a new collection. To use one that already exists, just instantiate `Infrang` with the same `collection` and `infrang_data_dir`, then call `search()`, `answer()` or `update()`.

### Adding documents later

```python
rag.update(rag.read_path('./new_docs'))

# From a URL (HTML or PDF)
content, final_url, fmt = rag.fetch_url('https://example.com/article')
rag.update([(content, final_url, fmt)])
```

`update()` takes any iterable of `(payload: bytes, filename: str, filetype: str)` tuples, so you can feed it your own sources too.

## API overview

| Method | Description |
| --- | --- |
| `Infrang(collection, ...)` | Initialize the instance and preload the models |
| `create(sources, overwrite=False)` | Create the collection (and optionally ingest `sources`). With `overwrite=True` the existing collection is deleted first |
| `update(sources)` | Ingest additional sources, skipping duplicates and failures |
| `read_path(path)` | Lazily read all files in a directory as ingestible tuples |
| `fetch_url(url)` | Download a URL and return `(bytes, final_url, format)` |
| `search(query, limit=8)` | Hybrid search; returns a list of `{'metadata', 'score'}` |
| `generate(query, context, groq_api_key=None)` | Generate an answer from a list of context strings |
| `answer(query, groq_api_key=None, num_of_results=8)` | `search()` followed by `generate()` |
| `get_sources()` | List ingested sources (`hash`, `filename`, `open_status`) |
| `get_chunk_number(reference_id=None, source=None)` | Count chunks overall, per document hash, or per source name |
| `get_collections()` | List the collections in the data directory |
| `remove(reference_id=... \| filename=...)` | Delete a document and all its chunks |
| `delete()` | Delete the whole collection |
| `set_groq_client(api_key)` | Set or replace the Groq client |
| `uninit()` | Close the Qdrant, Groq and HTTP connections (called automatically by the `with` block) |

## Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `collection` | required | Collection name (letters, digits, `_` and `-` only) |
| `dense_model` | `BAAI/bge-small-en-v1.5` | Dense embedding model |
| `sparse_model` | `Qdrant/bm42-all-minilm-l6-v2-attentions` | Sparse embedding model |
| `generative_model` | `openai/gpt-oss-120b` | Groq model used for answers |
| `groq_api_key` | `None` | Groq API key; can be set later with `set_groq_client()` |
| `parallel` | `4` | Parallel workers for uploading to Qdrant |
| `infrang_data_dir` | `.infrang_data` | Where the Qdrant data, SQLite registry and models are stored |
| `max_payload_size` | 8 MB | Larger files are skipped |
| `max_token_chunk` | `504` | Maximum tokens per chunk |
| `overlap_token_chunk` | `16` | Token overlap between chunks (must be smaller than the chunk size) |

## Data layout

```
.infrang_data/
├── documents/
│   └── <collection>/
│       └── sources.sqlite   # registry of ingested sources (hash, filename, open_status)
├── models/                  # cached embedding and tokenizer models
└── ...                      # embedded Qdrant storage
```

## How it works

1. **Ingest:** the file is hashed (BLAKE3). Known hashes are skipped, and the source is registered as "open".
2. **Extract:** text is pulled out according to the file type.
3. **Chunk:** the text is split into overlapping token windows.
4. **Embed and store:** each chunk gets a dense and a sparse vector and is uploaded to Qdrant, and the source is marked as complete.
5. **Search:** the query is embedded both ways, and the two result lists are merged with RRF.
6. **Generate:** the top chunks are sent to the Groq model with a system prompt that restricts it to the provided context. If the answer isn't there, it replies "I don't know the answer."

## Error handling

All library errors derive from `InfrangError`:

`InfrangInitializationError`, `InfrangDatabaseError`, `InfrangCollectionError`, `InfrangDocumentError`, `InfrangSearchError`, `InfrangGenerationError`, `InfrangValidationError`

## Notes and limitations

- Files that fail to ingest are logged and skipped rather than aborting the whole batch.
- Only HTML and PDF are accepted by `fetch_url()`.
- Local Qdrant storage is single-process, so don't open the same data directory from several processes at once.
- Answer quality depends on retrieval quality. Tune `limit`/`num_of_results` and the chunk settings for your documents.