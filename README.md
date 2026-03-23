
# fin-chat-bot (Financial QA)

This repo ingests financial PDFs into **Qdrant** (vector DB) and lets you ask questions via a **Streamlit** UI.

## Setup

### 1) Install dependencies

This project uses **uv** (see `pyproject.toml`). From the repo root:

```powershell
cd C:\Users\admin\Desktop\gic
uv sync
```

### 2) Start Qdrant

You can run Qdrant via Docker. Qdrant should be reachable at `http://localhost:6333`.

```powershell
docker run -p 6333:6333 -p 6334:6334 -v ${PWD}\qdrant_storage:/qdrant/storage qdrant/qdrant
```

If you already have a container, start it:

```powershell
docker start <container_id>
```

### 3) Configure keys + settings

Config is loaded from:

- `src/config/model_config.yml` (Gemini + Tavily keys, model names)
- `src/config/qdrant_config.yml` (Qdrant URL + collection name)

Important fields:

- `src/config/qdrant_config.yml` → `qdrant.url` should typically be:
	- `http://localhost:6333`
- `src/config/qdrant_config.yml` → `qdrant.default_collection` (the collection your app will query)
- `src/config/model_config.yml` → `api_key.gemini` and `api_key.tavily`

## How to run the system

### Step 1 — Ingest PDFs into Qdrant

The ingestion script is:

- `src/vector_db/qdrant_update.py`

It uses `PDFPlumberParser` to extract text/tables and uploads embeddings into Qdrant.

Run:

```powershell
cd C:\Users\admin\Desktop\gic
uv run python -m src.vector_db.qdrant_update
```

Notes:

- The script currently points to `data/pdfs` in its `__main__` block.
- If a PDF can’t be opened/parsed, ingestion will print `[SKIP] ...` and continue.

### Step 2 — Ask questions (Streamlit UI)

The UI entrypoint is `app.py`.

Run:

```powershell
cd C:\Users\admin\Desktop\gic
uv run streamlit run app.py
```

Then ask questions like:

- “Show me the revenue of 3M in 2015”

The app uses `MultiStepFinancialAgent` (`src/llm_service/pipeline/reasoning_graph.py`) which:

1. decomposes the question into sub-questions,
2. runs the base graph agent per step (`src/llm_service/pipeline/graph.py`),
3. synthesizes a final answer.

The UI also shows step-by-step results (route/source/answer/error) for debugging.

## Troubleshooting

### `getaddrinfo failed` / DNS errors when querying Qdrant

This usually means `qdrant.url` is set to a hostname that Windows can’t resolve.

Fix by setting `src/config/qdrant_config.yml`:

- `qdrant.url: "http://localhost:6333"`

### No answers from Qdrant

Retrieval requires metadata filters (company + year/report_type). If metadata extraction isn’t confident, the graph may route to web search instead.

## Documentation

Please refer to this link to fully understand the system design https://docs.google.com/document/d/1mlFIJj-WAQoKYQTpwIthb6dxWfY2T9tdEQ595L0vtmQ/edit?usp=sharing

