# Abhilekh — Knowledge Layer (Neo4j + semantic search)

This is Kunal's half of SIH26096: Postgres (source of truth) + Qdrant
(vector search) + Neo4j (graph) + a FastAPI service that your teammate's
UI and answer-generation code call. It never talks to their ingestion
or UI code directly — only through `passages.jsonl` in and the HTTP API
out. That boundary is what lets both halves of the team build at the
same time without blocking each other.

## Why this shape (quick recap)

- **Postgres is the only source of truth.** Qdrant and Neo4j are
  rebuildable indexes derived from it — if either gets corrupted or the
  graph has a bad edge, you fix Postgres and re-run the loaders.
- **BM25 + vector search are fused with Reciprocal Rank Fusion**, not
  averaged, because their scores live on incomparable scales.
- **The graph only stores IDs, not text.** Full passage text always
  comes from Postgres, so the graph stays light and there's never two
  versions of the truth to keep in sync.
- **No LLM-extracted edge reaches Neo4j unreviewed.** Everything lands
  in `edges_staging` first; only `status='approved'` rows get loaded.
  This is your answer if a judge asks "how do you stop the AI from
  inventing history?"

## 1. Start the services

```bash
docker compose up -d
```

This brings up Postgres (with `schema.sql` auto-applied on first boot),
Qdrant, and Neo4j (browser UI at http://localhost:7474, user `neo4j`,
password `abhilekh_dev` — change both before you ever go near real data).

## 2. Install Python deps

```bash
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```

The first `SentenceTransformer` call downloads BGE-M3 (~2GB) — do this
once, ahead of the demo, not live on kiosk wifi.

## 3. Load sample data (day 1, before real OCR output exists)

```bash
python -m app.ingest sample_data/passages.jsonl
```

`sample_data/passages.jsonl` is **placeholder text**, clearly marked as
such — it is NOT real Ambedkar quotes. It exists only so you can build
and test `/search` and `/verify-quote` before the ingestion pipeline
produces real passages. Swap it for your teammate's real output the
moment it's ready — same command, same file shape (see "Contract" below).

## 4. Bring up the graph

```bash
# apply constraints once
cat app/graph/schema.cypher | docker exec -i abhilekh-neo4j cypher-shell -u neo4j -p abhilekh_dev

# extract candidate relations (needs LLM_API_BASE set in .env)
python -m app.graph.extract

# review them — approve/reject each candidate edge against its source text
python -m app.graph.loader review

# load approved edges + base nodes into Neo4j
python -m app.graph.loader
```

Article-number references don't need this step — those are extracted by
regex during `app.ingest` and stored directly on each passage row.

## 5. Run the API

```bash
uvicorn app.api:app --reload --port 8001
```

- `POST /search` — hybrid search, optionally graph-expanded
- `POST /verify-quote` — the Quote Authenticator
- `GET /graph/article/{number}` — nodes/edges for the Article Explorer UI
- `GET /health`

## The contract with the ingestion/UI side

Your teammate's parser must output one JSON object per line, matching
`sample_data/passages.jsonl`'s shape: `passage_id, doc_id, doc_title,
doc_type, volume, date, language, speaker, page_start, page_end,
ocr_confidence, bbox (optional), text`. Anything else about their
pipeline (which OCR engine, PDF vs scan, Scrapy vs requests) is entirely
their call — it's invisible to this side as long as the JSONL shape holds.

## What's stubbed vs real

- **Real:** schema, hybrid search + RRF, Quote Authenticator (fuzzy +
  semantic), graph schema, staging/review workflow, API contract.
- **Needs your input before the demo:** `LLM_API_BASE` in `.env` (point
  it at whatever local or hosted model you pick) for graph extraction —
  everything else runs with zero LLM calls.
- **Replace before the real demo:** `sample_data/passages.jsonl` with
  actual verified transcriptions.
