-- Abhilekh knowledge base schema
-- Postgres is the source of truth. Qdrant (vectors) and Neo4j (graph)
-- are both DERIVED indexes that can be rebuilt from this data.

CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fuzzy quote matching
CREATE EXTENSION IF NOT EXISTS unaccent;

-- ---------------------------------------------------------------
-- Documents: one row per source volume / book / debate transcript
-- ---------------------------------------------------------------
CREATE TABLE documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    doc_type        TEXT NOT NULL,        -- 'speech' | 'debate' | 'book' | 'letter'
    volume          TEXT,
    doc_date        DATE,
    language        TEXT NOT NULL DEFAULT 'en',
    source_url      TEXT,
    sha256          TEXT,
    minio_path      TEXT,
    license_note    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------
-- Speakers / people (also mirrored as :Person nodes in Neo4j)
-- ---------------------------------------------------------------
CREATE TABLE speakers (
    speaker_id      TEXT PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    aliases         TEXT[] NOT NULL DEFAULT '{}',
    role            TEXT
);

-- ---------------------------------------------------------------
-- Passages: the atomic citable unit (one speaker turn / paragraph)
-- ---------------------------------------------------------------
CREATE TABLE passages (
    passage_id      TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents(doc_id),
    speaker_id      TEXT REFERENCES speakers(speaker_id),
    parent_passage_id TEXT REFERENCES passages(passage_id),
    page_start      INT,
    page_end        INT,
    bbox            JSONB,               -- [[x1,y1,x2,y2], ...] for highlight-on-scan
    language        TEXT NOT NULL DEFAULT 'en',
    text            TEXT NOT NULL,
    ocr_confidence  REAL,
    article_refs    TEXT[] NOT NULL DEFAULT '{}',   -- regex-extracted, e.g. {'17','32'}
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_passages_doc ON passages(doc_id);
CREATE INDEX idx_passages_speaker ON passages(speaker_id);
CREATE INDEX idx_passages_article_refs ON passages USING GIN(article_refs);

-- Full-text (BM25-style) search index
ALTER TABLE passages ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', unaccent(text))) STORED;
CREATE INDEX idx_passages_tsv ON passages USING GIN(tsv);

-- Trigram index for fuzzy quote matching (Quote Authenticator)
CREATE INDEX idx_passages_text_trgm ON passages USING GIN(text gin_trgm_ops);

-- ---------------------------------------------------------------
-- Entities: articles, topics, events, acts — anything besides people
-- ---------------------------------------------------------------
CREATE TABLE entities (
    entity_id       TEXT PRIMARY KEY,
    entity_type     TEXT NOT NULL,       -- 'Article' | 'Topic' | 'Event' | 'Act'
    canonical_name  TEXT NOT NULL,
    aliases         TEXT[] NOT NULL DEFAULT '{}'
);

CREATE TABLE mentions (
    passage_id      TEXT NOT NULL REFERENCES passages(passage_id),
    entity_id       TEXT NOT NULL REFERENCES entities(entity_id),
    confidence      REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (passage_id, entity_id)
);

-- ---------------------------------------------------------------
-- Staged graph edges: LLM/regex-extracted relations awaiting review
-- before being loaded into Neo4j. This is the human-in-the-loop
-- safety valve so a bad extraction never reaches the live graph.
-- ---------------------------------------------------------------
CREATE TABLE edges_staging (
    edge_id         BIGSERIAL PRIMARY KEY,
    subj_id         TEXT NOT NULL,       -- speaker_id or entity_id
    subj_type       TEXT NOT NULL,       -- 'Person' | 'Article' | ...
    predicate       TEXT NOT NULL,       -- 'OPPOSED' | 'SUPPORTED' | 'DISCUSSES' | ...
    obj_id          TEXT NOT NULL,
    obj_type        TEXT NOT NULL,
    passage_id      TEXT NOT NULL REFERENCES passages(passage_id),
    confidence      REAL NOT NULL,
    extraction_method TEXT NOT NULL,     -- 'regex' | 'llm'
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected
    reviewed_by     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_edges_status ON edges_staging(status);
