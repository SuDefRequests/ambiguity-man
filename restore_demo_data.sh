#!/usr/bin/env bash
set -euo pipefail

RELEASE_URL="https://github.com/SuDefRequests/ambiguity-man/releases/download/archives_ocr_v1"

echo "==> Restoring SIH demo data..."

if [ ! -f "chroma_db/chroma.sqlite3" ]; then
    echo "==> Downloading ChromaDB snapshot..."
    curl -fL \
      "$RELEASE_URL/archive-chroma-db.tar.gz" \
      -o /tmp/archive-chroma-db.tar.gz

    tar -xzf /tmp/archive-chroma-db.tar.gz
    rm /tmp/archive-chroma-db.tar.gz
else
    echo "==> ChromaDB already exists; skipping."
fi

if [ ! -d "ocr_data" ]; then
    echo "==> Downloading OCR data snapshot..."
    curl -fL \
      "$RELEASE_URL/ocr-data.tar.gz" \
      -o /tmp/ocr-data.tar.gz

    tar -xzf /tmp/ocr-data.tar.gz
    rm /tmp/ocr-data.tar.gz
else
    echo "==> OCR data already exists; skipping."
fi

echo "==> Demo data ready."
du -sh chroma_db ocr_data
