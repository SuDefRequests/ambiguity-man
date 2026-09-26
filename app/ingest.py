import json
import chromadb
from chromadb.utils import embedding_functions
import sqlite3
import hashlib
import os

# 1. Initialize Manifest DB to track processed files
conn = sqlite3.connect("manifest.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS processed_files (
        filepath TEXT PRIMARY KEY,
        file_hash TEXT
    )
""")
conn.commit()

# 2. Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(
    name="abhilekh_knowledge",
    embedding_function=emb_fn
)

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def ingest_file(filepath):
    # Check if file was already processed and hasn't changed
    current_hash = get_file_hash(filepath)
    cursor.execute("SELECT file_hash FROM processed_files WHERE filepath=?", (filepath,))
    row = cursor.fetchone()
    if row and row[0] == current_hash:
        print(f"Skipping {filepath} - no changes detected.")
        return

    print(f"Loading and processing {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    batch_size = 500
    total = len(data)
    
    for i in range(0, total, batch_size):
        batch = data[i:i + batch_size]
        
        # Using the exact chunk_id from the JSON ensures no duplicates
        ids = [item["chunk_id"] for item in batch]
        documents = [item["text"] for item in batch]
        metadatas = [
            {
                "archive_type": str(item.get("archive_type", "")),
                "volume": int(item.get("volume", 0)),
                "page": int(item.get("page", 0)),
                "source": str(item.get("source", "")),
                "title": str(item.get("title", ""))
            }
            for item in batch
        ]
        
        # upsert() overwrites if the ID exists, inserts if it's new
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Ingested {min(i + batch_size, total)} / {total}")

    # Mark as processed in manifest
    cursor.execute("REPLACE INTO processed_files (filepath, file_hash) VALUES (?, ?)", (filepath, current_hash))
    conn.commit()

# Run the smart ingestion
ingest_file("baws_data_for_graph.json")
ingest_file("cad_data_for_graph.json")

print("\nIngestion complete! Total records:", collection.count())