import chromadb
from chromadb.utils import embedding_functions
import argparse
import sys

def merge_databases(primary_path: str, teammate_path: str):
    print(f"Connecting to your primary DB at {primary_path}...")
    primary_client = chromadb.PersistentClient(path=primary_path)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    primary_collection = primary_client.get_or_create_collection(name="abhilekh_knowledge", embedding_function=emb_fn)

    print(f"Connecting to teammate's DB at {teammate_path}...")
    teammate_client = chromadb.PersistentClient(path=teammate_path)
    teammate_collection = teammate_client.get_collection(name="abhilekh_knowledge", embedding_function=emb_fn)

    # Fetch all records from the teammate's database
    teammate_data = teammate_collection.get(include=["documents", "metadatas", "embeddings"])
    
    total_records = len(teammate_data['ids'])
    if total_records == 0:
        print("Teammate's database is empty. Nothing to merge.")
        sys.exit(0)

    print(f"Found {total_records} records to merge. Syncing...")

    # Upsert in batches to avoid memory crashes
    batch_size = 500
    for i in range(0, total_records, batch_size):
        primary_collection.upsert(
            ids=teammate_data['ids'][i:i+batch_size],
            embeddings=teammate_data['embeddings'][i:i+batch_size],
            documents=teammate_data['documents'][i:i+batch_size],
            metadatas=teammate_data['metadatas'][i:i+batch_size]
        )
    print(f"Sync complete! Your DB now has {primary_collection.count()} total records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", default="./chroma_db")
    parser.add_argument("--teammate", required=True)
    args = parser.parse_args()
    merge_databases(args.primary, args.teammate)