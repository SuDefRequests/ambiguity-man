import json
import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize persistent Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Use default embedding model (all-MiniLM-L6-v2)
emb_fn = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="abhilekh_knowledge",
    embedding_function=emb_fn
)

def ingest_file(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    batch_size = 500
    total = len(data)
    
    for i in range(0, total, batch_size):
        batch = data[i:i + batch_size]
        
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
        
        # Add batch to ChromaDB
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Ingested {min(i + batch_size, total)} / {total}")

# Ingest both datasets
ingest_file("baws_data_for_graph.json")
ingest_file("cad_data_for_graph.json")

print("\nIngestion complete! Total records:", collection.count())