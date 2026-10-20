"""Check the collection: print the document count and run a sample similarity query.

Usage:  python verify.py "who hit the winning six?"
"""
import sys

import chromadb

import config
from ollama_client import embed

query = " ".join(sys.argv[1:]) or "Who hit the winning six in the final over?"

client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
collection = client.get_collection(config.COLLECTION_NAME)
print(f"Documents in '{config.COLLECTION_NAME}': {collection.count()}")

result = collection.query(
    query_embeddings=embed([query], prefix=config.QUERY_PREFIX),
    n_results=3,
    include=["documents", "metadatas", "distances"],
)
print(f"\nQuery: {query}")
for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
    print(f"\n--- {meta['source']} (chunk {meta['chunk_index']}, distance {dist:.3f})")
    print(doc[:300])
