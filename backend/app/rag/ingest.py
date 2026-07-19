import asyncio
from typing import cast
import chromadb
from chromadb.api.types import Embeddings
from app.config import CHROMA_DIR, COLLECTION_NAME, DOCUMENTS_DIR
from app.kernel_setup import azure_embedding
from app.rag.chunking import chunk_all

EMBED_BATCH_SIZE = 16

async def run_ingest() -> int:
    chunks = chunk_all(DOCUMENTS_DIR)
    if not chunks:
        raise RuntimeError(f"No documents found in {DOCUMENTS_DIR}. Please add some text files to ingest.")
    
    embedder = azure_embedding()
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = [c.text for c in chunks[i : i + EMBED_BATCH_SIZE]]
        batch_vectors = await embedder.generate_embeddings(batch)
        vectors.extend(v.tolist() for v in batch_vectors)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [c.name for c in client.list_collections()] else None
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[f"{c.source}::{c.chunk_index}" for c in chunks],
        embeddings=cast(Embeddings, vectors),
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "section": c.section, "chunk_index": c.chunk_index} for c in chunks],
    )
    return len(chunks)

def main() -> None:
    count = asyncio.run(run_ingest())
    print(f"Ingested {count} chunks into ChromaDB at {CHROMA_DIR} in collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()