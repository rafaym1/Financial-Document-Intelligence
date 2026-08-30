from pathlib import Path

from fdi.chunking import chunk_text
from fdi.embeddings import embed_chunks
from fdi.ingestion import ingest_vdr
from fdi.vector_store import VectorStore

VDR_ROOT = "data/dev_vdr"
INDEX_DIR = "data/vector_index"

# Only index categories not already covered by a prior run -- re-running with a category
# already present in the saved store would duplicate its chunks.
CATEGORIES = ["1 - Corporate Matters"]


def main():
    documents, skipped = ingest_vdr(VDR_ROOT)
    documents = [doc for doc in documents if doc.category in CATEGORIES]

    store = VectorStore.load(INDEX_DIR) if Path(INDEX_DIR).exists() else VectorStore()
    print(f"Starting from {len(store.chunks)} already-indexed chunks")

    for doc in documents:
        chunks = chunk_text(doc.text, chunk_size=1000, overlap=100)
        embeddings = embed_chunks(chunks)
        metadatas = [{"source_path": doc.source_path, "category": doc.category}] * len(chunks)
        store.add(chunks, metadatas, embeddings)
        print(f"  [{doc.category}] {doc.source_path} -> {len(chunks)} chunks")

    store.save(INDEX_DIR)
    print(f"\n{len(store.chunks)} chunks total ({len(documents)} new documents this run) -> {INDEX_DIR}")

    if skipped:
        print(f"Skipped {len(skipped)} unsupported files")


if __name__ == "__main__":
    main()
