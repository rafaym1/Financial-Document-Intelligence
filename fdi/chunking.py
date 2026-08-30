def chunk_text(text: str, chunk_size: int = 50_000, overlap: int = 5_000) -> list[str]:
    """Split text into overlapping character windows, small enough that no fact is 'lost in the middle' of a chunk."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
