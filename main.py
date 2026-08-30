from fdi.extraction import extract_financial_highlights_chunked
from fdi.ingestion import ingest_vdr
from fdi.knowledge_base import save_facts


def main():
    documents, skipped = ingest_vdr("data/dev_vdr")

    print(f"Ingested {len(documents)} documents:")
    for doc in documents:
        print(f"  [{doc.category}] {doc.source_path} — {len(doc.text)} chars, {len(doc.tables)} tables")

    if skipped:
        print(f"\nSkipped {len(skipped)} unsupported files:")
        for path in skipped:
            print(f"  {path}")

    print("\nExtracting financial highlights from '3 - Accounts' documents:")
    for doc in documents:
        if doc.category != "3 - Accounts":
            continue
        highlights = extract_financial_highlights_chunked(doc)
        dest_path = save_facts(doc, "FinancialHighlight", highlights)
        print(f"  {doc.source_path} -> {dest_path} ({len(highlights)} facts)")


if __name__ == "__main__":
    main()
