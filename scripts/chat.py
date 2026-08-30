from pathlib import Path

from fdi.chatbot import answer_question, verify_answer
from fdi.vector_store import VectorStore

INDEX_DIR = "data/vector_index"


def main():
    if not Path(INDEX_DIR).exists():
        print(f"No index found at {INDEX_DIR} -- run scripts/build_index.py first.")
        return

    store = VectorStore.load(INDEX_DIR)
    print(f"Loaded {len(store.chunks)} chunks. Ask a question (or 'exit' to quit).\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        answer, results = answer_question(question, store)
        print(f"\n{answer}\n")

        sources = sorted({(r["category"], r["source_path"]) for r in results})
        print("Sources:")
        for category, source_path in sources:
            print(f"  [{category}] {source_path}")

        unverified_claims, unmatched_numbers = verify_answer(answer, results)
        if unverified_claims:
            print(f"\n{len(unverified_claims)} unverified claim(s):")
            for claim in unverified_claims:
                print(f"  - {claim.claim} ({claim.reason})")
        if unmatched_numbers:
            print(f"\n{len(unmatched_numbers)} number(s) not found in source:")
            for number in unmatched_numbers:
                print(f"  - {number}")
        print()


if __name__ == "__main__":
    main()
