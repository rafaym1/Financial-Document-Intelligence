from pathlib import Path

from fdi.loaders.html_loader import load_html
from fdi.memo import generate_memo

SOURCE_10K = r"data\dev_vdr\3 - Accounts\2026-02-25_10-K_plnt-20251231.htm"
OUTPUT_PATH = Path("data/output/memo.md")


def main():
    doc = load_html(SOURCE_10K, "3 - Accounts")
    business_context = doc.text[:100_000]

    memo = generate_memo(business_context)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(memo, encoding="utf-8")
    print(f"Saved memo to {OUTPUT_PATH} ({len(memo)} chars)")


if __name__ == "__main__":
    main()
