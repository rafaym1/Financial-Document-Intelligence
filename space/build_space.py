"""Assemble a self-contained, deployable bundle for the Hugging Face Space.

HF Spaces need a flat repo with app.py at the root and the packages it imports alongside it.
Rather than hand-maintaining a second copy of fdi/, this script copies the current source of
truth (fdi/, the pre-built vector index, and the space/ config files) into space/dist/ -- a
disposable build artifact, not something to edit directly or commit.

Run from the "Financial Document Intelligence" directory: python space/build_space.py
"""

import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = ROOT / "space"
DIST_DIR = SPACE_DIR / "dist"

FDI_MODULES = [
    "vector_store.py",
    "embeddings.py",
    "reranker.py",
    "verification.py",
    "numbers.py",
    "rate_limit.py",
    "reports.py",
    "knowledge_base.py",
    "schema.py",
    "__init__.py",
]

# Source VDR filings bundled so visitors can open/download and verify the system's work directly.
SOURCE_DOCUMENTS = [
    "1 - Corporate Matters/2025-03-26_DEF_14A_plnt-20250326.htm",
    "1 - Corporate Matters/2026-03-25_DEF_14A_plnt-20260325.htm",
    "3 - Accounts/2025-02-25_10-K_plnt-20241231.htm",
    "3 - Accounts/2026-02-25_10-K_plnt-20251231.htm",
]


def main():
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)

    fdi_dist = DIST_DIR / "fdi"
    fdi_dist.mkdir()
    for name in FDI_MODULES:
        shutil.copy2(ROOT / "fdi" / name, fdi_dist / name)

    index_dist = DIST_DIR / "data" / "vector_index"
    index_dist.mkdir(parents=True)
    for path in (ROOT / "data" / "vector_index").iterdir():
        if path.name == "embeddings.npy":
            # Stored as float64, far more precision than cosine similarity needs -- float16
            # keeps retrieval quality (confirmed: identical top-10 rankings) while cutting the
            # file to a quarter of its size, which matters for browser-based upload limits.
            embeddings = np.load(path).astype(np.float16)
            np.save(index_dist / path.name, embeddings)
        else:
            shutil.copy2(path, index_dist / path.name)

    # Extracted facts the Reports tab charts, and the pre-generated memo the Memo tab reveals --
    # both real output from the real pipeline, just not regenerated live on every visitor click.
    kb_dist = DIST_DIR / "data" / "knowledge_base" / "3 - Accounts"
    kb_dist.mkdir(parents=True)
    for path in (ROOT / "data" / "knowledge_base" / "3 - Accounts").iterdir():
        shutil.copy2(path, kb_dist / path.name)

    output_dist = DIST_DIR / "data" / "output"
    output_dist.mkdir(parents=True)
    shutil.copy2(ROOT / "data" / "output" / "memo.md", output_dist / "memo.md")

    vdr_dist = DIST_DIR / "data" / "dev_vdr"
    for rel_path in SOURCE_DOCUMENTS:
        dest = vdr_dist / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "data" / "dev_vdr" / rel_path, dest)

    for name in ("app.py", "llm.py", "requirements.txt", "README.md"):
        shutil.copy2(SPACE_DIR / name, DIST_DIR / name)

    print(f"Bundle assembled at {DIST_DIR}")
    print("Next: cd into it, git init (or add as a remote), and push to your Hugging Face Space.")


if __name__ == "__main__":
    main()
