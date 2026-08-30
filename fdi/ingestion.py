from pathlib import Path

from fdi.loaders.csv_loader import load_csv
from fdi.loaders.docx_loader import load_docx
from fdi.loaders.html_loader import load_html
from fdi.loaders.pdf_loader import load_pdf
from fdi.loaders.xlsx_loader import load_xlsx
from fdi.schema import Document

LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".xlsx": load_xlsx,
    ".csv": load_csv,
    ".htm": load_html,
    ".html": load_html,
}


def ingest_vdr(root: str) -> tuple[list[Document], list[str]]:
    """Walk a VDR folder tree and load every supported file into a Document, tagged with its top-level category folder.

    Returns (documents, skipped_paths) so files in unsupported formats aren't silently lost.
    """
    documents = []
    skipped = []
    root_path = Path(root)
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            skipped.append(str(path))
            continue
        category = path.relative_to(root_path).parts[0]
        documents.append(loader(str(path), category))
    return documents, skipped
