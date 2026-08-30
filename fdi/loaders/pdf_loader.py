import pdfplumber

from fdi.schema import Document, Table


def load_pdf(path: str, category: str) -> Document:
    """Extract text and tables from a PDF into a normalized Document."""
    text_parts = []
    tables: list[Table] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
            for table in page.extract_tables():
                tables.append([[cell or "" for cell in row] for row in table])

    return Document(
        source_path=path,
        category=category,
        file_type="pdf",
        text="\n\n".join(text_parts),
        tables=tables,
    )
