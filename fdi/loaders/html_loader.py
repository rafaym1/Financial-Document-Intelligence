import pandas as pd
from lxml import html as lxml_html

from fdi.schema import Document, Table


def load_html(path: str, category: str) -> Document:
    """Extract text and tables from an HTML filing (e.g. SEC EDGAR) into a normalized Document."""
    tree = lxml_html.parse(path)
    text = tree.getroot().text_content()

    try:
        dataframes = pd.read_html(path)
    except ValueError:
        dataframes = []

    tables: list[Table] = []
    for df in dataframes:
        header = [str(column) for column in df.columns]
        rows = [
            ["" if pd.isna(cell) else str(cell) for cell in row]
            for row in df.itertuples(index=False, name=None)
        ]
        tables.append([header, *rows])

    return Document(
        source_path=path,
        category=category,
        file_type="html",
        text=text,
        tables=tables,
    )
