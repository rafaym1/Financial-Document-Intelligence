import pandas as pd

from fdi.schema import Document, Table


def load_csv(path: str, category: str) -> Document:
    """Load a CSV file as a single table into a normalized Document."""
    df = pd.read_csv(path, dtype=str).fillna("")
    table: Table = [df.columns.tolist(), *df.values.tolist()]

    return Document(
        source_path=path,
        category=category,
        file_type="csv",
        text="",
        tables=[table],
    )
