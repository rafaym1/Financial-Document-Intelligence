from pydantic import BaseModel

Table = list[list[str]]


class Document(BaseModel):
    """The common shape every format-specific loader normalizes its file into."""

    source_path: str
    category: str
    file_type: str
    text: str
    tables: list[Table] = []
