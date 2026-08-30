import json
from pathlib import Path

from pydantic import BaseModel

from fdi.schema import Document

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")


def save_facts(document: Document, fact_type: str, facts: list[BaseModel]) -> Path:
    """Persist extracted facts for one document as JSON, organized by VDR category."""
    dest_dir = KNOWLEDGE_BASE_DIR / document.category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{Path(document.source_path).stem}.json"

    payload = {
        "source_path": document.source_path,
        "fact_type": fact_type,
        "facts": [fact.model_dump() for fact in facts],
    }
    dest_path.write_text(json.dumps(payload, indent=2))
    return dest_path


def load_category_facts(category: str) -> list[dict]:
    """Load every saved fact record for a given VDR category."""
    dest_dir = KNOWLEDGE_BASE_DIR / category
    if not dest_dir.exists():
        return []
    return [json.loads(path.read_text()) for path in dest_dir.glob("*.json")]
