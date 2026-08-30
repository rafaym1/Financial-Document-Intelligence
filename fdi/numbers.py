import re

UNIT_MULTIPLIERS = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def parse_scaled_number(text: str) -> float | None:
    """Parse a numeric string like '$1,324,144 thousand' or '42.7%' into its canonical float value, scaling units."""
    lower = text.lower()
    multiplier = next((factor for unit, factor in UNIT_MULTIPLIERS.items() if unit in lower), 1)
    digits = re.sub(r"[^\d.\-]", "", text)
    if not digits or digits in {"-", "."}:
        return None
    try:
        return float(digits) * multiplier
    except ValueError:
        # Extraction artifacts occasionally leave a malformed remainder (e.g. a stray trailing
        # "-"), which float() rejects -- treat as unparseable rather than crashing the caller.
        return None
