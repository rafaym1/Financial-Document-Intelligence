import re

from fdi.numbers import parse_scaled_number

NUMBER_PATTERN = re.compile(r"\$?-?\d[\d,]*\.?\d*(?:\s?(?:thousand|million|billion))?%?", re.IGNORECASE)

# Financial tables commonly state their unit once in a header ("$ in thousands") rather than
# per-figure, so a bare source-table number is genuinely ambiguous about its own scale.
IMPLICIT_SCALE_MULTIPLIERS = (1, 1_000, 1_000_000)


def _parse_numeric_claim(token: str, expand_implicit_scale: bool = False) -> list[tuple[float, bool]] | None:
    """Parse a matched token into (canonical_value, is_percent) pairs, scaling million/billion/thousand to a common base.

    When expand_implicit_scale is set, a bare number with no explicit unit is returned at every plausible
    implicit scale instead of just its literal value -- see IMPLICIT_SCALE_MULTIPLIERS.
    """
    is_percent = token.rstrip().endswith("%")
    digits_only = re.sub(r"[^\d.\-]", "", token)
    has_explicit_unit = any(unit in token.lower() for unit in ("thousand", "million", "billion"))
    # A bare single digit with no currency/unit/percent is almost always structural (list numbering, "3 pillars").
    if not is_percent and not has_explicit_unit:
        if len(digits_only.replace(".", "").replace("-", "")) <= 1:
            return None
    value = parse_scaled_number(token)
    if value is None:
        return None
    if expand_implicit_scale and not is_percent and not has_explicit_unit:
        return [(value * multiplier, is_percent) for multiplier in IMPLICIT_SCALE_MULTIPLIERS]
    return [(value, is_percent)]


def find_numeric_claims(text: str, expand_implicit_scale: bool = False) -> set[tuple[float, bool]]:
    """Extract every numeric claim (dollar amount, percentage, plain number) from text as (value, is_percent) pairs."""
    claims: set[tuple[float, bool]] = set()
    for token in NUMBER_PATTERN.findall(text):
        parsed = _parse_numeric_claim(token, expand_implicit_scale)
        if parsed is not None:
            claims.update(parsed)
    return claims


def verify_numbers_in_context(
    section_text: str, context: str, rel_tol: float = 0.01, percent_tol: float = 0.15
) -> list[str]:
    """Flag every numeric claim in section_text with no approximate match in context, after unit scaling.

    A small tolerance accounts for legitimate rounding differences (e.g. a filing's "$1,324,144 thousand"
    vs. prose's "$1,324.1 million" -- the same fact, rounded for readability). The context side also expands
    bare numbers to every plausible implicit scale, to account for filings that state units once per table
    instead of per-figure.
    """
    claimed = find_numeric_claims(section_text)
    available = find_numeric_claims(context, expand_implicit_scale=True)

    unmatched = []
    for value, is_percent in claimed:
        tolerance = percent_tol if is_percent else abs(value) * rel_tol
        match_found = any(
            avail_is_percent == is_percent and abs(avail_value - value) <= tolerance
            for avail_value, avail_is_percent in available
        )
        if not match_found:
            unmatched.append(f"{value}{'%' if is_percent else ''}")
    return sorted(unmatched)
