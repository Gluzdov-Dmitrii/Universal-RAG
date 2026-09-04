"""Small, dependency-free normalization helpers for the sanitizer MVP."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)?", flags=re.UNICODE)

# These are deliberately conservative and cover the common Russian surname
# inflections needed by the current Russian-language policy. This is not intended to replace a full
# morphological analyzer or an entity linker.
_SURNAME_HINTS = (
    "ов",
    "ев",
    "ёв",
    "ин",
    "ын",
    "ский",
    "цкий",
    "ской",
    "цкой",
    "ова",
    "ева",
    "ёва",
    "ина",
    "ына",
    "енко",
    "чук",
    "ук",
    "юк",
    "ян",
    "дзе",
    "ых",
    "их",
)

_NOMINATIVE_SUFFIXES = (
    # -ов/-ев/-ёв surnames and their common case endings
    ("евыми", "ев"),
    ("ёвыми", "ёв"),
    ("овыми", "ов"),
    ("евых", "ев"),
    ("ёвых", "ёв"),
    ("овых", "ов"),
    ("евы", "ев"),
    ("ёвы", "ёв"),
    ("овы", "ов"),
    ("евому", "ев"),
    ("ёвому", "ёв"),
    ("овому", "ов"),
    ("овами", "ов"),
    ("евого", "ев"),
    ("ёвого", "ёв"),
    ("ового", "ов"),
    ("евым", "ев"),
    ("ёвым", "ёв"),
    ("овым", "ов"),
    ("евую", "ев"),
    ("ёвую", "ёв"),
    ("овую", "ов"),
    ("евой", "ев"),
    ("ёвой", "ёв"),
    ("овой", "ов"),
    ("еву", "ев"),
    ("ёву", "ёв"),
    ("ову", "ов"),
    ("еве", "ев"),
    ("ёве", "ёв"),
    ("ове", "ов"),
    ("евом", "ев"),
    ("ёвом", "ёв"),
    ("овом", "ов"),
    ("ева", "ев"),
    ("ёва", "ёв"),
    ("ова", "ов"),
    # -ин/-ын surnames
    ("иными", "ин"),
    ("ыными", "ын"),
    ("иному", "ин"),
    ("ыному", "ын"),
    ("иного", "ин"),
    ("ыного", "ын"),
    ("иным", "ин"),
    ("ыным", "ын"),
    ("иную", "ин"),
    ("ыную", "ын"),
    ("иной", "ин"),
    ("ыной", "ын"),
    ("ину", "ин"),
    ("ыну", "ын"),
    ("ине", "ин"),
    ("ыне", "ын"),
    ("ином", "ин"),
    ("ыном", "ын"),
    ("ина", "ин"),
    ("ына", "ын"),
    # adjective-like surnames such as Достоевский
    ("скими", "ский"),
    ("цкими", "цкий"),
    ("скому", "ский"),
    ("цкому", "цкий"),
    ("ского", "ский"),
    ("цкого", "цкий"),
    ("ским", "ский"),
    ("цким", "цкий"),
    ("скую", "ский"),
    ("цкую", "цкий"),
    ("ской", "ский"),
    ("цкой", "цкий"),
    ("ском", "ский"),
    ("цком", "цкий"),
    ("ская", "ский"),
    ("цкая", "цкий"),
    ("ские", "ский"),
    ("цкие", "цкий"),
)


def _restore_case(source: str, normalized: str) -> str:
    if source.isupper():
        return normalized.upper()
    if source[:1].isupper():
        return normalized[:1].upper() + normalized[1:]
    return normalized


def normalize_surname(surname: str) -> str:
    """Return a practical nominative-style form for a Russian surname."""

    value = surname.strip()
    folded = value.casefold()
    for suffix, replacement in _NOMINATIVE_SUFFIXES:
        if folded.endswith(suffix) and len(folded) > len(suffix) + 1:
            normalized = folded[: -len(suffix)] + replacement
            return _restore_case(value, normalized)
    return value


def _looks_like_surname(token: str) -> bool:
    folded = token.casefold()
    return len(folded) >= 4 and (
        folded.endswith(_SURNAME_HINTS)
        or any(folded.endswith(suffix) for suffix, _ in _NOMINATIVE_SUFFIXES)
    )


def canonical_person_value(value: str) -> str:
    """Extract and normalize the surname from a PER span.

    The current policy intentionally uses surname-level identity. Full names,
    initials, and inflected forms therefore share one canonical marker. A
    later identity layer can preserve and resolve the other name components.
    """

    tokens = _WORD_RE.findall(value)
    if not tokens:
        return value.strip()
    candidates = [token for token in tokens if _looks_like_surname(token)]
    surname = candidates[0] if candidates else next(
        (token for token in tokens if len(token) > 1), tokens[0]
    )
    return normalize_surname(surname)


def canonical_marker_value(label: str, value: str) -> str:
    if label.upper() in {"PER", "PERSON", "PERS"}:
        return canonical_person_value(value)
    return value


def marker_display_value(label: str, value: str) -> str:
    """Choose the value written back for a newly created marker.

    Bare surnames are returned in the policy's canonical form. For a full PER
    span we keep the original surface text for now; preserving and inflecting
    given names/patronymics belongs to the later FIO layer.
    """

    if label.upper() in {"PER", "PERSON", "PERS"}:
        tokens = _WORD_RE.findall(value)
        if len(tokens) == 1:
            return normalize_surname(tokens[0])
    return value


def marker_identity_key(label: str, value: str) -> tuple[str, str]:
    canonical = canonical_marker_value(label, value)
    # PER identity is case-insensitive; other marker types retain their
    # existing exact-value behavior.
    if label.upper() in {"PER", "PERSON", "PERS"}:
        canonical = canonical.casefold()
    return label.upper(), canonical
