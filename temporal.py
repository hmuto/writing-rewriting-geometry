"""Shared helpers: repository paths, extraction constants, and the official
genre/prompt labels from the released CoAuthor metadata.
"""
from pathlib import Path

import pandas as pd
import unicodedata

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data" / "coauthor-v1.0"
METADATA = REPO / "data" / "coauthor_metadata.csv"

MIN_TEXT_LENGTH = 20   # drop sentences shorter than this (characters)
N_TERTILES = 3         # session thirds: early / mid / late

_PROMPT_CODE = None


def _prompt_codes():
    """session_id -> prompt_code from the released metadata.

    The metadata covers exactly the 830 story (creative) sessions, so
    membership in it is the genre label; the remaining 617 sessions are the
    argumentative ones.
    """
    global _PROMPT_CODE
    if _PROMPT_CODE is None:
        m = pd.read_csv(METADATA)
        _PROMPT_CODE = dict(zip(m["session_id"].astype(str), m["prompt_code"]))
    return _PROMPT_CODE


def normalize_prompt(p: str) -> str:
    """Collapse whitespace and fold typographic punctuation to ASCII, so a
    prompt cannot split into two groups on a single quote character."""
    p = unicodedata.normalize("NFKC", p or "")
    for a, b in (("’", "'"), ("‘", "'"),
                 ("“", '"'), ("”", '"'), ("—", "-")):
        p = p.replace(a, b)
    return " ".join(p.split())


def session_genre_prompt(session_id: str, prompt_text: str):
    """Return (genre, prompt_key) for one session, using the official labels."""
    code = _prompt_codes().get(str(session_id))
    if code:
        return "creative", str(code)
    return "argumentative", normalize_prompt(prompt_text)[:80]
