"""Delta-replay extraction: reconstruct the FINAL document with per-character
authorship, then label sentences by who wrote them.

Each session's Quill delta stream is replayed op by op:

    retain n  -> advance the cursor by n
    insert s  -> insert s at the cursor, tagging each character with the author
                 ('H' = eventSource "user", 'A' = "api") and the event index
    delete n  -> remove n characters at the cursor

What survives to the end is the final document. It is split into sentences and
each one is labelled by who wrote its surviving characters, so a sentence is
Human only if the human actually typed (and kept) it.

Usage: python replay_extract.py

Output: results/replay/sentences.csv
  columns = session, prompt, genre, tertile, label, text, frac_h, frac_a
  labels  = Human | Accepted | Mixed | Dismissed
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import temporal as T

OUT_DIR = T.REPO / "results" / "replay"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "sentences.csv"

PURE = 0.80          # >= this fraction of one author -> that author's sentence
SENT_END = re.compile(r"(?<=[.!?])\s+")


def replay(events):
    """Return (text, authors, event_index) for the final document."""
    doc = []                                  # list of (char, author, event_idx)
    for i, e in enumerate(events):
        if e.get("eventName") == "system-initialize" and not doc:
            doc = [(c, "P", i) for c in (e.get("currentDoc") or "")]
            continue
        td = e.get("textDelta")
        if not isinstance(td, dict):
            continue
        author = "H" if e.get("eventSource") == "user" else "A"
        pos = 0
        for op in td.get("ops", []):
            if not isinstance(op, dict):
                continue
            if "retain" in op:
                pos = min(pos + int(op["retain"]), len(doc))
            elif "insert" in op:
                s = op["insert"]
                if isinstance(s, str) and s:
                    doc[pos:pos] = [(c, author, i) for c in s]
                    pos += len(s)
            elif "delete" in op:
                del doc[pos:pos + int(op["delete"])]
    if not doc:
        return "", np.empty(0, "U1"), np.empty(0, int)
    chars, authors, idx = zip(*doc)
    return "".join(chars), np.array(authors), np.array(idx)


def sentences_with_authorship(text, authors, idx, n_events):
    """Split the final document into sentences and label each by authorship."""
    out = []
    pos = 0
    for piece in SENT_END.split(text):
        start = text.find(piece, pos)
        if start < 0:
            start = pos
        end = start + len(piece)
        pos = end
        s = piece.strip()
        if len(s) < T.MIN_TEXT_LENGTH:
            continue
        a = authors[start:end]
        if len(a) == 0:
            continue
        fh, fa, fp = (a == "H").mean(), (a == "A").mean(), (a == "P").mean()
        if fp > 0.5:                       # prompt text, not written in-session
            continue
        label = ("Human" if fh >= PURE else
                 "Accepted" if fa >= PURE else "Mixed")
        written = idx[start:end][a != "P"]
        if len(written) == 0:
            continue
        tert = min(int(np.median(written) * T.N_TERTILES / max(n_events, 1)),
                   T.N_TERTILES - 1)
        out.append((tert, label, s, round(float(fh), 3), round(float(fa), 3)))
    return out


def dismissed(events, n_events):
    """Suggestions that were shown but never selected."""
    out, batch, sel, bt = [], None, None, None

    def flush():
        nonlocal batch, sel, bt
        if batch is not None:
            for c in batch:
                if c.get("index") != sel:
                    t = (c.get("trimmed") or c.get("original") or "").strip()
                    if len(t) >= T.MIN_TEXT_LENGTH:
                        out.append((bt, "Dismissed", t, 0.0, 1.0))
        batch, sel, bt = None, None, None

    for i, e in enumerate(events):
        name = e.get("eventName")
        t = min(int(i * T.N_TERTILES / max(n_events, 1)), T.N_TERTILES - 1)
        if name == "suggestion-open":
            flush()
            c = e.get("currentSuggestions") or []
            if isinstance(c, list) and c:
                batch, sel, bt = c, None, t
        elif name == "suggestion-select":
            h = e.get("currentHoverIndex")
            if isinstance(h, int) and batch:
                sel = h
        elif name == "suggestion-close":
            flush()
    flush()
    return out


def main():
    files = sorted(glob.glob(str(T.DATA_DIR / "*.jsonl")))
    rows, skipped = [], 0
    for fp in tqdm(files, desc="replay"):
        sid = Path(fp).stem
        try:
            events = [json.loads(l) for l in open(fp)]
        except Exception:
            skipped += 1
            continue
        prompt = ""
        for e in events[:3]:
            if e.get("eventName") == "system-initialize":
                prompt = (e.get("currentDoc") or "").strip()
                break
        genre, pk = T.session_genre_prompt(sid, prompt)
        n = len(events)
        text, authors, idx = replay(events)
        recs = sentences_with_authorship(text, authors, idx, n) + dismissed(events, n)
        for tert, label, s, fh, fa in recs:
            rows.append((sid, pk, genre, tert, label, s, fh, fa))

    df = pd.DataFrame(rows, columns=["session", "prompt", "genre", "tertile",
                                     "label", "text", "frac_h", "frac_a"])
    before = len(df)
    df = df.drop_duplicates(subset=["prompt", "label", "text"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)

    print(f"\nskipped files: {skipped};  rows {before} -> {len(df)} after dedup")
    print(f"sessions: {df['session'].nunique()}   prompts: {df['prompt'].nunique()}")
    print("\ncounts by genre x label:")
    print(df.groupby(["genre", "label"]).size().unstack(fill_value=0).to_string())
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
