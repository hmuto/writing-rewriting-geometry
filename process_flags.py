"""Process-based flags for replayed sentences (writing ACTION, not only final characters).

`replay_extract.py` labels a sentence by who typed the characters that survive
in the final document. A sentence the writer typed over a deleted suggestion,
or retyped after reading one, can therefore count as Human. This script
replays the same delta stream and adds two process flags per sentence:

    ai_ghost   number of suggestion (api) characters that were inserted and
               later deleted at this sentence's location. Deleted characters
               leave a mark on both surviving neighbours, so the flag is
               conservative: it over-marks rather than misses.
    sugg_bi    fraction of the sentence's word bigrams that appear in any
               suggestion shown earlier in the session (suggestion-open
               events before the sentence was finished). High values indicate
               retyping or close paraphrase of a shown suggestion.

Rows are produced exactly as in replay_extract.py and deduplicated the same
way, so the output aligns row-for-row with results/replay/sentences.csv
(asserted). Dismissed rows get NaN flags.

Output: results/replay/process_flags.csv  (session, label, ai_ghost, sugg_bi)
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import replay_extract as R
import temporal as T

OUT = R.OUT_DIR / "process_flags.csv"
TOK = re.compile(r"[a-z0-9']+")


def bigrams(s):
    w = TOK.findall(s.lower())
    return set(zip(w, w[1:]))


def replay_with_ghosts(events):
    """As R.replay, plus a per-character count of deleted api characters nearby."""
    doc = []                                  # [char, author, event_idx, ghost]
    for i, e in enumerate(events):
        if e.get("eventName") == "system-initialize" and not doc:
            doc = [[c, "P", i, 0] for c in (e.get("currentDoc") or "")]
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
                    doc[pos:pos] = [[c, author, i, 0] for c in s]
                    pos += len(s)
            elif "delete" in op:
                gone = doc[pos:pos + int(op["delete"])]
                del doc[pos:pos + int(op["delete"])]
                g = sum(1 for d in gone if d[1] == "A") + sum(d[3] for d in gone)
                if g:
                    if pos < len(doc):
                        doc[pos][3] += g
                    if pos > 0:
                        doc[pos - 1][3] += g
    if not doc:
        return "", np.empty(0, "U1"), np.empty(0, int), np.empty(0, int)
    chars, authors, idx, ghost = zip(*doc)
    return "".join(chars), np.array(authors), np.array(idx), np.array(ghost)


def shown_suggestions(events):
    """(event_index, bigram set) for every suggestion shown in the session."""
    out = []
    for i, e in enumerate(events):
        if e.get("eventName") == "suggestion-open":
            for c in e.get("currentSuggestions") or []:
                if isinstance(c, dict):
                    t = (c.get("trimmed") or c.get("original") or "").strip()
                    if t:
                        out.append((i, bigrams(t)))
    return out


def sentence_flags(text, authors, idx, ghost, n_events, shown):
    """Mirror R.sentences_with_authorship, returning flags for each kept sentence."""
    out = []
    pos = 0
    for piece in R.SENT_END.split(text):
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
        if fp > 0.5:
            continue
        label = ("Human" if fh >= R.PURE else
                 "Accepted" if fa >= R.PURE else "Mixed")
        written = idx[start:end][a != "P"]
        if len(written) == 0:
            continue
        g = int(ghost[start:end].sum())
        bi = bigrams(s)
        done = written.max()
        seen = set().union(*[b for j, b in shown if j < done]) if shown else set()
        sb = len(bi & seen) / len(bi) if bi else 0.0
        out.append((label, s, g, round(sb, 3)))
    return out


def main():
    files = sorted(glob.glob(str(T.DATA_DIR / "*.jsonl")))
    rows = []
    for fp in tqdm(files, desc="process flags"):
        sid = Path(fp).stem
        try:
            events = [json.loads(l) for l in open(fp)]
        except Exception:
            continue
        prompt = ""
        for e in events[:3]:
            if e.get("eventName") == "system-initialize":
                prompt = (e.get("currentDoc") or "").strip()
                break
        genre, pk = T.session_genre_prompt(sid, prompt)
        n = len(events)
        text, authors, idx, ghost = replay_with_ghosts(events)
        for label, s, g, sb in sentence_flags(text, authors, idx, ghost, n,
                                              shown_suggestions(events)):
            rows.append((sid, pk, label, s, g, sb))
        for _, label, s, _, _ in R.dismissed(events, n):
            rows.append((sid, pk, label, s, np.nan, np.nan))

    df = pd.DataFrame(rows, columns=["session", "prompt", "label", "text",
                                     "ai_ghost", "sugg_bi"])
    df = df.drop_duplicates(subset=["prompt", "label", "text"]).reset_index(drop=True)

    base = pd.read_csv(R.OUT)
    assert len(base) == len(df), (len(base), len(df))
    assert (base["text"].values == df["text"].values).all()
    assert (base["label"].values == df["label"].values).all()
    df[["session", "label", "ai_ghost", "sugg_bi"]].to_csv(OUT, index=False)

    h = df[df["label"] == "Human"]
    print(f"Human sentences: {len(h)}")
    print(f"  with deleted-suggestion marks (ai_ghost>0): {(h['ai_ghost'] > 0).mean():.1%}")
    for t in (0.3, 0.5, 0.7):
        print(f"  sugg_bi >= {t}: {(h['sugg_bi'] >= t).mean():.1%}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
