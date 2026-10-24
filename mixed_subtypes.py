"""Split Mixed (rewriting) sentences by how human and suggestion text are arranged.

h(s) is a share of surviving characters, not a description of the editing
process. A Mixed sentence can arise in different ways:

    completion  human text, then suggestion text   (writer starts, model finishes)
    append      suggestion text, then human text   (writer extends a suggestion)
    interleaved three or more alternating runs      (edits inside a suggestion)

We replay each session exactly as replay_extract.py does, read the author
sequence of every surviving Mixed sentence, collapse it into runs, and ignore
runs that contain only whitespace or punctuation. The result is joined to
results/replay/sentences.csv on (session, text).

Output: results/replay/mixed_subtypes.csv
"""
import glob
import json
import re
from pathlib import Path

import pandas as pd

import replay_extract as RX
import temporal as T

OUT = RX.OUT_DIR / "mixed_subtypes.csv"
WORDISH = re.compile(r"\w")


def runs(text, authors):
    out = []
    for c, a in zip(text, authors):
        if a == "P":
            continue
        if out and out[-1][0] == a:
            out[-1][1].append(c)
        else:
            out.append([a, [c]])
    # drop runs that carry no word characters, then re-merge neighbours
    kept = []
    for a, cs in out:
        if not WORDISH.search("".join(cs)):
            continue
        if kept and kept[-1] == a:
            continue
        kept.append(a)
    return "".join(kept)


def subtype(pattern):
    if pattern == "HA":
        return "completion"
    if pattern == "AH":
        return "append"
    if len(pattern) >= 3:
        return "interleaved"
    return "other"


def main():
    rows = []
    for fp in sorted(glob.glob(str(T.DATA_DIR / "*.jsonl"))):
        sid = Path(fp).stem
        events = [json.loads(l) for l in open(fp)]
        text, authors, _ = RX.replay(events)
        pos = 0
        for piece in RX.SENT_END.split(text):
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
            fh, fa, fp_ = (a == "H").mean(), (a == "A").mean(), (a == "P").mean()
            if fp_ > 0.5 or fh >= RX.PURE or fa >= RX.PURE:
                continue
            pat = runs(text[start:end], a)
            rows.append((sid, s, pat, subtype(pat)))
    d = pd.DataFrame(rows, columns=["session", "text", "pattern", "subtype"])
    d = d.drop_duplicates(["session", "text"])
    sent = pd.read_csv(RX.OUT)
    sent["row"] = sent.index
    mixed = sent[sent.label == "Mixed"][["row", "session", "prompt", "genre", "text", "frac_h"]]
    m = mixed.merge(d, on=["session", "text"], how="left")
    m.to_csv(OUT, index=False)
    print(f"Mixed rows {len(mixed)}, matched {m['subtype'].notna().sum()}")
    print(m["subtype"].value_counts(dropna=False).to_string())
    print(m.groupby("genre")["subtype"].value_counts(normalize=True).round(3).to_string())
    print(m.groupby("subtype")["frac_h"].median().round(3).to_string())


if __name__ == "__main__":
    main()
