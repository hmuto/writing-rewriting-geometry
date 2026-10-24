"""Sidecar file: menu (suggestion-batch) id for every row of sentences.csv.

Five suggestions arrive from one context, so the Dismissed
pool carries within-menu redundancy that the session-level cap does not
remove. To cap the reference at one sentence per menu, each Dismissed row
needs its menu id. This script replays the extraction of replay_extract.py
verbatim, carrying the suggestion-open event index as the menu id, applies
the identical dedup, verifies row-for-row agreement with the existing
sentences.csv, and writes results/replay/menu_ids.csv (one column, aligned by
row). Non-Dismissed rows get an empty id.

Usage: python menu_ids.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import temporal as T
from replay_extract import replay, sentences_with_authorship

REPO = Path(__file__).resolve().parent
OUT = REPO / "results" / "replay" / "menu_ids.csv"


def dismissed_with_menu(sid, events, n_events):
    out, batch, sel, bt, bid = [], None, None, None, None

    def flush():
        nonlocal batch, sel, bt, bid
        if batch is not None:
            for c in batch:
                if c.get("index") != sel:
                    t = (c.get("trimmed") or c.get("original") or "").strip()
                    if len(t) >= T.MIN_TEXT_LENGTH:
                        out.append((bt, "Dismissed", t, 0.0, 1.0, f"{sid}:{bid}"))
        batch, sel, bt, bid = None, None, None, None

    for i, e in enumerate(events):
        name = e.get("eventName")
        t = min(int(i * T.N_TERTILES / max(n_events, 1)), T.N_TERTILES - 1)
        if name == "suggestion-open":
            flush()
            c = e.get("currentSuggestions") or []
            if isinstance(c, list) and c:
                batch, sel, bt, bid = c, None, t, i
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
    rows = []
    for fp in tqdm(files, desc="menu-ids"):
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
        text, authors, idx = replay(events)
        recs = [(t, lab, s, fh, fa, "")
                for t, lab, s, fh, fa in sentences_with_authorship(text, authors, idx, n)]
        recs += dismissed_with_menu(sid, events, n)
        for tert, label, s, fh, fa, mid in recs:
            rows.append((sid, pk, genre, tert, label, s, fh, fa, mid))

    df = pd.DataFrame(rows, columns=["session", "prompt", "genre", "tertile",
                                     "label", "text", "frac_h", "frac_a",
                                     "menu_id"])
    df = df.drop_duplicates(subset=["prompt", "label", "text"]).reset_index(drop=True)

    ref = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    assert len(df) == len(ref), f"row count mismatch: {len(df)} vs {len(ref)}"
    for col in ["session", "prompt", "label"]:
        assert (df[col].astype(str).values == ref[col].astype(str).values).all(), col
    assert (df["text"].astype(str).values == ref["text"].astype(str).values).all()
    df[["menu_id"]].to_csv(OUT, index=False)
    n_menus = df.loc[df.label == "Dismissed", "menu_id"].nunique()
    n_dis = (df.label == "Dismissed").sum()
    print(f"verified row-for-row. Dismissed rows {n_dis}, distinct menus {n_menus} "
          f"({n_dis / n_menus:.2f} per menu). saved -> {OUT}")


if __name__ == "__main__":
    main()
