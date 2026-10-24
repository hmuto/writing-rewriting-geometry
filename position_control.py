"""Document-position control.

Model suggestions appear only where the writer requested them; human
sentences come from everywhere in the document. Here Human sentences are
kept only if their median surviving-character insertion-event index lies
within W events of a suggestion-open event in the same session, so the
compared human text occupies the same document neighbourhoods as the
suggestions. The deff / orientation contrast is then recomputed.

Usage: python position_control.py build      # replay -> position_meta.csv
       python position_control.py st|openai  # analysis at W = 25/50/100
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import temporal as T
from replay_extract import replay, PURE, SENT_END
from orientation_frames import decompose

REPO = Path(__file__).resolve().parent
META = REPO / "results" / "replay" / "position_meta.csv"


def sentences_with_idx(text, authors, idx, n_events):
    """replay_extract.sentences_with_authorship, plus the median event index."""
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
        if fp > 0.5:
            continue
        label = ("Human" if fh >= PURE else
                 "Accepted" if fa >= PURE else "Mixed")
        written = idx[start:end][a != "P"]
        if len(written) == 0:
            continue
        med = float(np.median(written))
        tert = min(int(med * T.N_TERTILES / max(n_events, 1)), T.N_TERTILES - 1)
        out.append((tert, label, s, round(float(fh), 3), round(float(fa), 3), med))
    return out


def dismissed_with_idx(events, n_events):
    out, batch, sel, bt, bi = [], None, None, None, None

    def flush():
        nonlocal batch, sel, bt, bi
        if batch is not None:
            for c in batch:
                if c.get("index") != sel:
                    t = (c.get("trimmed") or c.get("original") or "").strip()
                    if len(t) >= T.MIN_TEXT_LENGTH:
                        out.append((bt, "Dismissed", t, 0.0, 1.0, float(bi)))
        batch, sel, bt, bi = None, None, None, None

    for i, e in enumerate(events):
        name = e.get("eventName")
        t = min(int(i * T.N_TERTILES / max(n_events, 1)), T.N_TERTILES - 1)
        if name == "suggestion-open":
            flush()
            c = e.get("currentSuggestions") or []
            if isinstance(c, list) and c:
                batch, sel, bt, bi = c, None, t, i
        elif name == "suggestion-select":
            h = e.get("currentHoverIndex")
            if isinstance(h, int) and batch:
                sel = h
        elif name == "suggestion-close":
            flush()
    flush()
    return out


def build():
    files = sorted(glob.glob(str(T.DATA_DIR / "*.jsonl")))
    rows = []
    for fp in tqdm(files, desc="position"):
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
        menus = [i for i, e in enumerate(events)
                 if e.get("eventName") == "suggestion-open"
                 and isinstance(e.get("currentSuggestions"), list)
                 and e["currentSuggestions"]]
        text, authors, idx = replay(events)
        recs = (sentences_with_idx(text, authors, idx, n)
                + dismissed_with_idx(events, n))
        for tert, label, s, fh, fa, med in recs:
            dist = (min(abs(med - m) for m in menus) if menus else np.inf)
            rows.append((sid, pk, genre, tert, label, s, fh, fa, med, dist))

    df = pd.DataFrame(rows, columns=["session", "prompt", "genre", "tertile",
                                     "label", "text", "frac_h", "frac_a",
                                     "med_idx", "dist_to_menu"])
    df = df.drop_duplicates(subset=["prompt", "label", "text"]).reset_index(drop=True)

    ref = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    assert len(df) == len(ref), f"{len(df)} vs {len(ref)}"
    for col in ["session", "prompt", "label"]:
        assert (df[col].astype(str).values == ref[col].astype(str).values).all(), col
    assert (df["text"].astype(str).values == ref["text"].astype(str).values).all()
    df[["med_idx", "dist_to_menu"]].to_csv(META, index=False)
    hum = df[df.label == "Human"]["dist_to_menu"]
    print(f"verified row-for-row. Human dist_to_menu quantiles: "
          f"p25={hum.quantile(.25):.0f} p50={hum.quantile(.5):.0f} "
          f"p75={hum.quantile(.75):.0f}. saved -> {META}")


def analyze(backend):
    df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")
    meta = pd.read_csv(META)
    assert len(meta) == len(df)
    dist = meta["dist_to_menu"].values

    import geometry_forms as G
    for W in (25, 50, 100):
        keep = ~((df["label"].values == "Human") & (dist > W))
        sub_all = df[keep]
        n_h = (df["label"] == "Human").sum()
        n_k = (sub_all["label"] == "Human").sum()
        rows = []
        for (genre, prompt), sub in sub_all.groupby(["genre", "prompt"]):
            rng = np.random.default_rng(G.SEED)
            out = decompose(sub, emb, ["Human"], 100, 40, rng)
            if out is None:
                continue
            res, n = out
            rows.append(dict(genre=genre, prompt=prompt, n=n, **res["Human"]))
        d = pd.DataFrame(rows)
        line = f"[{backend} W={W:3d}] Human kept {n_k}/{n_h} ({n_k/n_h:.0%})  groups={len(d)}"
        if d.empty:
            print(line + "  (no group met the size requirement)")
            continue
        for m in ("d_deff", "dL_orient"):
            x = d[m].values
            t, p = stats.ttest_rel(x, np.zeros(len(x)))
            line += f"  {m} {x.mean():+.3f} (p={p:.2g}, {int((x>0).sum())}/{len(x)})"
        print(line)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        analyze(sys.argv[1])
