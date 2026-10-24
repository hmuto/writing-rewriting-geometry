"""Selection versus editing: the original suggestions behind rewritten sentences.

For every Mixed sentence, the surviving suggestion characters carry the index
of the event that inserted them. The full string inserted by that event is
the suggestion as the model produced it. We collect these original strings
(joining several events in order when a sentence draws on more than one),
embed them with both models, and compare, within each prompt group:

  selection  d_eff(originals of rewritten sentences) - d_eff(held-out Dismissed)
             both are raw model suggestions, so this asks whether the writer
             chose a less varied set of suggestions to rewrite
  editing    d_eff(final rewritten sentences) - d_eff(their own originals),
             paired: the same sampled sentences in both sets

Usage: python mixed_origins.py extract
       python mixed_origins.py embed st|openai
       python mixed_origins.py analyze st|openai [--single]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G
import replay_extract as RX
import temporal as T

D = Path(__file__).resolve().parent / "results" / "replay"
OUT = D / "mixed_origins.csv"
REPS = 20


def inserted_strings(events):
    ins = {}
    for i, e in enumerate(events):
        td = e.get("textDelta")
        if e.get("eventSource") == "user" or not isinstance(td, dict):
            continue
        s = "".join(op["insert"] for op in td.get("ops", [])
                    if isinstance(op, dict) and isinstance(op.get("insert"), str))
        if s:
            ins[i] = s
    return ins


def extract():
    rows = []
    for fp in sorted(glob.glob(str(T.DATA_DIR / "*.jsonl"))):
        sid = Path(fp).stem
        events = [json.loads(l) for l in open(fp)]
        ins = inserted_strings(events)
        text, authors, idx = RX.replay(events)
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
            ev = list(dict.fromkeys(idx[start:end][a == "A"]))
            orig = " ".join(ins[i].strip() for i in ev if i in ins).strip()
            rows.append((sid, s, orig, len(ev)))
    d = pd.DataFrame(rows, columns=["session", "text", "original", "n_events"])
    d = d.drop_duplicates(["session", "text"])
    sent = pd.read_csv(RX.OUT)
    sent["row"] = sent.index
    mixed = sent[sent.label == "Mixed"][["row", "session", "prompt", "genre", "text"]]
    m = mixed.merge(d, on=["session", "text"], how="left")
    m.to_csv(OUT, index=False)
    print(f"Mixed {len(mixed)}; with original {m['original'].fillna('').str.len().gt(0).sum()}; "
          f"multi-event {m['n_events'].gt(1).sum()}")
    print("median words final / original:",
          m.text.str.split().str.len().median(), m.original.fillna("").str.split().str.len().median())


def embed(backend):
    m = pd.read_csv(OUT)
    texts = m["original"].fillna("").astype(str).tolist()
    if backend == "st":
        from sentence_transformers import SentenceTransformer
        E = SentenceTransformer("all-MiniLM-L6-v2", device="cpu").encode(
            texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    else:
        from openai import OpenAI
        client = OpenAI()
        E = []
        for i in range(0, len(texts), 500):
            batch = [t if t else " " for t in texts[i:i + 500]]
            r = client.embeddings.create(model="text-embedding-3-small", input=batch)
            E.extend(x.embedding for x in r.data)
        E = np.asarray(E, dtype=np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
    np.save(D / f"mixed_origins_emb_{backend}.npy", np.asarray(E, dtype=np.float32))
    print(backend, np.asarray(E).shape)


def analyze(backend, single_only=False):
    m = pd.read_csv(OUT)
    eo = np.load(D / f"mixed_origins_emb_{backend}.npy")
    sent = pd.read_csv(RX.OUT)
    emb = np.load(D / f"emb_{backend}.npy")
    ok = m["original"].fillna("").str.len().gt(0).to_numpy().copy()
    if single_only:
        # only sentences whose surviving suggestion text comes from one insertion event
        ok &= (m["n_events"] == 1).values
    rng = np.random.default_rng(G.SEED)
    rows = []
    for prompt, g in m[ok].groupby("prompt"):
        # one original per distinct original string, so repeats do not shrink d_eff
        g = g.drop_duplicates("original")
        R = sent.index[(sent.prompt == prompt) & (sent.label == "Dismissed")].values
        n = int(min(len(g), len(R) // 4, 200))
        if n < 25:
            continue
        sel, edit = [], []
        pos = g.index.values
        for _ in range(REPS):
            pick = rng.choice(pos, n, replace=False)
            c = G.deff(emb[rng.choice(R, n, replace=False)])
            do = G.deff(eo[pick])
            df_ = G.deff(emb[m.loc[pick, "row"].values])
            sel.append(do - c)
            edit.append(df_ - do)
        rows.append(dict(prompt=prompt, n=n, selection=np.mean(sel), editing=np.mean(edit),
                         total=np.mean(sel) + np.mean(edit)))
    out = pd.DataFrame(rows)
    tag = "_single" if single_only else ""
    out.to_csv(D / f"mixed_origins_geometry_{backend}{tag}.csv", index=False)
    print(f"[{backend}{tag}] groups {len(out)}, n {out.n.min()}-{out.n.max()}")
    for col in ("selection", "editing", "total"):
        t, p = stats.ttest_1samp(out[col], 0)
        print(f"  {col:9s} {out[col].mean():+.2f} ({(out[col] > 0).sum()}/{len(out)}, p={p:.2g})")


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "extract":
        extract()
    elif step == "embed":
        embed(sys.argv[2])
    else:
        analyze(sys.argv[2], single_only="--single" in sys.argv)
