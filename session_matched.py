"""Writing versus rewriting with identical session composition.

Within each prompt group we keep sessions that contain at least m Mixed and
m Human sentences (and m Dismissed). Each draw takes m sentences of each set
from every one of these sessions, so the writing, rewriting and reference
samples come from exactly the same sessions with the same number of
sentences per session. We repeat 20 draws and average.

Usage: python session_matched.py [st|openai]
Output: results/replay/session_matched_{backend}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

N_SESS_MIN, REPS = 15, 20


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    d = Path(__file__).resolve().parent / "results" / "replay"
    sent = pd.read_csv(d / "sentences.csv")
    emb = np.load(d / f"emb_{backend}.npy")
    rng = np.random.default_rng(G.SEED)
    rows = []
    for m in (1, 2):
        for prompt, g in sent.groupby("prompt"):
            cnt = g.groupby(["session", "label"]).size().unstack(fill_value=0)
            ok = cnt.index[(cnt.get("Mixed", 0) >= m) & (cnt.get("Human", 0) >= m)
                           & (cnt.get("Dismissed", 0) >= m)]
            if len(ok) < N_SESS_MIN:
                continue
            by = {lab: {s: g.index[(g.session == s) & (g.label == lab)].values for s in ok}
                  for lab in ("Mixed", "Human", "Dismissed")}
            dw, dr, dwr = [], [], []
            for _ in range(REPS):
                pick = {lab: np.concatenate([rng.choice(by[lab][s], m, replace=False)
                                             for s in ok]) for lab in by}
                c = G.deff(emb[pick["Dismissed"]])
                w = G.deff(emb[pick["Human"]])
                r = G.deff(emb[pick["Mixed"]])
                dw.append(w - c); dr.append(r - c); dwr.append(w - r)
            rows.append(dict(m=m, prompt=prompt, genre=g.genre.iloc[0], sessions=len(ok),
                             n=len(ok) * m, write_ctrl=np.mean(dw),
                             rewrite_ctrl=np.mean(dr), write_rewrite=np.mean(dwr)))
    out = pd.DataFrame(rows)
    out.to_csv(d / f"session_matched_{backend}.csv", index=False)
    print(f"[{backend}]")
    for m, x in out.groupby("m"):
        line = (f"  m={m}: groups {len(x)}, sessions {x.sessions.min()}-{x.sessions.max()}, "
                f"n {x.n.min()}-{x.n.max()}")
        for col in ("write_rewrite", "write_ctrl", "rewrite_ctrl"):
            t, p = stats.ttest_1samp(x[col], 0)
            line += f"\n     {col:14s} {x[col].mean():+.2f} ({(x[col] > 0).sum()}/{len(x)}, p={p:.2g})"
        print(line)


if __name__ == "__main__":
    main()
