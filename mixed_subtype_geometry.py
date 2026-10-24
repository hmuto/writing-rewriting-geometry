"""Effective dimension of each Mixed subtype against writing and the reference.

Within each prompt group and for each subtype S (completion, append,
interleaved) with at least N_MIN sentences, we draw equal-sized samples of S,
of Human sentences, and of held-out Dismissed sentences (n = min(|S|, |R|/4,
200)), 20 times, and average the differences in effective dimension.
A second block compares completion with interleaved at a common n.

Usage: python mixed_subtype_geometry.py [st|openai]
Output: results/replay/mixed_subtype_geometry_{backend}.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

DIR = G.REPO / "results" / "replay" if hasattr(G, "REPO") else None
N_MIN, REPS = 25, 20


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    from pathlib import Path
    d = Path(__file__).resolve().parent / "results" / "replay"
    sent = pd.read_csv(d / "sentences.csv")
    emb = np.load(d / f"emb_{backend}.npy")
    sub = pd.read_csv(d / "mixed_subtypes.csv").set_index("row")["subtype"]
    sent["sub"] = sent.index.map(sub)
    rng = np.random.default_rng(G.SEED)
    rows = []
    for prompt, g in sent.groupby("prompt"):
        R = g.index[g.label == "Dismissed"].values
        H = g.index[g.label == "Human"].values
        pools = {s: g.index[(g.label == "Mixed") & (g["sub"] == s)].values
                 for s in ("completion", "append", "interleaved")}
        pools["all_mixed"] = g.index[g.label == "Mixed"].values
        for s, S in pools.items():
            if len(S) < N_MIN:
                continue
            n = int(min(len(S), len(R) // 4, len(H), 200))
            ds, dh = [], []
            for _ in range(REPS):
                hold = emb[rng.choice(R, n, replace=False)]
                xs = emb[rng.choice(S, n, replace=False)]
                xh = emb[rng.choice(H, n, replace=False)]
                c = G.deff(hold)
                ds.append(G.deff(xs) - c)
                dh.append(G.deff(xh) - c)
            rows.append(dict(prompt=prompt, genre=g.genre.iloc[0], subtype=s, n=n,
                             sub_ctrl=np.mean(ds), human_ctrl=np.mean(dh),
                             human_sub=np.mean(dh) - np.mean(ds)))
        C, I = pools["completion"], pools["interleaved"]
        if len(C) >= N_MIN and len(I) >= N_MIN:
            n = int(min(len(C), len(I), 200))
            diff = [G.deff(emb[rng.choice(I, n, replace=False)])
                    - G.deff(emb[rng.choice(C, n, replace=False)]) for _ in range(REPS)]
            rows.append(dict(prompt=prompt, genre=g.genre.iloc[0],
                             subtype="interleaved_minus_completion", n=n,
                             human_sub=np.mean(diff)))
    out = pd.DataFrame(rows)
    out.to_csv(d / f"mixed_subtype_geometry_{backend}.csv", index=False)
    print(f"[{backend}]")
    for s, x in out.groupby("subtype", sort=False):
        line = f"  {s:28s} groups {len(x):2d}  n {x.n.min()}-{x.n.max()}"
        for col in ("sub_ctrl", "human_ctrl", "human_sub"):
            v = x[col].dropna()
            if len(v) > 1:
                t, p = stats.ttest_1samp(v, 0)
                line += f"  {col} {v.mean():+.2f} ({(v > 0).sum()}/{len(v)}, p={p:.2g})"
        print(line)


if __name__ == "__main__":
    main()
