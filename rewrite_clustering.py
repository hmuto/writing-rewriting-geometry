"""Clustering of rewritten sentences.

Tests whether rewrites stay anchored to their source suggestion's context, so
that the Mixed pool carries internal semantic clumps. Two diagnostics at
matched n: the mean within-set nearest-neighbour cosine distance and the
share of sentences whose within-set nearest neighbour comes from the same
session. Word-count spread is reported alongside.

Usage: python rewrite_clustering.py [st|openai]
Output: results/replay/rewrite_clustering_{backend}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"


def nn_stats(E, sess):
    sims = E @ E.T
    np.fill_diagonal(sims, -np.inf)
    j = sims.argmax(1)
    return float((1.0 - sims.max(1)).mean()), float((sess == sess[j]).mean())


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    w = df["text"].astype(str).str.split().str.len()
    print(f"[{backend}] word count per label (mean / SD / p90):")
    for l in ("Dismissed", "Accepted", "Mixed", "Human"):
        x = w[df["label"] == l]
        print(f"  {l:9} {x.mean():5.1f} / {x.std():5.1f} / {x.quantile(.9):4.0f}")

    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        idx = sub.index.values
        lab = sub["label"].values
        sess = sub["session"].values
        pools = {c: np.flatnonzero(lab == c) for c in
                 ("Accepted", "Mixed", "Human")}
        D = np.flatnonzero(lab == "Dismissed")
        n = min(G.N_MATCH, len(D) // 4, *(len(p) for p in pools.values()))
        if n < 40:
            continue
        acc = {c: {"nnd": [], "same": []} for c in
               ("control", "Accepted", "Mixed", "Human")}
        for _ in range(G.N_REP):
            p = rng.permutation(len(D))
            hold = D[p[len(D) // 2:]]
            sel = {"control": hold[rng.choice(len(hold), n, False)]}
            for c, pool in pools.items():
                sel[c] = pool[rng.choice(len(pool), n, False)]
            for c, s in sel.items():
                nnd, same = nn_stats(emb[idx[s]], sess[s])
                acc[c]["nnd"].append(nnd)
                acc[c]["same"].append(same)
        for c, d in acc.items():
            rows.append(dict(genre=genre, prompt=prompt, cond=c,
                             nnd=np.mean(d["nnd"]), same=np.mean(d["same"])))
    d = pd.DataFrame(rows)
    d.to_csv(DIR / f"rewrite_clustering_{backend}.csv", index=False)
    piv = d.pivot_table(index="prompt", columns="cond", values=["nnd", "same"])
    print(f"\nwithin-set structure at matched n ({d['prompt'].nunique()} groups)")
    lbl = {"nnd": "mean NN cosine distance (low = clumped)",
           "same": "share of NN pairs from the same session"}
    for m in ("nnd", "same"):
        print(f"  {lbl[m]}: control={piv[(m,'control')].mean():.4f}")
        for c in ("Accepted", "Mixed", "Human"):
            x = (piv[(m, c)] - piv[(m, "control")]).dropna()
            t, p = stats.ttest_1samp(x, 0)
            print(f"    {c:9} {x.mean():+.4f}  t({len(x)-1})={t:5.2f} p={p:.3g} "
                  f"({(x>0).sum()}/{len(x)} up)")


if __name__ == "__main__":
    main()
