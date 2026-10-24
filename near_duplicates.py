"""Near-duplicate collapse beyond exact-string deduplication.

The model reproduces near-identical suggestions across sessions and menus, so
clusters of paraphrase-level repeats could depress the reference's effective
dimension. Every set is greedily thinned so that no two kept sentences within
a prompt group and label exceed a cosine-similarity threshold, and the main
pipeline is rerun at thresholds 0.99, 0.97 and 0.95.

Usage: python near_duplicates.py [st|openai]
Output: results/replay/geometry_{backend}_dedup{99,97,95}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
THRESHOLDS = [0.99, 0.97, 0.95]


def greedy_keep(E, thr):
    """Indices to keep so that pairwise cosine similarity stays below thr."""
    n = len(E)
    if n <= 1:
        return np.arange(n)
    sims = E @ E.T
    keep = []
    for i in range(n):
        if not keep or sims[i, keep].max() < thr:
            keep.append(i)
    return np.array(keep)


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)

    for thr in THRESHOLDS:
        keep = np.zeros(len(df), dtype=bool)
        for _, idx in df.groupby(["genre", "prompt", "label"]).indices.items():
            idx = np.asarray(idx)
            keep[idx[greedy_keep(emb[idx], thr)]] = True
        kept = df[keep]
        drop = (df["label"].value_counts() - kept["label"].value_counts())
        print(f"\n[{backend}] thr={thr}  dropped per label: "
              + "  ".join(f"{l}={int(drop[l])}" for l in
                          ("Dismissed", "Accepted", "Mixed", "Human")))
        rows = []
        for (genre, prompt), sub in kept.groupby(["genre", "prompt"]):
            rng = np.random.default_rng(G.SEED)
            out = G.run_prompt(sub, emb, rng, None,
                               ("Accepted", "Mixed", "Human"), 40)
            if out is None:
                continue
            res, n = out
            for c in res:
                rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c,
                                 **res[c]))
        d = pd.DataFrame(rows)
        d.to_csv(DIR / f"geometry_{backend}_dedup{int(thr*100)}.csv", index=False)
        piv = d.pivot_table(index="prompt", columns="cond",
                            values=["deff", "leak"])
        print(f"  {d['prompt'].nunique()} groups, n={d['n'].min()}-{d['n'].max()}")
        for m in ("deff", "leak"):
            for c in ("Accepted", "Mixed", "Human"):
                x = (piv[(m, c)] - piv[(m, "control")]).dropna()
                t, p = stats.ttest_1samp(x, 0)
                print(f"    {m:5} {c:9} {x.mean():+.4f}  t({len(x)-1})={t:5.2f} "
                      f"p={p:.3g}  ({(x>0).sum()}/{len(x)} up)")


if __name__ == "__main__":
    main()
