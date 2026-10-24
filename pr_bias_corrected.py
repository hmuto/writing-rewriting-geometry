"""Bias-corrected participation ratio for the main and length-matched contrasts.

The plug-in participation ratio (tr S)^2 / tr(S^2) is biased downward in
finite samples. Under a Gaussian working model,
    a2 = (n-1)^2 / ((n-2)(n+1)) * [tr(S^2) - (tr S)^2 / (n-1)]
is unbiased for tr(Sigma^2) (Srivastava, 2005), and
    E[(tr S)^2] = (tr Sigma)^2 + 2 tr(Sigma^2) / (n-1),
so the corrected ratio is [(tr S)^2 - 2 a2 / (n-1)] / a2. Embeddings are not
Gaussian and sentences within a document are dependent, so the correction is
approximate. We compute plug-in and corrected values on the same draws, at
the main sample size and at n = 50, to show how the correction changes the
scale of the writing-rewriting gap.

Usage: python pr_bias_corrected.py [st|openai]
Output: results/replay/pr_bias_corrected_{backend}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPS = 20


def pr_both(X):
    n = len(X)
    Xc = X - X.mean(0)
    K = Xc @ Xc.T / (n - 1)               # n x n Gram; same nonzero spectrum as S
    trS = np.trace(K)
    trS2 = (K ** 2).sum()
    a2 = (n - 1) ** 2 / ((n - 2) * (n + 1)) * (trS2 - trS ** 2 / (n - 1))
    return trS ** 2 / trS2, (trS ** 2 - 2 * a2 / (n - 1)) / a2


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    d = Path(__file__).resolve().parent / "results" / "replay"
    sent = pd.read_csv(d / "sentences.csv")
    emb = np.load(d / f"emb_{backend}.npy")
    rng = np.random.default_rng(G.SEED)
    rows = []
    for prompt, g in sent.groupby("prompt"):
        pools = {lab: g.index[g.label == lab].values
                 for lab in ("Dismissed", "Accepted", "Mixed", "Human")}
        n_main = int(min(200, len(pools["Dismissed"]) // 4,
                         *(len(pools[c]) for c in ("Accepted", "Mixed", "Human"))))
        for size, n in (("main", n_main), ("n50", 50)):
            acc = {k: [] for k in ("ctrl", "Mixed", "Human")}
            for _ in range(REPS):
                acc["ctrl"].append(pr_both(emb[rng.choice(pools["Dismissed"], n, replace=False)]))
                for c in ("Mixed", "Human"):
                    acc[c].append(pr_both(emb[rng.choice(pools[c], n, replace=False)]))
            m = {k: np.mean(v, axis=0) for k, v in acc.items()}
            rows.append(dict(prompt=prompt, size=size, n=n,
                             ctrl_plug=m["ctrl"][0], ctrl_corr=m["ctrl"][1],
                             wr_plug=m["Human"][0] - m["Mixed"][0],
                             wr_corr=m["Human"][1] - m["Mixed"][1],
                             wc_plug=m["Human"][0] - m["ctrl"][0],
                             wc_corr=m["Human"][1] - m["ctrl"][1]))
    out = pd.DataFrame(rows)
    out.to_csv(d / f"pr_bias_corrected_{backend}.csv", index=False)
    print(f"[{backend}]")
    for size, x in out.groupby("size", sort=False):
        line = f"  {size:4s} n {x.n.min()}-{x.n.max()}  ref PR plug {x.ctrl_plug.mean():.1f} corr {x.ctrl_corr.mean():.1f}"
        for col in ("wr_plug", "wr_corr", "wc_plug", "wc_corr"):
            t, p = stats.ttest_1samp(x[col], 0)
            line += f"\n     {col:8s} {x[col].mean():+.2f} ({(x[col] > 0).sum()}/{len(x)}, p={p:.2g})"
        print(line)


if __name__ == "__main__":
    main()
