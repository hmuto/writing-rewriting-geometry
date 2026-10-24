"""Orientation component at several subspace cuts k.

For k in {10, 20, 30, 40}, computes the control-adjusted orientation
component
    [L_k(X; R_fit) - T_k(X)] - [L_k(R_hold; R_fit) - T_k(R_hold)]
for writing (Human) and rewriting (Mixed), with equal-sized samples and 20
draws per prompt group, as in the main analysis.

Usage: python orient_k_sweep.py [st|openai]
Output: results/replay/orient_k_sweep_{backend}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import geometry_forms as G

KS, REPS = (10, 20, 30, 40), 20


def tail(X, k):
    lam = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    return lam[k:].sum() / lam.sum()


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
        R = pools["Dismissed"]
        n = int(min(200, len(R) // 4, *(len(pools[c]) for c in ("Accepted", "Mixed", "Human"))))
        acc = {(c, k): [] for c in ("Human", "Mixed") for k in KS}
        for _ in range(REPS):
            p = rng.permutation(R)
            fit, hold = emb[p[:n]], emb[p[n:2 * n]]
            xs = {c: emb[rng.choice(pools[c], n, replace=False)] for c in ("Human", "Mixed")}
            for k in KS:
                base = G.leakage(hold, fit, k) - tail(hold, k)
                for c, X in xs.items():
                    acc[(c, k)].append(G.leakage(X, fit, k) - tail(X, k) - base)
        for (c, k), v in acc.items():
            rows.append(dict(prompt=prompt, cond=c, k=k, orient=np.mean(v)))
    out = pd.DataFrame(rows)
    out.to_csv(d / f"orient_k_sweep_{backend}.csv", index=False)
    print(f"[{backend}]")
    for (c, k), x in out.groupby(["cond", "k"]):
        print(f"  {c:6s} k={k:2d}  {x.orient.mean():+.4f}  ({(x.orient > 0).sum()}/{len(x)})")


if __name__ == "__main__":
    main()
