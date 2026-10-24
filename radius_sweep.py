"""Out-of-reference rate across radius multipliers (1.0-3.0).

The main analysis sets the neighbourhood scale at 2x the reference's median
nearest-neighbour distance. Here the multiplier sweeps 1.0-3.0 and the rate
is recomputed for every condition.

Usage: python radius_sweep.py [st|openai]
Output: results/replay/radius_sweep_{backend}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
MULT = [1.0, 1.5, 2.0, 2.5, 3.0]


def oor_multi(X, R, mults):
    dr = 1.0 - (R @ R.T)
    np.fill_diagonal(dr, np.inf)
    base = np.median(dr.min(axis=1))
    dx = (1.0 - (X @ R.T)).min(axis=1)
    return {m: float((dx > m * base).mean()) for m in mults}


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        idx = sub.index.values
        lab = sub["label"].values
        pools = {c: idx[lab == c] for c in ("Accepted", "Mixed", "Human")}
        D = idx[lab == "Dismissed"]
        n = min(G.N_MATCH, len(D) // 4, *(len(p) for p in pools.values()))
        if n < 40:
            continue
        acc = {c: {m: [] for m in MULT} for c in
               ("control", "Accepted", "Mixed", "Human")}
        for _ in range(G.N_REP):
            p = rng.permutation(len(D))
            Rfit = emb[D[p[:len(D) // 2]][rng.choice(len(D) // 2, n, False)]]
            hold = D[p[len(D) // 2:]]
            sel = {"control": hold[rng.choice(len(hold), n, False)]}
            for c, pool in pools.items():
                sel[c] = pool[rng.choice(len(pool), n, False)]
            for c, s in sel.items():
                o = oor_multi(emb[s], Rfit, MULT)
                for m in MULT:
                    acc[c][m].append(o[m])
        for c in acc:
            for m in MULT:
                rows.append(dict(genre=genre, prompt=prompt, cond=c, mult=m,
                                 oor=np.mean(acc[c][m])))
    d = pd.DataFrame(rows)
    d.to_csv(DIR / f"radius_sweep_{backend}.csv", index=False)
    print(f"[{backend}]  OOR by radius multiplier "
          f"({d['prompt'].nunique()} groups)")
    piv = d.pivot_table(index=["prompt"], columns=["cond", "mult"], values="oor")
    for m in MULT:
        line = f"  x{m:3}  control={piv[('control', m)].mean():.4f}"
        for c in ("Accepted", "Mixed", "Human"):
            x = (piv[(c, m)] - piv[("control", m)]).dropna()
            t, p = stats.ttest_1samp(x, 0)
            line += f"  {c[0]}:{x.mean():+.4f}(p={p:.2g})"
        print(line)


if __name__ == "__main__":
    main()
