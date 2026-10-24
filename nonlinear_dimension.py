"""Nonlinear dimension estimators and sample-size saturation.

Two parts. First, two nonlinear intrinsic-dimension estimators (TwoNN and
the Levina-Bickel MLE with k=10) are computed inside the standard pipeline,
so the dimensional contrast does not rest on the participation ratio alone.
Second, a saturation analysis reruns the Human-vs-control contrast at
matched n from 25 to 400.

Usage: python nonlinear_dimension.py [st|openai]
Output: results/replay/nonlinear_dimension_{backend}.csv,
        results/replay/pr_saturation_{backend}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
N_SAT = [25, 50, 100, 200, 400]
SEED = 42


def nn_dists(X, k):
    sq = (X ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    part = np.sort(d2, axis=1)[:, :k]
    return np.sqrt(np.maximum(part, 0))


def twonn(X):
    """Facco et al. 2017, MLE form: d = N / sum(log r2/r1)."""
    r = nn_dists(X, 2)
    mu = r[:, 1] / np.maximum(r[:, 0], 1e-12)
    mu = mu[mu > 1.0]
    return len(mu) / np.log(mu).sum() if len(mu) else np.nan


def levina_bickel(X, k=10):
    """Levina-Bickel 2004 MLE with MacKay-Ghahramani averaging."""
    r = nn_dists(X, k)
    logs = np.log(np.maximum(r[:, -1:], 1e-12) / np.maximum(r[:, :-1], 1e-12))
    inv = logs.sum(1) / (k - 1)
    return 1.0 / inv.mean()


# --------------------------------------------------------------------------- #
def part_intrinsic(df, emb, backend):
    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(SEED)
        idx = sub.index.values
        lab = sub["label"].values
        pools = {c: idx[lab == c] for c in ("Accepted", "Mixed", "Human")}
        D = idx[lab == "Dismissed"]
        n = min(G.N_MATCH, len(D) // 4, *(len(p) for p in pools.values()))
        if n < G.N_MIN:
            continue
        acc = {c: {"twonn": [], "lb": []} for c in
               ("control", "Accepted", "Mixed", "Human")}
        for _ in range(G.N_REP):
            p = rng.permutation(len(D))
            hold = D[p[len(D) // 2:]]
            sel = {"control": hold[rng.choice(len(hold), n, replace=False)]}
            for c, pool in pools.items():
                sel[c] = pool[rng.choice(len(pool), n, replace=False)]
            for c, s in sel.items():
                X = emb[s]
                acc[c]["twonn"].append(twonn(X))
                acc[c]["lb"].append(levina_bickel(X))
        for c, d in acc.items():
            rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c,
                             twonn=np.nanmean(d["twonn"]),
                             lb=np.nanmean(d["lb"])))
    d = pd.DataFrame(rows)
    d.to_csv(DIR / f"nonlinear_dimension_{backend}.csv", index=False)
    piv = d.pivot_table(index="prompt", columns="cond", values=["twonn", "lb"])
    print(f"\nINTRINSIC DIMENSION  ({d['prompt'].nunique()} groups, "
          f"matched n as in the main analysis)")
    for m in ("twonn", "lb"):
        print(f"  {m}: control={piv[(m,'control')].mean():.2f}")
        for c in ("Accepted", "Mixed", "Human"):
            x = (piv[(m, c)] - piv[(m, "control")]).dropna()
            t, p = stats.ttest_1samp(x, 0)
            print(f"    {c:9} {x.mean():+.3f}  t({len(x)-1})={t:5.2f} p={p:.3g} "
                  f"({(x>0).sum()}/{len(x)} groups up)")


# --------------------------------------------------------------------------- #
def part_saturation(df, emb, backend):
    nmax = N_SAT[-1]
    eligible = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        lab = sub["label"].values
        if (lab == "Dismissed").sum() >= 4 * nmax and \
           (lab == "Human").sum() >= nmax:
            eligible.append((genre, prompt))
    print(f"\nSATURATION  ({len(eligible)} groups eligible at n={nmax})")
    rows = []
    for n in N_SAT:
        for (genre, prompt) in eligible:
            sub = df[(df["genre"] == genre) & (df["prompt"] == prompt)]
            rng = np.random.default_rng(SEED)
            idx = sub.index.values
            lab = sub["label"].values
            H = idx[lab == "Human"]
            D = idx[lab == "Dismissed"]
            dc, dh = [], []
            for _ in range(G.N_REP):
                p = rng.permutation(len(D))
                hold = D[p[len(D) // 2:]]
                dc.append(G.deff(emb[hold[rng.choice(len(hold), n, False)]]))
                dh.append(G.deff(emb[H[rng.choice(len(H), n, False)]]))
            rows.append(dict(n=n, genre=genre, prompt=prompt,
                             control=np.mean(dc), human=np.mean(dh)))
    d = pd.DataFrame(rows)
    d.to_csv(DIR / f"pr_saturation_{backend}.csv", index=False)
    for n in N_SAT:
        g = d[d["n"] == n]
        gap = g["human"] - g["control"]
        t, p = stats.ttest_1samp(gap, 0)
        print(f"  n={n:4}  control={g['control'].mean():6.2f}  "
              f"human={g['human'].mean():6.2f}  gap={gap.mean():+5.2f}  "
              f"t({len(gap)-1})={t:5.2f} p={p:.3g}  "
              f"({(gap>0).sum()}/{len(gap)} up)")


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)
    print(f"[{backend}]")
    part_intrinsic(df, emb, backend)
    part_saturation(df, emb, backend)


if __name__ == "__main__":
    main()
