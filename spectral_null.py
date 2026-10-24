"""Decompose directional novelty into a spectral-flatness component
and an orientation-specific component.

If a condition's spectrum is flatter than the control's, its leakage past the
reference top-k rises even with maximally aligned axes. The spectrum-aligned
null assigns each condition the leakage it would have if its principal axes
coincided, in order, with the reference's: tail share of its OWN spectrum
beyond k. The orientation component is observed minus aligned.

  dL_obs(c)    = L(X_c; Rfit) - L(control; Rfit)          (the paper's DL20)
  dL_spec(c)   = tail_k(X_c)  - tail_k(control)           (flatness only)
  dL_orient(c) = dL_obs(c) - dL_spec(c)                   (orientation proper)

Usage: python spectral_null.py st|openai
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
CONDS = ["Accepted", "Mixed", "Human"]


def tail_share(X, k=20):
    Xc = X - X.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    lam = sv ** 2
    return float(lam[k:].sum() / lam.sum())


def main():
    backend = sys.argv[1]
    df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")
    assert len(df) == len(emb)

    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        lab = sub["label"].values
        row = sub.index.values
        pos = {c: np.flatnonzero(lab == c) for c in CONDS + ["Dismissed"]}
        D = pos["Dismissed"]
        if len(D) < 4 * G.N_MIN:
            continue
        n = min(G.N_MATCH, len(D) // 4, *(len(pos[c]) for c in CONDS))
        if n < G.N_MIN:
            continue
        acc = {c: {m: [] for m in ("obs", "spec")} for c in ["control"] + CONDS}
        for _ in range(G.N_REP):
            p = rng.permutation(len(D))
            sel = {"Rfit": row[D[p[: len(D) // 2]][rng.choice(len(D) // 2, n, replace=False)]]
                   if len(D) // 2 >= n else None}
            half2 = D[p[len(D) // 2:]]
            sel["control"] = (row[half2[rng.choice(len(half2), n, replace=False)]]
                              if len(half2) >= n else None)
            for c in CONDS:
                sel[c] = (row[pos[c][rng.choice(len(pos[c]), n, replace=False)]]
                          if len(pos[c]) >= n else None)
            if any(v is None for v in sel.values()):
                continue
            Rfit = emb[sel["Rfit"]]
            for c in ["control"] + CONDS:
                X = emb[sel[c]]
                acc[c]["obs"].append(G.leakage(X, Rfit))
                acc[c]["spec"].append(tail_share(X, G.K_PCA))
        if not acc["Human"]["obs"]:
            continue
        mean = {c: {m: float(np.mean(v)) for m, v in d.items()} for c, d in acc.items()}
        for c in CONDS:
            rows.append(dict(
                genre=genre, prompt=prompt, n=n, cond=c,
                dL_obs=mean[c]["obs"] - mean["control"]["obs"],
                dL_spec=mean[c]["spec"] - mean["control"]["spec"],
            ))
    d = pd.DataFrame(rows)
    d["dL_orient"] = d["dL_obs"] - d["dL_spec"]
    d.to_csv(REPO / "results" / "replay" / f"spectral_null_{backend}.csv", index=False)

    print(f"[{backend}] spectrum-aligned decomposition of DL20 (k={G.K_PCA}, "
          f"reps={G.N_REP}, groups={d['prompt'].nunique()})")
    for c in CONDS:
        s = d[d["cond"] == c]
        for m in ["dL_obs", "dL_spec", "dL_orient"]:
            x = s[m].values
            t, p = stats.ttest_rel(x, np.zeros(len(x)))
            print(f"  {c:9s} {m:9s} mean {x.mean():+.4f}  p={p:.2g}  "
                  f"pos {int((x > 0).sum())}/{len(x)}")
        print()


if __name__ == "__main__":
    main()
