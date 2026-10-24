"""Orientation measured by principal angles, without tail subtraction.

For each condition X, take the top-m principal directions of X (m=20) and
measure their alignment with the reference top-k subspace P_k:

  aff(X) = ||P_k V_X||_F^2 / m   (mean squared cosine of principal angles)

Misalignment = 1 - aff. Reported: condition misalignment minus held-out
control misalignment.

Usage: python principal_angles.py st|openai
Output: results/replay/angles_{backend}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
CONDS = ["Accepted", "Mixed", "Human"]
M = 20


def top_dirs(X, m):
    Xc = X - X.mean(0)
    _, _, vh = np.linalg.svd(Xc, full_matrices=False)
    return vh[:m]


def main():
    backend = sys.argv[1]
    df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")

    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        lab = sub["label"].values
        row = sub.index.values
        pos = {c: np.flatnonzero(lab == c) for c in CONDS + ["Dismissed"]}
        D = pos["Dismissed"]
        if len(D) < 4 * G.N_MIN:
            continue
        n = min(G.N_MATCH, len(D) // 2 // 1, len(D) // 4,
                *(len(pos[c]) for c in CONDS))
        if n < G.N_MIN:
            continue
        acc = {c: [] for c in ["control"] + CONDS}
        for _ in range(G.N_REP):
            p = rng.permutation(len(D))
            h1, h2 = D[p[: len(D) // 2]], D[p[len(D) // 2:]]
            if len(h1) < n or len(h2) < n:
                continue
            try:
                Rfit = emb[row[h1[rng.choice(len(h1), n, replace=False)]]]
                Rc = Rfit - Rfit.mean(0)
                _, _, vhR = np.linalg.svd(Rc, full_matrices=False)
                Pk = vhR[:G.K_PCA]                      # k x d
                sel = {"control": row[h2[rng.choice(len(h2), n, replace=False)]]}
                for c in CONDS:
                    sel[c] = row[pos[c][rng.choice(len(pos[c]), n, replace=False)]]
                for c in ["control"] + CONDS:
                    V = top_dirs(emb[sel[c]], M)         # m x d
                    aff = float(np.linalg.norm(Pk @ V.T) ** 2) / M
                    acc[c].append(1.0 - aff)             # misalignment
            except np.linalg.LinAlgError:
                continue
        if not acc["Human"]:
            continue
        base = float(np.mean(acc["control"]))
        for c in CONDS:
            rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c,
                             mis_excess=float(np.mean(acc[c])) - base))
    d = pd.DataFrame(rows)
    d.to_csv(REPO / "results" / "replay" / f"angles_{backend}.csv", index=False)
    print(f"[principal angles x {backend}] groups={d['prompt'].nunique()} "
          f"(m={M}, k={G.K_PCA})")
    for c in CONDS:
        x = d[d["cond"] == c]["mis_excess"].values
        t, p = stats.ttest_rel(x, np.zeros(len(x)))
        print(f"  {c:9s} misalignment excess {x.mean():+.4f} "
              f"(p={p:.2g}, {int((x>0).sum())}/{len(x)})")


if __name__ == "__main__":
    main()
