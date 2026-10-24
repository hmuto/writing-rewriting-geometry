"""Residual variance spectrum (Supplementary Section S.VII).

For each prompt group, the top-20 principal directions of the Dismissed
sentences are projected out of the Human sentences; the residuals are pooled
by genre and their principal-component variance shares are computed. As a
control, the same is done for the held-out half of the Dismissed pool, with
the other half defining the subspace. Figure S2 is drawn by make_figures.py.

Usage: python residual_spectrum.py [openai|st] [n_comp]
Output: results/replay/residual_spectrum_{backend}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
K_LLM = 20          # size of the model subspace projected out
SEED = 42


def residual_variance(df, emb, genre, n_comp, rng, source="Human"):
    """Variance shares of the pooled residual of `source` sentences."""
    res = []
    for _, sub in df[df["genre"] == genre].groupby("prompt"):
        D = sub.index[sub["label"] == "Dismissed"].values
        X = sub.index[sub["label"] == source].values
        if source == "control":                       # held-out half of Dismissed
            p = rng.permutation(len(D))
            D, X = D[p[:len(p) // 2]], D[p[len(p) // 2:]]
        if len(D) < 4 * K_LLM or len(X) < 50:
            continue
        Rc = emb[D] - emb[D].mean(0)
        V = np.linalg.svd(Rc, full_matrices=False)[2][:K_LLM]
        Xc = emb[X] - emb[X].mean(0)
        res.append(Xc - (Xc @ V.T) @ V)
    Rall = np.vstack(res)
    Rall -= Rall.mean(0)
    S = np.linalg.svd(Rall, compute_uv=False)
    var = S ** 2 / (S ** 2).sum()
    return len(Rall), var[:n_comp]


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "openai"
    n_comp = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    rows = []
    for genre in ("creative", "argumentative"):
        for source in ("Human", "control"):
            rng = np.random.default_rng(SEED)
            n, var = residual_variance(df, emb, genre, n_comp, rng, source)
            print(f"[{genre}] {source:7s} n={n:,}  top {n_comp}: "
                  + ", ".join(f"{v:.2%}" for v in var)
                  + f"  (sum {var.sum():.2%}; isotropic {1 / (emb.shape[1] - K_LLM):.3%})")
            rows += [dict(genre=genre, source=source, pc=c + 1, var=float(v))
                     for c, v in enumerate(var)]
    pd.DataFrame(rows).to_csv(DIR / f"residual_spectrum_{backend}.csv", index=False)


if __name__ == "__main__":
    main()
