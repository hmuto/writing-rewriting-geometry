"""Union dimension: dimension added to the reference region.

  U(X) = deff(Rfit u X) - deff(Rfit u Rhold)

Positive U(X) means adding X to the reference region raises its effective
dimension more than adding further model output does.

Usage: python union_dimension.py main|gpt5 st|openai [--centered]
Output: results/replay/union_deff_{backend}[_centered].csv (main),
        results/gpt5ref/union_deff_{backend}.csv (gpt5)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent


def main():
    frame, backend = sys.argv[1], sys.argv[2]
    centered = "--centered" in sys.argv
    if frame == "main":
        df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
        emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")
        conds = ["Accepted", "Mixed", "Human"]
        n_cap, n_min, out_path = G.N_MATCH, G.N_MIN, \
            REPO / "results" / "replay" / f"union_deff_{backend}.csv"
    else:
        df = pd.read_csv(REPO / "results" / "gpt5ref" / "sentences.csv")
        emb = np.load(REPO / "results" / "gpt5ref" / f"emb_{backend}.npy")
        conds = ["GPT3", "Human"]
        n_cap, n_min, out_path = 120, 40, \
            REPO / "results" / "gpt5ref" / f"union_deff_{backend}.csv"

    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        lab = sub["label"].values
        row = sub.index.values
        pos = {c: np.flatnonzero(lab == c) for c in conds + ["Dismissed"]}
        D = pos["Dismissed"]
        if len(D) < 4 * n_min:
            continue
        n = min(n_cap, len(D) // 4, *(len(pos[c]) for c in conds))
        if n < n_min:
            continue
        acc = {c: [] for c in conds}
        for _ in range(G.N_REP):
            p = rng.permutation(len(D))
            half1, half2 = D[p[: len(D) // 2]], D[p[len(D) // 2:]]
            if len(half1) < n or len(half2) < n:
                continue
            try:
                Rfit = emb[row[half1[rng.choice(len(half1), n, replace=False)]]]
                Rhold = emb[row[half2[rng.choice(len(half2), n, replace=False)]]]
                if centered:
                    Rfit = Rfit - Rfit.mean(0)
                    Rhold = Rhold - Rhold.mean(0)
                base = G.deff(np.vstack([Rfit, Rhold]))
                for c in conds:
                    X = emb[row[pos[c][rng.choice(len(pos[c]), n, replace=False)]]]
                    if centered:
                        X = X - X.mean(0)
                    acc[c].append(G.deff(np.vstack([Rfit, X])) - base)
            except np.linalg.LinAlgError:
                continue
        if not acc[conds[-1]]:
            continue
        for c in conds:
            rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c,
                             u_deff=float(np.mean(acc[c]))))
    d = pd.DataFrame(rows)
    if centered:
        out_path = out_path.with_name(out_path.stem + "_centered.csv")
    d.to_csv(out_path, index=False)
    print(f"[union {frame} x {backend} centered={centered}] groups={d['prompt'].nunique()}")
    for c in conds:
        x = d[d["cond"] == c]["u_deff"].values
        t, p = stats.ttest_rel(x, np.zeros(len(x)))
        print(f"  {c:9s} U(deff) {x.mean():+.3f} (p={p:.2g}, {int((x>0).sum())}/{len(x)})")


if __name__ == "__main__":
    main()
