"""Apply the spectrum-aligned decomposition to the four h-bins.

Mirrors geometry_dose.py's binning (EDGES/NAMES, N_MATCH=100, shared n
across bins) and spectral_null.py's decomposition: per bin, observed control-adjusted
leakage, spectrum-aligned (flatness) component, and orientation component.

Usage: python dose_orientation.py st|openai
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
EDGES = [0.0, 0.2, 0.5, 0.8, 1.001]
NAMES = ["h<0.2 (LLM text)", "h 0.2-0.5 (light edit)",
         "h 0.5-0.8 (heavy edit)", "h>0.8 (human text)"]
N_MIN = 50


def tail_share(X, k=20):
    Xc = X - X.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    lam = sv ** 2
    return float(lam[k:].sum() / lam.sum())


def main():
    backend = sys.argv[1]
    df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")
    lab = df["label"].copy()
    m = lab != "Dismissed"
    lab[m] = pd.cut(df.loc[m, "frac_h"], EDGES, right=False, labels=NAMES).astype(str)
    df = df.assign(label=lab)

    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        labv = sub["label"].values
        row = sub.index.values
        pos = {c: np.flatnonzero(labv == c) for c in NAMES + ["Dismissed"]}
        D = pos["Dismissed"]
        if len(D) < 4 * N_MIN:
            continue
        n = min(100, len(D) // 4, *(len(pos[c]) for c in NAMES))
        if n < N_MIN:
            continue
        acc = {c: {mm: [] for mm in ("obs", "spec")} for c in ["control"] + NAMES}
        for _ in range(G.N_REP):
            p = rng.permutation(len(D))
            half1, half2 = D[p[: len(D) // 2]], D[p[len(D) // 2:]]
            if len(half1) < n or len(half2) < n:
                continue
            sel = {"Rfit": row[half1[rng.choice(len(half1), n, replace=False)]],
                   "control": row[half2[rng.choice(len(half2), n, replace=False)]]}
            for c in NAMES:
                sel[c] = row[pos[c][rng.choice(len(pos[c]), n, replace=False)]]
            Rfit = emb[sel["Rfit"]]
            for c in ["control"] + NAMES:
                X = emb[sel[c]]
                acc[c]["obs"].append(G.leakage(X, Rfit))
                acc[c]["spec"].append(tail_share(X, G.K_PCA))
        if not acc[NAMES[-1]]["obs"]:
            continue
        mean = {c: {mm: float(np.mean(v)) for mm, v in d2.items()}
                for c, d2 in acc.items()}
        for c in NAMES:
            rows.append(dict(
                genre=genre, prompt=prompt, n=n, cond=c,
                dL_obs=mean[c]["obs"] - mean["control"]["obs"],
                dL_spec=mean[c]["spec"] - mean["control"]["spec"],
            ))
    d = pd.DataFrame(rows)
    d["dL_orient"] = d["dL_obs"] - d["dL_spec"]
    d.to_csv(REPO / "results" / "replay" / f"dose_orient_{backend}.csv", index=False)

    print(f"[{backend}] h-bin decomposition (k={G.K_PCA}, reps={G.N_REP}, "
          f"groups={d['prompt'].nunique()})")
    for c in NAMES:
        s = d[d["cond"] == c]
        line = f"  {c:24s}"
        for mm in ["dL_obs", "dL_spec", "dL_orient"]:
            x = s[mm].values
            t, p = stats.ttest_rel(x, np.zeros(len(x)))
            line += f"  {mm[3:]} {x.mean():+.4f} (p={p:.1g}, {int((x>0).sum())}/{len(x)})"
        print(line)


if __name__ == "__main__":
    main()
