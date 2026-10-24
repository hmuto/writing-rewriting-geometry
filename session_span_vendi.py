"""Session span per draw, and the Vendi score.

span : median number of distinct sessions in a matched draw, per condition
       (main sets) and per authorship bin.
vendi: exp(Shannon entropy of cosine-gram eigenvalues) at matched n,
       Human vs held-out control.

Usage: python session_span_vendi.py st|openai
Output: results/replay/span_{backend}.csv, results/replay/vendi_{backend}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
CONDS = ["Accepted", "Mixed", "Human"]
EDGES = [0.0, 0.2, 0.5, 0.8, 1.001]
BINS = ["h<0.2", "h 0.2-0.5", "h 0.5-0.8", "h>=0.8"]


def vendi(X):
    Xc = X - X.mean(0)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    K = Xn @ Xn.T / len(Xn)
    lam = np.linalg.eigvalsh(K)
    lam = np.clip(lam, 1e-12, None)
    lam = lam / lam.sum()
    return float(np.exp(-(lam * np.log(lam)).sum()))


def main():
    backend = sys.argv[1]
    df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")

    span_rows, vendi_rows = [], []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        lab = sub["label"].values
        sess = sub["session"].values
        row = sub.index.values
        pos = {c: np.flatnonzero(lab == c) for c in CONDS + ["Dismissed"]}
        D = pos["Dismissed"]
        if len(D) < 4 * G.N_MIN:
            continue
        n = min(G.N_MATCH, len(D) // 4, *(len(pos[c]) for c in CONDS))
        if n < G.N_MIN:
            continue
        spans = {c: [] for c in ["control"] + CONDS}
        vend = {"control": [], "Human": []}
        for _ in range(G.N_REP):
            p = rng.permutation(len(D))
            h2 = D[p[len(D) // 2:]]
            if len(h2) < n:
                continue
            sel = {"control": h2[rng.choice(len(h2), n, replace=False)]}
            for c in CONDS:
                sel[c] = pos[c][rng.choice(len(pos[c]), n, replace=False)]
            for c in ["control"] + CONDS:
                spans[c].append(len(set(sess[sel[c]])))
            for c in ("control", "Human"):
                vend[c].append(vendi(emb[row[sel[c]]]))
        for c in ["control"] + CONDS:
            span_rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c,
                                  med_sessions=float(np.median(spans[c]))))
        vendi_rows.append(dict(genre=genre, prompt=prompt, n=n,
                               v_control=float(np.mean(vend["control"])),
                               v_human=float(np.mean(vend["Human"]))))

    # dose-frame spans
    lab = df["label"].copy()
    m = lab != "Dismissed"
    lab[m] = pd.cut(df.loc[m, "frac_h"], EDGES, right=False, labels=BINS).astype(str)
    df2 = df.assign(label=lab)
    for (genre, prompt), sub in df2.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        labv = sub["label"].values
        sess = sub["session"].values
        pos = {c: np.flatnonzero(labv == c) for c in BINS + ["Dismissed"]}
        if len(pos["Dismissed"]) < 200:
            continue
        n = min(100, len(pos["Dismissed"]) // 4, *(len(pos[c]) for c in BINS))
        if n < 50:
            continue
        for c in BINS:
            sp = [len(set(sess[pos[c][rng.choice(len(pos[c]), n, replace=False)]]))
                  for _ in range(G.N_REP)]
            span_rows.append(dict(genre=genre, prompt=prompt, n=n,
                                  cond=f"bin {c}",
                                  med_sessions=float(np.median(sp))))

    sd = pd.DataFrame(span_rows)
    sd.to_csv(REPO / "results" / "replay" / f"span_{backend}.csv", index=False)
    piv = sd.groupby("cond")["med_sessions"].median()
    print(f"[span x {backend}] median distinct sessions per draw:")
    print(piv.round(1).to_string())

    vd = pd.DataFrame(vendi_rows)
    vd.to_csv(REPO / "results" / "replay" / f"vendi_{backend}.csv", index=False)
    ex = vd["v_human"] - vd["v_control"]
    t, p = stats.ttest_rel(ex, np.zeros(len(ex)))
    print(f"[vendi x {backend}] control {vd['v_control'].mean():.1f}  "
          f"human {vd['v_human'].mean():.1f}  excess {ex.mean():+.2f} "
          f"(p={p:.2g}, {int((ex>0).sum())}/{len(ex)})")


if __name__ == "__main__":
    main()
