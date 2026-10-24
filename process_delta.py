"""Same-condition comparison: does process filtering change the Human excess?

For each prompt group and replicate, ONE reference split (fit half / held-out
control) is drawn, and two Human samples of the same size n are drawn: one
from all Human sentences (character-origin definition) and one from the
process-filtered Human sentences (frac_a == 0, ai_ghost == 0, sugg_bi < thr).
Both excesses are computed against the same control sample, so their
difference isolates the filtering. All draws are without replacement.

Usage: python process_delta.py st|openai [thr]
Output: results/replay/process_delta_{backend}_bi{thr*100}.csv (per prompt group)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"


def main():
    backend = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
    df = pd.read_csv(DIR / "sentences.csv")
    fl = pd.read_csv(DIR / "process_flags.csv")
    assert (df["label"].values == fl["label"].values).all()
    df = pd.concat([df, fl[["ai_ghost", "sugg_bi"]]], axis=1)
    emb = np.load(DIR / f"emb_{backend}.npy")
    ok = (df["frac_a"] == 0) & (df["ai_ghost"] == 0) & (df["sugg_bi"] < thr)

    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        lab = sub["label"].values
        D = sub.index.values[lab == "Dismissed"]
        H_all = sub.index.values[lab == "Human"]
        H_proc = sub.index.values[(lab == "Human") & ok.loc[sub.index].values]
        n = min(G.N_MATCH, len(D) // 4, len(H_proc))
        if n < G.N_MIN:
            continue
        e_all, e_proc = [], []
        for _ in range(G.N_REP):
            p = rng.permutation(D)
            hold = p[len(D) // 2:]                       # held-out half, as in the main analysis
            C = emb[rng.choice(hold, n, replace=False)]
            dc = G.deff(C)
            e_all.append(G.deff(emb[rng.choice(H_all, n, replace=False)]) - dc)
            e_proc.append(G.deff(emb[rng.choice(H_proc, n, replace=False)]) - dc)
        rows.append(dict(genre=genre, prompt=prompt, n=n,
                         n_human_all=len(H_all), n_human_proc=len(H_proc),
                         excess_all=np.mean(e_all), excess_proc=np.mean(e_proc)))
    r = pd.DataFrame(rows)
    r["delta"] = r["excess_proc"] - r["excess_all"]
    tag = f"bi{int(round(thr * 100))}"
    r.to_csv(DIR / f"process_delta_{backend}_{tag}.csv", index=False)

    d = r["delta"]
    t, p = stats.ttest_1samp(d, 0)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    print(f"[{backend} {tag}] groups={len(r)}  n per set {r['n'].min()}-{r['n'].max()}")
    print(f"  excess all  {r['excess_all'].mean():+.2f}   excess proc {r['excess_proc'].mean():+.2f}")
    print(f"  delta (proc - all) {d.mean():+.2f}  95% CI [{d.mean()-half:+.2f}, {d.mean()+half:+.2f}]"
          f"  {(d > 0).sum()}/{len(d)} positive  t={t:.2f} p={p:.3g}")


if __name__ == "__main__":
    main()
