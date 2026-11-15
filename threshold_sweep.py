"""Sensitivity of the main contrasts to the human-typed-share cutoffs.

The main analysis labels a non-dismissed sentence as writing when its share
of surviving human-typed characters h is at least 0.8, as choosing when the
share of surviving suggestion characters is at least 0.8, and as rewriting
otherwise. Here the same rule is applied with cutoffs (lo, hi):

    h >= hi                      -> writing   (Human)
    suggestion share >= 1 - lo   -> choosing  (Accepted)
    otherwise                    -> rewriting (Mixed)

(0.2, 0.8) reproduces the main labels. For each pair, the main pipeline (geometry_forms.run_prompt, seed 42, 20 draws per
prompt group) is rerun for each pair. Dismissed sentences are unchanged.

Usage: python threshold_sweep.py st|openai
Output: results/replay/threshold_sweep_{backend}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
CUTS = [(0.2, 0.8), (0.1, 0.9), (0.3, 0.7), (0.25, 0.75), (0.2, 0.9), (0.1, 0.8)]
CONDS = ("Accepted", "Mixed", "Human")


def relabel(df, lo, hi):
    d = df.copy()
    keep = d["label"] != "Dismissed"
    h, a = d.loc[keep, "frac_h"], d.loc[keep, "frac_a"]
    d.loc[keep, "label"] = np.where(h >= hi, "Human",
                                    np.where(a >= 1 - lo, "Accepted", "Mixed"))
    return d


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    rows = []
    for lo, hi in CUTS:
        d = relabel(df, lo, hi)
        counts = d[d.label != "Dismissed"].label.value_counts()
        for (genre, prompt), sub in d.groupby(["genre", "prompt"]):
            out = G.run_prompt(sub, emb, np.random.default_rng(G.SEED), None, CONDS)
            if out is None:
                continue
            res, n = out
            rows.append(dict(lo=lo, hi=hi, genre=genre, prompt=prompt, n=n,
                             n_human=int(counts.get("Human", 0)),
                             n_mixed=int(counts.get("Mixed", 0)),
                             w_minus_r=res["Human"]["deff"] - res["Mixed"]["deff"],
                             w_minus_ref=res["Human"]["deff"] - res["control"]["deff"],
                             r_minus_ref=res["Mixed"]["deff"] - res["control"]["deff"]))
    out = pd.DataFrame(rows)
    out.to_csv(DIR / f"threshold_sweep_{backend}.csv", index=False)
    print(f"[{backend}]")
    for (lo, hi), g in out.groupby(["lo", "hi"], sort=False):
        line = f"  lo={lo:.2f} hi={hi:.2f}  groups={len(g)}  n={g.n.min()}-{g.n.max()}  " \
               f"Human={g.n_human.iloc[0]:,} Mixed={g.n_mixed.iloc[0]:,}"
        for col, name in (("w_minus_r", "W-R"), ("w_minus_ref", "W-ref"), ("r_minus_ref", "R-ref")):
            p = stats.ttest_1samp(g[col], 0).pvalue
            line += f"\n      {name:6s} {g[col].mean():+.2f} ({(g[col] > 0).sum()}/{len(g)}, p={p:.2g})"
        print(line)


if __name__ == "__main__":
    main()
