"""Is the GPT-5 pool's lower dimension a decoding/length artifact?

(a) Report word-count distributions of the GPT-3 (rejected) pool and the
    regenerated GPT-5 pool.
(b) Length-match the three sets that the concentration claim needs
    (reference = GPT-5 pool, GPT3, Human) in 4-word bins and re-estimate:
      - concentration: GPT3 deff excess over the held-out GPT-5 control
        (positive = GPT-5 pool is narrower at matched length)
      - anchored human contrast: Human - GPT3

Usage: python pool_lenmatch.py st|openai [--binw 4]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

DIR = Path(__file__).resolve().parent / "results" / "gpt5ref"
CONDS = ["GPT3", "Human"]
N_MIN = 40


def main():
    backend = sys.argv[1]
    binw = int(sys.argv[sys.argv.index("--binw") + 1]) if "--binw" in sys.argv else 4
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)

    w = df["text"].astype(str).str.split().str.len()
    print(f"=== word counts (pooled) ===")
    for lab in ["GPT3", "Dismissed", "Human"]:
        ww = w[df["label"] == lab]
        name = "GPT-5" if lab == "Dismissed" else lab
        print(f"  {name:6s} n={len(ww):6d}  p10={ww.quantile(.1):.0f} "
              f"p50={ww.median():.0f} p90={ww.quantile(.9):.0f} mean={ww.mean():.1f}")

    G.N_MATCH = 120
    print(f"\n[{backend}] 3-set length match (binw={binw}), ref=GPT-5, k={G.K_PCA}, reps={G.N_REP}")
    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        sub = G.length_match(sub, rng, CONDS, bin_w=binw)
        if len(sub) == 0:
            continue
        out = G.run_prompt(sub, emb, rng, None, list(CONDS), N_MIN)
        if out is None:
            continue
        res, n = out
        for c in res:
            rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c, **res[c]))
    d = pd.DataFrame(rows)
    if d.empty:
        print("no prompt group met the sample-size requirement")
        return
    d.to_csv(DIR / f"geometry_{backend}_lenmatch3.csv", index=False)

    piv = d.pivot_table(index=["genre", "prompt"], columns="cond", values="deff")
    piv_l = d.pivot_table(index=["genre", "prompt"], columns="cond", values="leak")
    conc = piv["GPT3"] - piv["control"]          # GPT-5 narrower if positive
    anch = piv["Human"] - piv["GPT3"]
    anch_l = piv_l["Human"] - piv_l["GPT3"]
    hum = piv["Human"] - piv["control"]
    print(f"\ngroups kept: {len(piv)}  matched n: {d['n'].min()}--{d['n'].max()}")
    for name, x in [("concentration (GPT3 - GPT5ctrl), deff", conc),
                    ("Human - control, deff", hum),
                    ("Human - GPT3 (anchored), deff", anch),
                    ("Human - GPT3 (anchored), leak", anch_l)]:
        t, p = stats.ttest_rel(x, np.zeros(len(x)))
        print(f"  {name:38s} mean {x.mean():+.2f}  p={p:.2g}  "
              f"positive {int((x > 0).sum())}/{len(x)}")


if __name__ == "__main__":
    main()
