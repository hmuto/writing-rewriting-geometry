"""Do the 2021 writers' sentences expand a MODERN model's region?

Builds results/gpt5ref/sentences.csv:
    reference ("Dismissed" slot) = GPT-5 suggestions regenerated on the same
                                   document contexts (gpt5_regen.py)
    conditions                   = GPT3 (the original rejected suggestions),
                                   Accepted, Mixed, Human -- all from the
                                   corrected replay dataset

The GPT3 condition anchors the scale: it says how far one model's output sits
from another model's region. The question is whether the writer's sentences
exceed THAT, not merely zero. A length-matched pass is included because GPT-5's
suggestions are much longer (median ~25 words) than everything else (12--20).

Usage:
  python gpt5_analyze.py build            # assemble dataset (then embed)
  python gpt5_analyze.py st [--lenmatch]  # analyze under an encoder
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "gpt5ref"
CONDS = ["GPT3", "Accepted", "Mixed", "Human"]
N_MIN = 50


def build():
    g5 = pd.read_csv(DIR / "gpt5_suggestions.csv")
    g5["label"] = "Dismissed"                      # occupies the reference slot
    g5["frac_h"], g5["frac_a"] = 0.0, 0.0
    base = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    base["label"] = base["label"].replace({"Dismissed": "GPT3"})
    df = pd.concat([base, g5[base.columns]], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["prompt", "label", "text"]).reset_index(drop=True)
    df.to_csv(DIR / "sentences.csv", index=False)
    print(f"rows {before} -> {len(df)} after dedup")
    print(df.groupby(["genre", "label"]).size().unstack(fill_value=0).to_string())
    print(f"\nsaved -> {DIR/'sentences.csv'}; now run:")
    print("  python fullscale_embed.py st gpt5ref")
    print("  python fullscale_embed.py openai gpt5ref")


def analyze(backend, lenmatch):
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)
    G.N_MATCH = 120
    print(f"[gpt5ref x {backend}] reference = GPT-5 regenerated suggestions  "
          f"lenmatch={lenmatch}  k={G.K_PCA}  reps={G.N_REP}")

    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        if lenmatch:
            sub = G.length_match(sub, rng, CONDS)
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
    suffix = "_lenmatch" if lenmatch else ""
    d.to_csv(DIR / f"geometry_{backend}{suffix}.csv", index=False)

    name = {"oor": "tail reach (OOR)", "deff": "effective dimension",
            "leak": "directional novelty (leakage)"}
    for genre in ("creative", "argumentative", "ALL"):
        g = d if genre == "ALL" else d[d["genre"] == genre]
        if g["prompt"].nunique() < 3:
            continue
        print(f"\n{'='*74}\n[{genre}]  {g['prompt'].nunique()} prompt groups, "
              f"matched n = {g['n'].min()}-{g['n'].max()}")
        piv = {m: g.pivot_table(index="prompt", columns="cond", values=m)
               for m in ("oor", "deff", "leak")}
        for m in ("oor", "deff", "leak"):
            p = piv[m]
            print(f"\n  {name[m]}   (excess over held-out GPT-5 control)")
            for c in CONDS:
                ex = p[c] - p["control"]
                t, pv = stats.ttest_1samp(ex, 0)
                print(f"    {c:9} {ex.mean():+.3f}  t({len(ex)-1})={t:5.2f} "
                      f"p={pv:.3g}  ({(ex > 0).sum()}/{len(ex)} up)")
            hu = (p["Human"] - p["control"]) - (p["GPT3"] - p["control"])
            t, pv = stats.ttest_1samp(hu, 0)
            print(f"    Human - GPT3      : {hu.mean():+.3f}  t={t:.2f} p={pv:.3g} "
                  f"({(hu > 0).sum()}/{len(hu)} up)   <- beyond the model-vs-model gap?")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        analyze(sys.argv[1], "--lenmatch" in sys.argv)
