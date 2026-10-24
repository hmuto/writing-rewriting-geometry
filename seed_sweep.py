"""Main-analysis Human excess under seeds 41-45.

Usage: python seed_sweep.py st|openai
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import geometry_forms as G

REPO = Path(__file__).resolve().parent
CONDS = ["Accepted", "Mixed", "Human"]


def main():
    backend = sys.argv[1]
    df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")

    means = {}
    for seed in [41, 42, 43, 44, 45]:
        rows = []
        for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
            rng = np.random.default_rng(seed)
            out = G.run_prompt(sub, emb, rng, None, CONDS)
            if out is None:
                continue
            res, n = out
            rows.append(dict(prompt=prompt,
                             deff=res["Human"]["deff"] - res["control"]["deff"],
                             leak=res["Human"]["leak"]))
        d = pd.DataFrame(rows)
        means[seed] = (d["deff"].mean(), d["leak"].mean(),
                       int((d["leak"] > 0).sum()), len(d))
        print(f"[{backend}] seed {seed}: Human deff {d['deff'].mean():+.2f}  "
              f"leak {d['leak'].mean():+.4f}  leak-pos {int((d['leak']>0).sum())}/{len(d)}")
    de = [v[0] for v in means.values()]
    le = [v[1] for v in means.values()]
    print(f"[{backend}] across 5 seeds: deff {np.mean(de):+.2f} (SD {np.std(de, ddof=1):.3f})  "
          f"leak {np.mean(le):+.4f} (SD {np.std(le, ddof=1):.5f})")


if __name__ == "__main__":
    main()
