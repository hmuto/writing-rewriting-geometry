"""Menu-matched selection analysis.

Reference = only the rejected siblings from menus in which the writer
accepted a candidate (same menus and contexts as the accepted set; only the
writer's choice separates the sets). X = Accepted.

Usage: python menu_matched.py st|openai
Output: results/replay/menu_matched_{backend}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G
from orientation_frames import decompose

REPO = Path(__file__).resolve().parent


def main():
    backend = sys.argv[1]
    df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")
    menus = pd.read_csv(REPO / "results" / "replay" / "menu_ids.csv")["menu_id"]
    acc_menus = set(pd.read_csv(REPO / "results" / "replay" /
                                "accepted_menus.csv")["menu_id"])
    assert len(menus) == len(df)
    is_sib = menus.isin(acc_menus).values

    keep = (df["label"] != "Dismissed") | is_sib
    df2 = df[keep.values]
    rows = []
    for (genre, prompt), sub in df2.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        out = decompose(sub, emb, ["Accepted"], G.N_MATCH, 40, rng)
        if out is None:
            continue
        res, n = out
        rows.append(dict(genre=genre, prompt=prompt, n=n, **res["Accepted"]))
    d = pd.DataFrame(rows)
    d.to_csv(REPO / "results" / "replay" / f"menu_matched_{backend}.csv",
             index=False)
    print(f"[menu-matched x {backend}] groups={len(d)}  n={d['n'].min()}--{d['n'].max()}")
    for m in ["d_deff", "dL_obs", "dL_orient"]:
        x = d[m].values
        t, p = stats.ttest_rel(x, np.zeros(len(x)))
        print(f"  Accepted {m}: {x.mean():+.4f} (p={p:.2g}, {int((x>0).sum())}/{len(x)})")


if __name__ == "__main__":
    main()
