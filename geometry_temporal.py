"""Do the three geometric forms survive the session? (early / mid / late)

Same decomposition as geometry_forms.py, but computed separately inside each
third of the session, against the reference GPT-3 was offering AT THAT POINT.
The question is not whether the writer stays far from the model, but which
kind of contribution fades: reaching the tails, using more directions, or
using different ones.

Prompt group is the unit of generalization; early vs late is a paired test
across prompt groups.

Usage: python geometry_temporal.py [st|openai] [replay] [--decorrelate]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
PART = {0: "early", 1: "mid", 2: "late"}
CONDS = ["Accepted", "Human"]          # Mixed is too small once split in thirds
N_MIN = 50


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "replay"
    decor = "--decorrelate" in sys.argv
    DIR = REPO / "results" / dataset
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)
    per_session = 2 if decor else None

    print(f"[{dataset} x {backend}] temporal  per_session={per_session}  "
          f"conds={CONDS}  k={G.K_PCA}  reps={G.N_REP}")

    rows = []
    for (genre, prompt, tert), sub in df.groupby(["genre", "prompt", "tertile"]):
        rng = np.random.default_rng(G.SEED)
        out = G.run_prompt(sub, emb, rng, per_session, list(CONDS), N_MIN)
        if out is None:
            continue
        res, n = out
        for c in res:
            rows.append(dict(genre=genre, prompt=prompt, part=PART[tert],
                             n=n, cond=c, **res[c]))
    d = pd.DataFrame(rows)
    if d.empty:
        print("no (prompt, tertile) cell met the sample-size requirement")
        return
    d.to_csv(DIR / f"geometry_temporal_{backend}{'_dec' if decor else ''}.csv",
             index=False)

    label = {"oor": "tail reach (OOR)", "deff": "effective dimension",
             "leak": "directional novelty (leakage)", "rad": "radius ratio"}
    for genre in ("creative", "argumentative", "ALL"):
        g = d if genre == "ALL" else d[d["genre"] == genre]
        # keep only prompts measured in all three parts, for a clean paired test
        ok = (g.groupby("prompt")["part"].nunique() == 3)
        g = g[g["prompt"].isin(ok[ok].index)]
        if g["prompt"].nunique() < 3:
            continue
        print(f"\n{'='*76}\n[{genre}]  {g['prompt'].nunique()} prompt groups "
              f"measured in all three parts, matched n = {g['n'].min()}-{g['n'].max()}")

        for m in ("oor", "deff", "leak"):
            piv = g.pivot_table(index="prompt", columns=["cond", "part"], values=m)
            print(f"\n  {label[m]}")
            for c in ["control"] + CONDS:
                e, l = piv[(c, "early")], piv[(c, "late")]
                mid = piv[(c, "mid")]
                t, p = stats.ttest_rel(l, e)
                print(f"    {c:9} early={e.mean():+.3f} mid={mid.mean():+.3f} "
                      f"late={l.mean():+.3f}   late-early={l.mean()-e.mean():+.3f} "
                      f"t({len(e)-1})={t:.2f} p={p:.3g} "
                      f"({(l < e).sum()}/{len(e)} down)")
            # the quantity of interest: the condition's excess over its control
            for c in CONDS:
                ex_e = piv[(c, "early")] - piv[("control", "early")]
                ex_l = piv[(c, "late")] - piv[("control", "late")]
                t, p = stats.ttest_rel(ex_l, ex_e)
                te, pe = stats.ttest_1samp(ex_e, 0)
                tl, pl = stats.ttest_1samp(ex_l, 0)
                print(f"    {c} - control: early={ex_e.mean():+.3f}(p={pe:.3g}) "
                      f"late={ex_l.mean():+.3f}(p={pl:.3g})  "
                      f"change={ex_l.mean()-ex_e.mean():+.3f} p={p:.3g}")


if __name__ == "__main__":
    main()
