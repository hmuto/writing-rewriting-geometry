"""Does the Human dimensional excess survive a process-based definition of writing?

Restricts the Human set to sentences that never passed through a suggestion:
no surviving suggestion characters (frac_a == 0), no deleted-suggestion marks
(ai_ghost == 0), and low bigram overlap with suggestions shown earlier
(sugg_bi < threshold). Other Human sentences are dropped from every sample.
The geometry pipeline (geometry_forms.run_prompt) is otherwise unchanged.

The reassign variant instead moves AI-touched Human sentences into Mixed,
so Mixed becomes 'passed through a suggestion and was modified'.

Usage: python process_robust.py st|openai [--lenmatch]
Output: results/replay/process_robust_{backend}[_lenmatch].csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
CONDS = ["Accepted", "Mixed", "Human"]
VARIANTS = {                       # name -> (sugg_bi threshold, require frac_h == 1)
    "original": None,
    "process_bi30": (0.30, False),
    "process_bi20": (0.20, False),
    "process_bi30_h1": (0.30, True),
    "reassign_bi30": "reassign",   # AI-touched Human -> Mixed (action-based rewrite)
}


def run(df, emb, lenm):
    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        if lenm:
            sub = G.length_match(sub, rng, CONDS)
            if len(sub) == 0:
                continue
        out = G.run_prompt(sub, emb, rng, None, CONDS, 40 if lenm else G.N_MIN)
        if out is None:
            continue
        res, n = out
        for c in res:
            rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c, **res[c]))
    return pd.DataFrame(rows)


def main():
    backend = sys.argv[1]
    lenm = "--lenmatch" in sys.argv
    df = pd.read_csv(DIR / "sentences.csv")
    fl = pd.read_csv(DIR / "process_flags.csv")
    assert (df["label"].values == fl["label"].values).all()
    df = pd.concat([df, fl[["ai_ghost", "sugg_bi"]]], axis=1)
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)

    out = []
    for name, v in VARIANTS.items():
        d = df
        if v == "reassign":
            h = d["label"] == "Human"
            ok = (d["frac_a"] == 0) & (d["ai_ghost"] == 0) & (d["sugg_bi"] < 0.30)
            d = d.copy()
            d.loc[h & ~ok, "label"] = "Mixed"
        elif v is not None:
            thr, h1 = v
            h = d["label"] == "Human"
            ok = (d["frac_a"] == 0) & (d["ai_ghost"] == 0) & (d["sugg_bi"] < thr)
            if h1:
                ok &= d["frac_h"] == 1
            d = d[~h | ok]
        nh = int((d["label"] == "Human").sum())
        g = run(d, emb, lenm)
        piv = g.pivot_table(index="prompt", columns="cond", values="deff")
        for c, ref in (("Human", "control"), ("Mixed", "control"),
                       ("Accepted", "control"), ("Human", "Mixed")):
            diff = piv[c] - piv[ref]
            t, p = stats.ttest_rel(piv[c], piv[ref])
            half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            out.append(dict(variant=name, n_human=nh, contrast=f"{c}-{ref}",
                            groups=len(diff), n_min=int(g['n'].min()), n_max=int(g['n'].max()),
                            mean=diff.mean(), ci_lo=diff.mean() - half, ci_hi=diff.mean() + half,
                            pos=int((diff > 0).sum()), t=t, p=p))
    r = pd.DataFrame(out)
    suf = "_lenmatch" if lenm else ""
    r.to_csv(DIR / f"process_robust_{backend}{suf}.csv", index=False)
    print(f"[{backend}{suf}] effective-dimension contrasts across prompt groups")
    print(r.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
