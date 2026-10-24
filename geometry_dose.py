"""How much of a sentence must the human write before it carries new directions?

geometry_forms.py compares three discrete conditions (Accepted / Mixed / Human).
Those are cut points on one continuous quantity: frac_h, the fraction of a
sentence's surviving characters the writer typed. Binning on frac_h turns the
comparison into a dose-response curve from "GPT-3 wrote it" to "the human wrote
it", with human editing occupying the middle.

Reference and measures are unchanged (Dismissed split fit/control; OOR, deff,
control-adjusted leakage; prompt group is the unit of generalization).

Usage: python geometry_dose.py [st|openai] [replay] [--fullsent]
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


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "replay"
    DIR = REPO / "results" / dataset
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)
    fullsent = "--fullsent" in sys.argv
    if fullsent:
        import re as _re
        ok = df["text"].astype(str).str.strip().map(
            lambda t: len(t.split()) >= 3
            and bool(_re.match(r'^["\u201c\(\'A-Z0-9]', t))
            and bool(_re.search(r'[.!?"\u201d\')\]]$', t)))
        df = df[ok.values]

    # relabel every non-reference sentence by how much of it the human typed
    lab = df["label"].copy()
    m = lab != "Dismissed"
    lab[m] = pd.cut(df.loc[m, "frac_h"], EDGES, right=False, labels=NAMES).astype(str)
    df = df.assign(label=lab)
    G.N_MATCH = 100

    print(f"[{dataset} x {backend}] dose-response on frac_h  "
          f"k={G.K_PCA}  matched n<=100  reps={G.N_REP}")
    print(df[df.label != "Dismissed"].groupby("label").size().to_string())

    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        out = G.run_prompt(sub, emb, rng, None, list(NAMES), N_MIN)
        if out is None:
            continue
        res, n = out
        for c in res:
            rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c, **res[c]))
    d = pd.DataFrame(rows)
    if d.empty:
        print("no prompt group met the sample-size requirement")
        return
    d.to_csv(DIR / f"geometry_dose_{backend}{'_fullsent' if fullsent else ''}.csv",
             index=False)

    label = {"oor": "tail reach (OOR)", "deff": "effective dimension",
             "leak": "directional novelty (leakage)"}
    for genre in ("creative", "ALL"):
        g = d if genre == "ALL" else d[d["genre"] == genre]
        if g["prompt"].nunique() < 3:
            continue
        print(f"\n{'='*76}\n[{genre}]  {g['prompt'].nunique()} prompt groups, "
              f"matched n = {g['n'].min()}-{g['n'].max()}")
        for m in ("oor", "deff", "leak"):
            piv = g.pivot_table(index="prompt", columns="cond", values=m)
            print(f"\n  {label[m]}   (excess over the held-out LLM control)")
            means = []
            for c in NAMES:
                ex = piv[c] - piv["control"]
                t, p = stats.ttest_1samp(ex, 0)
                means.append(ex.mean())
                print(f"    {c:24} {ex.mean():+.3f}   t({len(ex)-1})={t:5.2f} "
                      f"p={p:.3g}  ({(ex > 0).sum()}/{len(ex)} prompts positive)")
            # monotone trend across the four doses, prompt by prompt
            x = np.arange(len(NAMES))
            slopes = [np.polyfit(x, [piv.loc[pr, c] - piv.loc[pr, "control"]
                                     for c in NAMES], 1)[0] for pr in piv.index]
            t, p = stats.ttest_1samp(slopes, 0)
            rho = stats.spearmanr(x, means).statistic
            print(f"    trend: slope/step={np.mean(slopes):+.3f} "
                  f"t({len(slopes)-1})={t:.2f} p={p:.3g}  "
                  f"({sum(s > 0 for s in slopes)}/{len(slopes)} prompts rising)"
                  f"  Spearman rho={rho:+.2f}")


if __name__ == "__main__":
    main()
