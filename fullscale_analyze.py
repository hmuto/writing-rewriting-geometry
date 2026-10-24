"""Scalar benchmark: per-sentence distances + mixed-effects model.

Design
------
For every (prompt, tertile) cell we draw a reference pool P of LLM sentences
(size ref_size). Human queries are scored against P; LLM queries are the LLM
sentences NOT in P, so a sentence is never compared with itself and both
sources see a reference pool of the SAME size (matched baseline). Repeating
this R times and averaging gives a stable per-sentence distance.

Then, separately per genre, we fit

    distance ~ C(source)*C(part) + C(prompt),  random intercept per session

  * C(source)[Human]                -> the gap at the EARLY part
  * C(source)[Human]:C(part)[late]  -> change of the gap early->late

Usage: python fullscale_analyze.py [st|openai] [replay|gpt5ref|fullscale]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
DATASET = sys.argv[2] if len(sys.argv) > 2 else "replay"
DIR = REPO / "results" / DATASET
K_NN = 5
N_REF = 50
N_REP = 20
SEED = 42
PART = {0: "early", 1: "mid", 2: "late"}


def knn_mean(Q, R, k=K_NN):
    d = 1.0 - (Q @ R.T)
    kk = min(k, d.shape[1])
    return np.partition(d, kk - 1, axis=1)[:, :kk].mean(axis=1)


def per_sentence_distances(df, emb, rng):
    """Return a distance array aligned with df (NaN where not estimated)."""
    dist_sum = np.zeros(len(df))
    dist_n = np.zeros(len(df))
    is_llm = df["label"].isin(["Accepted", "Dismissed"]).values

    for (prompt, tert), idx in df.groupby(["prompt", "tertile"]).indices.items():
        idx = np.asarray(idx)
        L = idx[is_llm[idx]]
        H = idx[~is_llm[idx]]
        if len(L) < 4 or len(H) == 0:
            continue
        ref_size = int(min(N_REF, len(L) // 2))
        if ref_size < K_NN:
            continue
        EL, EH = emb[L], emb[H]
        for _ in range(N_REP):
            p = rng.choice(len(L), size=ref_size, replace=False)
            P = EL[p]
            dist_sum[H] += knn_mean(EH, P)
            dist_n[H] += 1
            mask = np.ones(len(L), bool)
            mask[p] = False                     # queries exclude the pool (no self-match)
            q = np.flatnonzero(mask)
            if len(q):
                dist_sum[L[q]] += knn_mean(EL[q], P)
                dist_n[L[q]] += 1

    out = np.full(len(df), np.nan)
    ok = dist_n > 0
    out[ok] = dist_sum[ok] / dist_n[ok]
    return out


def fit_genre(d, genre, backend):
    import statsmodels.formula.api as smf
    sub = d[d["genre"] == genre].dropna(subset=["distance"]).copy()
    # "Mixed" (human-edited LLM text, replay dataset only) is scored but is
    # neither a pure human nor a pure LLM sentence: report it, model without it.
    mixed = sub[sub["label"] == "Mixed"]
    sub = sub[sub["label"] != "Mixed"]
    sub["part"] = pd.Categorical(sub["tertile"].map(PART),
                                 categories=["early", "mid", "late"])
    sub["source"] = np.where(sub["label"] == "Human", "Human", "LLM")
    sub["source"] = pd.Categorical(sub["source"], categories=["LLM", "Human"])

    print(f"\n{'='*72}\n[{genre} x {backend}]  n_sentences={len(sub):,} "
          f"sessions={sub['session'].nunique()} prompts={sub['prompt'].nunique()}")

    g = (sub.groupby(["part", "source"], observed=True)["distance"]
            .mean().unstack())
    g["gap"] = g["Human"] - g["LLM"]
    print("\n  descriptive mean distance / gap:")
    print(g.round(4).to_string())
    if len(mixed):
        m = mixed.groupby(mixed["tertile"].map(PART), observed=True)["distance"].mean()
        print(f"  Mixed (human-edited LLM text, n={len(mixed):,}): "
              + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    md = smf.mixedlm("distance ~ C(source)*C(part) + C(prompt)", sub,
                     groups=sub["session"])
    r = md.fit(method="lbfgs", maxiter=200)

    def row(name, label):
        if name not in r.params.index:
            return
        print(f"  {label:34} beta={r.params[name]:+.4f} "
              f"CI[{r.conf_int().loc[name,0]:+.4f},{r.conf_int().loc[name,1]:+.4f}] "
              f"p={r.pvalues[name]:.4g}")

    print("\n  mixed-effects (random intercept per session):")
    row("C(source)[T.Human]", "gap at EARLY")
    row("C(source)[T.Human]:C(part)[T.mid]", "gap change early->mid")
    row("C(source)[T.Human]:C(part)[T.late]", "gap change early->late")
    return r


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)
    rng = np.random.default_rng(SEED)
    print(f"computing per-sentence matched-baseline distances "
          f"(k={K_NN}, ref={N_REF}, reps={N_REP}) ...")
    df["distance"] = per_sentence_distances(df, emb, rng)
    df.drop(columns=["text"]).to_csv(DIR / f"distances_{backend}.csv", index=False)
    for genre in ("creative", "argumentative"):
        fit_genre(df, genre, backend)


if __name__ == "__main__":
    main()
