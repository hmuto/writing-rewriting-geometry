"""Sensitivity analyses of the scalar benchmark (Supplementary Section S.III).

Every variant re-estimates the nearest-neighbour distances on the same
cached embeddings and re-tests two quantities:

  gap      Human minus model distance in the first session third
  change   change of that gap from the first to the last third

Two readouts per variant:
  * mixed model  distance ~ C(source)*C(part) + C(prompt), session random intercept
  * session level  paired Wilcoxon on per-session mean gaps

Variants (16 per genre and embedding)
  k        : number of nearest model neighbours (1, 3, 5, 10, 20)
  ref      : size of the matched reference pool (20, 100; default 50)
  minlen   : minimum sentence length in characters (40, 80; default 20)
  cap      : at most 150 sentences per (prompt, third, label) cell
  pool     : model reference pool = Accepted only / Dismissed only
  notypo   : out-of-dictionary rate 0 or at most 10% (pyspellchecker)
  lencov   : word count added as a covariate to the mixed model
  lenmatch : Human and model sentences matched in 5-word bins

Usage: python scalar_sensitivity.py [st|openai] replay
Output: results/replay/sensitivity_{backend}.csv
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent
DATASET = sys.argv[2] if len(sys.argv) > 2 else "replay"
DIR = REPO / "results" / DATASET
N_REP = 20
SEED = 42
PART = {0: "early", 1: "mid", 2: "late"}
GENRES = ("creative", "argumentative")
WORD_BIN = 5  # words per bin for length matching


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #
def knn_prefix_means(Q, R, ks):
    """Mean distance to the k nearest rows of R, for every k in ks at once."""
    d = 1.0 - (Q @ R.T)
    kmax = min(max(ks), d.shape[1])
    part = np.partition(d, kmax - 1, axis=1)[:, :kmax]
    part.sort(axis=1)
    csum = np.cumsum(part, axis=1)
    return {k: csum[:, min(k, kmax) - 1] / min(k, kmax) for k in ks}


def distances(sub, E, rng, ks, n_ref):
    """Per-sentence matched-baseline distance, one array per k (NaN = not scored)."""
    acc = {k: np.zeros(len(sub)) for k in ks}
    n = np.zeros(len(sub))
    is_llm = sub["label"].isin(["Accepted", "Dismissed"]).values

    for _, idx in sub.groupby(["prompt", "tertile"]).indices.items():
        idx = np.asarray(idx)
        L, H = idx[is_llm[idx]], idx[~is_llm[idx]]
        if len(L) < 4 or len(H) == 0:
            continue
        ref = int(min(n_ref, len(L) // 2))
        if ref < max(ks) and ref < 5:
            continue
        EL, EH = E[L], E[H]
        for _ in range(N_REP):
            p = rng.choice(len(L), size=ref, replace=False)
            P = EL[p]
            for k, v in knn_prefix_means(EH, P, ks).items():
                acc[k][H] += v
            n[H] += 1
            mask = np.ones(len(L), bool)
            mask[p] = False                  # queries exclude the pool: no self-match
            q = np.flatnonzero(mask)
            if len(q):
                for k, v in knn_prefix_means(EL[q], P, ks).items():
                    acc[k][L[q]] += v
                n[L[q]] += 1

    ok = n > 0
    out = {}
    for k in ks:
        a = np.full(len(sub), np.nan)
        a[ok] = acc[k][ok] / n[ok]
        out[k] = a
    return out


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def fit(sub, genre, lencov=False):
    import statsmodels.formula.api as smf
    s = sub[(sub["genre"] == genre) & (sub["label"] != "Mixed")]
    s = s.dropna(subset=["distance"]).copy()
    if s["session"].nunique() < 20:
        return None
    s["part"] = pd.Categorical(s["tertile"].map(PART),
                               categories=["early", "mid", "late"])
    s["source"] = pd.Categorical(
        np.where(s["label"] == "Human", "Human", "LLM"), categories=["LLM", "Human"])

    f = "distance ~ C(source)*C(part) + C(prompt)" + (" + nwords" if lencov else "")
    r = smf.mixedlm(f, s, groups=s["session"]).fit(method="lbfgs", maxiter=200)
    a, b = "C(source)[T.Human]", "C(source)[T.Human]:C(part)[T.late]"

    g = s.groupby(["session", "tertile", "source"], observed=True)["distance"].mean()
    g = g.unstack("source").dropna()
    g["gap"] = g["Human"] - g["LLM"]
    w = g["gap"].unstack("tertile")
    w = w.dropna(subset=[c for c in (0, 2) if c in w.columns])
    delta = w[0] - w[2]

    return dict(
        n=len(s), sessions=s["session"].nunique(),
        rq1_beta=r.params[a], rq1_p=r.pvalues[a],
        rq2_beta=r.params[b], rq2_p=r.pvalues[b],
        s_n=len(w), s_rq1_gap=w[0].mean(), s_rq1_p=wilcoxon(w[0]).pvalue,
        s_rq2_delta=delta.mean(), s_rq2_p=wilcoxon(delta).pvalue,
    )


# --------------------------------------------------------------------------- #
# subsetting
# --------------------------------------------------------------------------- #
def cap_cells(df, rng, cap):
    keep = []
    for _, idx in df.groupby(["prompt", "tertile", "label"]).indices.items():
        idx = np.asarray(idx)
        keep.append(idx if len(idx) <= cap
                    else rng.choice(idx, size=cap, replace=False))
    return df.iloc[np.sort(np.concatenate(keep))]


def length_match(df, rng):
    """Down-sample so Human and LLM share a word-count distribution per cell."""
    src = np.where(df["label"].values == "Human", "H", "L")
    binid = (df["nwords"].values // WORD_BIN).astype(int)
    keep = []
    for _, idx in df.groupby(["prompt", "tertile"]).indices.items():
        idx = np.asarray(idx)
        for b in np.unique(binid[idx]):
            cell = idx[binid[idx] == b]
            h, l = cell[src[cell] == "H"], cell[src[cell] == "L"]
            m = min(len(h), len(l))
            if m == 0:
                continue
            keep.append(rng.choice(h, m, replace=False) if len(h) > m else h)
            keep.append(rng.choice(l, m, replace=False) if len(l) > m else l)
    return df.iloc[np.sort(np.concatenate(keep))] if keep else df.iloc[:0]


def unknown_rate(texts):
    """Fraction of tokens not in an English dictionary, per sentence.

    Applied to every label, not only Human: the dictionary also misses proper
    nouns, so a one-sided filter would remove typos from the human side and
    names from neither.
    """
    from spellchecker import SpellChecker
    sp = SpellChecker()
    tok = re.compile(r"[A-Za-z']+")
    out = np.zeros(len(texts))
    for i, t in enumerate(texts):
        w = [x.lower().strip("'") for x in tok.findall(str(t))]
        w = [x for x in w if len(x) > 2]
        if w:
            out[i] = len(sp.unknown(w)) / len(w)
    return out


# --------------------------------------------------------------------------- #
def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)
    df["nwords"] = df["text"].astype(str).str.split().str.len()
    df["clen"] = df["text"].astype(str).str.len()
    is_hum = (df["label"] == "Human").values

    print(f"[{backend}] {len(df):,} sentences, {df['session'].nunique()} sessions")
    print("computing out-of-dictionary rates (all labels) ...", flush=True)
    df["unk"] = unknown_rate(df["text"].tolist())
    print(df.groupby("label")["unk"].agg(clean=lambda x: (x == 0).mean(),
                                         mean="mean").round(3).to_string(), flush=True)
    clean0 = (df["unk"] == 0).values
    clean10 = (df["unk"] <= 0.10).values
    print(f"  kept: rate=0 -> {100*clean0.mean():.1f}% of all "
          f"({100*clean0[is_hum].mean():.1f}% of human); "
          f"rate<=10% -> {100*clean10.mean():.1f}% "
          f"({100*clean10[is_hum].mean():.1f}% of human)", flush=True)

    # (name, row-mask, k list, n_ref, lencov)
    plans = [
        ("k=1/3/5/10/20 x ref=50", None, [1, 3, 5, 10, 20], 50, False),
        ("ref=20", None, [5], 20, False),
        ("ref=100", None, [5], 100, False),
        ("minlen>=40ch", (df["clen"] >= 40).values, [5], 50, False),
        ("minlen>=80ch", (df["clen"] >= 80).values, [5], 50, False),
        ("pool=Accepted only", (df["label"] != "Dismissed").values, [5], 50, False),
        ("pool=Dismissed only", (df["label"] != "Accepted").values, [5], 50, False),
        ("spell-clean (rate=0)", clean0, [5], 50, False),
        ("spell-clean (rate<=10%)", clean10, [5], 50, False),
        ("+ word count covariate", None, [5], 50, True),
    ]

    rows = []
    for name, mask, ks, n_ref, lencov in plans:
        rng = np.random.default_rng(SEED)
        sub = (df if mask is None else df[mask]).reset_index(drop=True)
        E = emb[(df.index if mask is None else df.index[mask]).values]
        dd = distances(sub, E, rng, ks, n_ref)
        for k in ks:
            tag = name if len(ks) == 1 else f"k={k} (ref={n_ref})"
            sub2 = sub.assign(distance=dd[k])
            for genre in GENRES:
                r = fit(sub2, genre, lencov)
                if r:
                    rows.append(dict(variant=tag, genre=genre, **r))
                    print(f"  {tag:24} {genre:13} "
                          f"RQ1 {r['rq1_beta']:+.4f} p={r['rq1_p']:.2g} | "
                          f"RQ2 {r['rq2_beta']:+.4f} p={r['rq2_p']:.2g} | "
                          f"sess RQ2 p={r['s_rq2_p']:.2g}", flush=True)

    # variants that resample rows and so need their own subset
    for name in ("cap=150/cell", "length-matched"):
        rng = np.random.default_rng(SEED)
        d2 = df.copy()
        d2["pos"] = np.arange(len(d2))
        d2 = cap_cells(d2, rng, 150) if name.startswith("cap") else length_match(d2, rng)
        sub = d2.reset_index(drop=True)
        E = emb[sub["pos"].values]
        sub = sub.assign(distance=distances(sub, E, rng, [5], 50)[5])
        for genre in GENRES:
            r = fit(sub, genre)
            if r:
                rows.append(dict(variant=name, genre=genre, **r))
                print(f"  {name:24} {genre:13} "
                      f"RQ1 {r['rq1_beta']:+.4f} p={r['rq1_p']:.2g} | "
                      f"RQ2 {r['rq2_beta']:+.4f} p={r['rq2_p']:.2g} | "
                      f"sess RQ2 p={r['s_rq2_p']:.2g}", flush=True)

    out = pd.DataFrame(rows)
    out.insert(0, "backend", backend)
    out.to_csv(DIR / f"sensitivity_{backend}.csv", index=False)
    print(f"\nsaved -> sensitivity_{backend}.csv  ({len(out)} rows)")


if __name__ == "__main__":
    main()
