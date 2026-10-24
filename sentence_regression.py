"""Sentence-level regression: length and session position as covariates,
and the writer as the unit of inference.

Per-sentence outcome: the share of a sentence's own-centred variance that
falls outside the reference's top-20 subspace (a per-sentence analogue of
leakage). Within each prompt group with at least 200 Dismissed sentences, a
random half of them is the reference and the other half the control.
Fits: (0) source + word count, prompt random intercept;
(1) as (0) plus session third; (2) OLS with prompt fixed effects and
cluster-robust SEs by session; (3) story prompts only, cluster-robust SEs by
writer (worker id from data/coauthor_metadata.csv).

Usage: python sentence_regression.py [st|openai]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
K = 20
SEED = 42


def build(df, emb):
    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(SEED)
        idx = sub.index.values
        lab = sub["label"].values
        D = idx[lab == "Dismissed"]
        if len(D) < 200:
            continue
        p = rng.permutation(len(D))
        fit, hold = D[p[:len(D) // 2]], D[p[len(D) // 2:]]
        Rc = emb[fit] - emb[fit].mean(0)
        V = np.linalg.svd(Rc, full_matrices=False)[2][:K]
        for source, sel in [("control", hold),
                            ("Accepted", idx[lab == "Accepted"]),
                            ("Mixed", idx[lab == "Mixed"]),
                            ("Human", idx[lab == "Human"])]:
            if len(sel) == 0:
                continue
            X = emb[sel] - emb[sel].mean(0)
            tot = (X ** 2).sum(1)
            out = 1.0 - ((X @ V.T) ** 2).sum(1) / np.maximum(tot, 1e-12)
            s = sub.loc[sel]
            for o, (ri, r) in zip(out, s.iterrows()):
                rows.append(dict(genre=genre, prompt=prompt, source=source,
                                 session=r["session"], tertile=r["tertile"],
                                 words=len(str(r["text"]).split()), leak=o))
    return pd.DataFrame(rows)


def coef(res, name="C(source, Treatment('control'))[T.Human]"):
    return res.params[name], res.bse[name]


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    meta = pd.read_csv(REPO / "data" / "coauthor_metadata.csv")
    worker = dict(zip(meta["session_id"].astype(str), meta["worker_id"].astype(str)))
    d = build(df, emb)
    d["worker"] = d["session"].astype(str).map(worker)
    print(f"[{backend}] {len(d):,} sentences in the regression frame")

    f0 = "leak ~ C(source, Treatment('control')) + words"
    f1 = f0 + " + C(tertile)"
    m0 = smf.mixedlm(f0, d, groups=d["prompt"]).fit(method="lbfgs", maxiter=200)
    b, se = coef(m0)
    print(f"  (0) main spec (prompt RE):               Human {b:+.4f} (se {se:.4f})")
    m1 = smf.mixedlm(f1, d, groups=d["prompt"]).fit(method="lbfgs", maxiter=200)
    b, se = coef(m1)
    print(f"  (1) + tertile (prompt RE):               Human {b:+.4f} (se {se:.4f})")
    KEY = "C(source, Treatment('control'))[T.Human]"
    o2 = smf.ols(f1 + " + C(prompt)", d).fit(
        cov_type="cluster", cov_kwds={"groups": d["session"]})
    b, se = coef(o2)
    print(f"  (2) prompt FE, SE clustered by session:  Human {b:+.4f} (se {se:.4f}) "
          f"p={o2.pvalues[KEY]:.3g}")
    ds = d[d["worker"].notna()].copy()
    o3 = smf.ols(f1 + " + C(prompt)", ds).fit(
        cov_type="cluster", cov_kwds={"groups": ds["worker"]})
    b, se = coef(o3)
    print(f"  (3) story side, SE clustered by writer ({ds['worker'].nunique()} "
          f"writers): Human {b:+.4f} (se {se:.4f}) "
          f"p={o3.pvalues[KEY]:.3g}")
    tert = [k for k in o2.params.index if "tertile" in k]
    print("  tertile terms (spec 2): " +
          "  ".join(f"{k.split('[')[-1][:-1]}={o2.params[k]:+.4f}" for k in tert))


if __name__ == "__main__":
    main()
