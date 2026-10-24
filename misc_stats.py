"""Five supporting statistics quoted in the Supplementary Material (S.I, S.II, S.V).

1. Holm-adjusted p values for the six-test family: writing - rewriting
   effective dimension, writing - held-out reference effective dimension,
   and the writing orientation component, each under MiniLM (st) and OpenAI.
   Prompt-level paired / one-sample t tests on the saved per-prompt results
   (geometry_{enc}.csv, spectral_null_{enc}.csv).
2. Agreement between replayed session-level human-written shares and the
   corpus metadata (written_by_human, story sessions): Pearson r and mean
   absolute difference in percentage points. Replayed share = character-
   weighted mean of frac_h over the retained (non-Dismissed) sentences.
3. Share of sentences per set that fail the complete-sentence rule.
4. Within-set nearest-neighbour cosine distance of heavy (0.5 <= h < 0.8)
   and light (0.2 <= h < 0.5) rewrite bins, drawn exactly as in the
   authorship-bin analysis (geometry_dose.py: matched n <= 100, 20 draws per
   prompt group, seed 42), plus mean word counts of the two bins.
5. Mean share of a reference-fit sample's centred variance captured by its
   own top 20 principal directions, using the main-analysis draws
   (geometry_forms.py: matched n <= 200, 20 draws per prompt group, seed 42).

Usage: python misc_stats.py [REPO]   (default: this script's parent dir)
Needs: results/replay/{sentences.csv, emb_st.npy, emb_openai.npy,
       geometry_{st,openai}.csv, spectral_null_{st,openai}.csv},
       data/coauthor_metadata.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
SEED, N_REP, K = 42, 20, 20
ENCS = ("st", "openai")
EDGES = [0.0, 0.2, 0.5, 0.8, 1.001]
BINS = ["h<0.2", "light", "heavy", "h>=0.8"]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]).clip(max=1)
    out = np.empty_like(adj)
    out[o] = adj
    return out


def complete(t):
    t = str(t).strip()
    return (len(t.split()) >= 3 and bool(re.match(r'^["“\(\'A-Z0-9]', t))
            and bool(re.search(r'[.!?"”\')\]]$', t)))


def frame_draws(sub, conds, n_cap, n_min, rng):
    """Yield per-draw row selections in the same RNG order as G.run_prompt
    (per_session=None): permute Dismissed, draw R_fit, draw control, then
    each condition in order."""
    lab, row = sub["label"].values, sub.index.values
    pos = {c: np.flatnonzero(lab == c) for c in conds}
    D = np.flatnonzero(lab == "Dismissed")
    if len(D) < 4 * n_min:
        return
    n = min(n_cap, len(D) // 4, *(len(pos[c]) for c in conds))
    if n < n_min:
        return
    draw = lambda q: row[q[rng.choice(len(q), n, replace=False)]]
    for _ in range(N_REP):
        p = rng.permutation(len(D))
        sel = {"Rfit": draw(D[p[:len(D) // 2]]), "control": draw(D[p[len(D) // 2:]])}
        for c in conds:
            sel[c] = draw(pos[c])
        yield n, sel


def nn_dist(E):
    s = E @ E.T
    np.fill_diagonal(s, -np.inf)
    return float((1.0 - s.max(1)).mean())


def top_k_share(X, k=K):
    lam = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    return float(lam[:k].sum() / lam.sum())


def main():
    df = pd.read_csv(DIR / "sentences.csv")
    emb = {e: np.load(DIR / f"emb_{e}.npy") for e in ENCS}
    assert all(len(v) == len(df) for v in emb.values())

    # 1. Holm family -------------------------------------------------------
    names, ps = [], []
    for e in ENCS:
        g = pd.read_csv(DIR / f"geometry_{e}.csv")
        piv = g.pivot_table(index="prompt", columns="cond", values="deff")
        o = pd.read_csv(DIR / f"spectral_null_{e}.csv")
        orient = o.loc[o["cond"] == "Human", "dL_orient"]
        tests = [("deff writing-rewriting", stats.ttest_rel(piv["Human"], piv["Mixed"])),
                 ("deff writing-reference", stats.ttest_rel(piv["Human"], piv["control"])),
                 ("orientation writing", stats.ttest_1samp(orient, 0))]
        for nm, r in tests:
            names.append(f"{e:6s} {nm}")
            ps.append(r.pvalue)
    print("1. Holm family (6 tests, prompt-level t tests, 20 groups)")
    for nm, p, q in zip(names, ps, holm(ps)):
        print(f"   {nm:32s} raw p={p:.3g}  Holm p={q:.3g}")

    # 2. Replay vs metadata ------------------------------------------------
    s = df[df["label"] != "Dismissed"].copy()
    s["nc"] = s["text"].astype(str).str.len()
    s["hc"] = s["nc"] * s["frac_h"]
    agg = s.groupby("session")[["hc", "nc"]].sum()
    share = (100 * agg["hc"] / agg["nc"]).rename("replay")
    meta = pd.read_csv(REPO / "data" / "coauthor_metadata.csv")
    meta = meta.dropna(subset=["written_by_human"]).set_index("session_id")
    j = meta.join(share, how="inner")
    r = np.corrcoef(j["replay"], j["written_by_human"])[0, 1]
    mad = (j["replay"] - j["written_by_human"]).abs().mean()
    print(f"\n2. Replay vs metadata human share: sessions={len(j)} "
          f"(metadata rows={len(meta)})  r={r:.3f}  MAD={mad:.2f} pp")

    # 3. Fragment shares ---------------------------------------------------
    ok = df["text"].map(complete)
    print("\n3. Share failing the complete-sentence rule")
    for lab in ("Dismissed", "Accepted", "Mixed", "Human"):
        m = df["label"] == lab
        print(f"   {lab:9s} {100 * (1 - ok[m].mean()):5.1f}%  (n={m.sum()})")

    # 4. Heavy vs light rewrite NN distance (authorship-bin frame) ---------
    d = df.copy()
    m = d["label"] != "Dismissed"
    d.loc[m, "label"] = pd.cut(d.loc[m, "frac_h"], EDGES, right=False,
                               labels=BINS).astype(str)
    w = d["text"].astype(str).str.split().str.len()
    print("\n4. Heavy vs light rewrite bins: mean words heavy="
          f"{w[d.label == 'heavy'].mean():.1f}  light={w[d.label == 'light'].mean():.1f}")
    for e in ENCS:
        rows = []
        for (_, prompt), sub in d.groupby(["genre", "prompt"]):
            rng = np.random.default_rng(SEED)
            acc, n = {"heavy": [], "light": []}, None
            for n, sel in frame_draws(sub, BINS, 100, 50, rng):
                for c in acc:
                    acc[c].append(nn_dist(emb[e][sel[c]]))
            if n is not None:
                rows.append((n, np.mean(acc["heavy"]), np.mean(acc["light"])))
        a = np.array(rows)
        t = stats.ttest_rel(a[:, 1], a[:, 2])
        print(f"   {e:6s} groups={len(a)} n={a[:, 0].min():.0f}-{a[:, 0].max():.0f}  "
              f"heavy={a[:, 1].mean():.4f}  light={a[:, 2].mean():.4f}  "
              f"heavy<light in {(a[:, 1] < a[:, 2]).sum()}/{len(a)}  p={t.pvalue:.2g}")

    # 5. Variance captured by the reference sample's own top-20 directions -
    print(f"\n5. Top-{K} variance share of R_fit (main-analysis draws)")
    for e in ENCS:
        per_prompt, allv = [], []
        for (_, prompt), sub in df.groupby(["genre", "prompt"]):
            rng = np.random.default_rng(SEED)
            v = [top_k_share(emb[e][sel["Rfit"]])
                 for _, sel in frame_draws(sub, ["Accepted", "Mixed", "Human"], 200, 60, rng)]
            if v:
                per_prompt.append(np.mean(v))
                allv += v
        print(f"   {e:6s} groups={len(per_prompt)}  mean={100 * np.mean(per_prompt):.1f}%  "
              f"median over draws={100 * np.median(allv):.1f}%  "
              f"range of group means={100 * min(per_prompt):.1f}-{100 * max(per_prompt):.1f}%")


if __name__ == "__main__":
    main()
