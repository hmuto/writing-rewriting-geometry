"""Orientation decomposition in three further frames.

  gpt5      -- gpt5ref frame (ref = GPT-5 pool; conds GPT3/Accepted/Mixed/Human)
  temporal  -- story prompts, per session third (ref/cond from same third)
  pure      -- main frame with Human restricted to frac_h >= 0.999

Usage: python orientation_frames.py gpt5|temporal|pure st|openai
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent


def tail_share(X, k=20):
    Xc = X - X.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    lam = sv ** 2
    return float(lam[k:].sum() / lam.sum())


def decompose(sub, emb, conds, n_cap, n_min, rng):
    lab = sub["label"].values
    row = sub.index.values
    pos = {c: np.flatnonzero(lab == c) for c in conds + ["Dismissed"]}
    D = pos["Dismissed"]
    if len(D) < 4 * n_min:
        return None
    n = min(n_cap, len(D) // 4, *(len(pos[c]) for c in conds))
    if n < n_min:
        return None
    acc = {c: {m: [] for m in ("obs", "spec", "deff")} for c in ["control"] + conds}
    for _ in range(G.N_REP):
        p = rng.permutation(len(D))
        half1, half2 = D[p[: len(D) // 2]], D[p[len(D) // 2:]]
        if len(half1) < n or len(half2) < n:
            continue
        try:
            sel = {"Rfit": row[half1[rng.choice(len(half1), n, replace=False)]],
                   "control": row[half2[rng.choice(len(half2), n, replace=False)]]}
            for c in conds:
                sel[c] = row[pos[c][rng.choice(len(pos[c]), n, replace=False)]]
            Rfit = emb[sel["Rfit"]]
            for c in ["control"] + conds:
                X = emb[sel[c]]
                acc[c]["obs"].append(G.leakage(X, Rfit))
                acc[c]["spec"].append(tail_share(X, G.K_PCA))
                acc[c]["deff"].append(G.deff(X))
        except np.linalg.LinAlgError:
            continue
    if not acc[conds[-1]]["obs"]:
        return None
    mean = {c: {m: float(np.mean(v)) for m, v in d.items()} for c, d in acc.items()}
    out = {}
    for c in conds:
        out[c] = dict(
            dL_obs=mean[c]["obs"] - mean["control"]["obs"],
            dL_spec=mean[c]["spec"] - mean["control"]["spec"],
            d_deff=mean[c]["deff"] - mean["control"]["deff"],
        )
        out[c]["dL_orient"] = out[c]["dL_obs"] - out[c]["dL_spec"]
    return out, n


def report(d, conds, tag):
    print(f"[{tag}] groups={d['prompt'].nunique()}")
    for c in conds:
        s = d[d["cond"] == c]
        line = f"  {c:10s}"
        for m in ["dL_obs", "dL_orient", "d_deff"]:
            x = s[m].values
            t, p = stats.ttest_rel(x, np.zeros(len(x)))
            line += f"  {m} {x.mean():+.4f} (p={p:.1g}, {int((x>0).sum())}/{len(x)})"
        print(line)


def main():
    mode, backend = sys.argv[1], sys.argv[2]
    if mode == "gpt5":
        df = pd.read_csv(REPO / "results" / "gpt5ref" / "sentences.csv")
        emb = np.load(REPO / "results" / "gpt5ref" / f"emb_{backend}.npy")
        conds = ["GPT3", "Accepted", "Mixed", "Human"]
        rows = []
        for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
            rng = np.random.default_rng(G.SEED)
            out = decompose(sub, emb, conds, 120, 40, rng)
            if out is None:
                continue
            res, n = out
            for c in res:
                rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c, **res[c]))
        d = pd.DataFrame(rows)
        d.to_csv(REPO / "results" / "gpt5ref" / f"orient_{backend}.csv", index=False)
        report(d, conds, f"gpt5 x {backend}")

    elif mode == "temporal":
        df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
        emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")
        df = df[df["genre"] == "creative"]
        conds_t = ["Accepted", "Human"]
        rows = []
        for prompt, subp in df.groupby("prompt"):
            for t in (0, 1, 2):
                sub = subp[subp["tertile"] == t]
                rng = np.random.default_rng(G.SEED)
                out = decompose(sub, emb, conds_t, 100, 40, rng)
                if out is None:
                    continue
                res, n = out
                for c in conds_t:
                    rows.append(dict(prompt=prompt, tertile=t, n=n, cond=c,
                                     **res[c]))
        d = pd.DataFrame(rows)
        d.to_csv(REPO / "results" / "replay" /
                 f"temporal_orient_full_{backend}.csv", index=False)
        d = d[d["cond"] == "Human"]
        piv = d.pivot_table(index="prompt", columns="tertile", values="dL_orient")
        piv = piv.dropna()
        early, late = piv[0].values, piv[2].values
        ch = late - early
        t_, p = stats.ttest_rel(late, early)
        print(f"[temporal x {backend}] groups={len(piv)}  "
              f"orient early {early.mean():+.4f}  late {late.mean():+.4f}  "
              f"change {ch.mean():+.4f} (p={p:.2g})")
        margin = early.mean() / 2
        se = ch.std(ddof=1) / np.sqrt(len(ch))
        p1 = 1 - stats.t.cdf((ch.mean() + margin) / se, len(ch) - 1)
        p2 = stats.t.cdf((ch.mean() - margin) / se, len(ch) - 1)
        print(f"  TOST at half-early margin ({margin:+.4f}): p={max(p1,p2):.2g}")

    elif mode == "pure":
        df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
        emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")
        lab = df["label"].copy()
        drop = (lab == "Human") & (df["frac_h"] < 0.999)
        df = df[~drop.values]
        rows = []
        for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
            rng = np.random.default_rng(G.SEED)
            out = decompose(sub, emb, ["Human"], G.N_MATCH, G.N_MIN, rng)
            if out is None:
                continue
            res, n = out
            rows.append(dict(genre=genre, prompt=prompt, n=n, cond="Human",
                             **res["Human"]))
        d = pd.DataFrame(rows)
        d.to_csv(REPO / "results" / "replay" / f"pure_h1_{backend}.csv", index=False)
        report(d, ["Human"], f"pure h=1.0 x {backend}")


if __name__ == "__main__":
    main()
