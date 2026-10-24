"""Every conservative control at once.

Each conservative control alone shrinks the group-level dimensional excess
(+2.3) toward the null-calibration ceiling; none were ever stacked. This
script stacks them all: the comparison is fully within-session (writer and
context fixed), restricted to complete sentences, with near-duplicates
collapsed at cosine 0.97, and with the control draw length-paired to the
writer's sentences (each Human sentence matched to the same session's
unused Dismissed sentence closest in word count). Reported: point estimate
and 95% CI over prompt groups, significant or not.

Usage: python check_all_controls.py [st|openai]
"""
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
N1 = 10
MIN_DIS = 44          # >= fit 33 (k=20 headroom) + hold N1
DEDUP = 0.97
REPS = 20


def fullsent_mask(texts):
    return texts.map(lambda t: len(str(t).split()) >= 3
                     and bool(re.match(r'^["“\(\'A-Z0-9]', str(t).strip()))
                     and bool(re.search(r'[.!?"”\')\]]$', str(t).strip())))


def greedy_keep(E, thr):
    keep = []
    sims = E @ E.T
    for i in range(len(E)):
        if not keep or sims[i, keep].max() < thr:
            keep.append(i)
    return np.array(keep)


def length_pair(wh, wd, rng):
    """Pair each Human sentence with the closest-length unused Dismissed one."""
    order = rng.permutation(len(wh))
    used = np.zeros(len(wd), dtype=bool)
    pairs = []
    for i in order:
        free = np.flatnonzero(~used)
        j = free[np.argmin(np.abs(wd[free] - wh[i]))]
        used[j] = True
        pairs.append((i, j))
    return pairs


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    ok = fullsent_mask(df["text"]).values
    df = df[ok]

    keep = np.zeros(len(df), dtype=bool)
    pos = {tuple(k): np.asarray(v) for k, v in
           df.groupby(["genre", "prompt", "label"]).indices.items()}
    for _, idx in pos.items():
        keep[idx[greedy_keep(emb[df.index.values[idx]], DEDUP)]] = True
    df = df[keep]

    rows = []
    for (genre, prompt, sess), sub in df.groupby(["genre", "prompt", "session"]):
        idx = sub.index.values
        lab = sub["label"].values
        H = idx[lab == "Human"]
        D = idx[lab == "Dismissed"]
        if len(H) < N1 or len(D) < MIN_DIS:
            continue
        w = sub["text"].astype(str).str.split().str.len().values
        wh = w[lab == "Human"]
        wd = w[lab == "Dismissed"]
        rng = np.random.default_rng(zlib.crc32(str(sess).encode()) % (2**31))
        dh, dl = [], []
        for _ in range(REPS):
            hi = rng.choice(len(H), N1, replace=False)
            pairs = length_pair(wh[hi], wd, rng)
            dj = np.array([j for _, j in pairs])
            rest = np.setdiff1d(np.arange(len(D)), dj)
            nfit = min(len(rest), 120)
            Rfit = emb[D[rest[rng.choice(len(rest), nfit, replace=False)]]]
            Xh = emb[H[hi]]
            Xd = emb[D[dj]]
            dh.append(G.deff(Xh) - G.deff(Xd))
            dl.append(G.leakage(Xh, Rfit) - G.leakage(Xd, Rfit))
        rows.append(dict(genre=genre, prompt=prompt, session=sess,
                         dw=float(np.mean(wh)) - float(np.mean(wd[np.array([j for _, j in pairs])])),
                         deff_ex=np.mean(dh), leak_ex=np.mean(dl)))
    d = pd.DataFrame(rows)
    d.to_csv(DIR / f"all_controls_{backend}.csv", index=False)
    g = d.groupby(["genre", "prompt"])[["deff_ex", "leak_ex"]].mean()
    print(f"[{backend}] all-controls-at-once: {len(d)} sessions, "
          f"{len(g)} prompt groups (fullsent + dedup{DEDUP} + within-session "
          f"+ length-paired, n={N1})")
    for m in ("deff_ex", "leak_ex"):
        x = g[m].dropna()
        ci = stats.t.ppf(.975, len(x) - 1) * stats.sem(x)
        t, p = stats.ttest_1samp(x, 0)
        print(f"  {m:8} {x.mean():+.4f}  95% CI [{x.mean()-ci:+.4f}, "
              f"{x.mean()+ci:+.4f}]  t({len(x)-1})={t:5.2f} p={p:.3g}  "
              f"({(x>0).sum()}/{len(x)} groups up; "
              f"{(d[m]>0).sum()}/{len(d)} sessions up)")


if __name__ == "__main__":
    main()
