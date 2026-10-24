"""Null calibration for the all-controls-stacked frame,
and a writer-level jackknife for the dimensional excess.

Stacked null: the all-controls pipeline (check_all_controls.py) (within-session, complete sentences,
near-duplicates collapsed at 0.97, length-paired draws) rerun B times with a
second draw of the session's own Dismissed sentences in place of the
writer's, giving that frame's own null distribution.

Writer jackknife: the story-side main analysis recomputed 57 times, each
time deleting one writer's sessions from every pool, giving a jackknife CI
for the writer's dimensional excess with the writer as the unit.

Usage: python check_stacked_null.py [st|openai] [--skip-jack]
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import geometry_forms as G
from check_all_controls import fullsent_mask, greedy_keep, length_pair, DEDUP, N1

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
B = 30


def stacked_pass(df, emb, seed, null=False):
    rows = []
    for (genre, prompt, sess), sub in df.groupby(["genre", "prompt", "session"]):
        idx = sub.index.values
        lab = sub["label"].values
        H = idx[lab == "Human"]
        D = idx[lab == "Dismissed"]
        if len(H) < N1 or len(D) < 44 + (N1 if null else 0):
            continue
        w = sub["text"].astype(str).str.split().str.len().values
        wh = w[lab == "Human"]
        wd = w[lab == "Dismissed"]
        rng = np.random.default_rng((zlib.crc32(str(sess).encode()) + seed) % (2**31))
        dh, dl = [], []
        for _ in range(10):
            if null:
                qi = rng.choice(len(D), N1, replace=False)
                pairs = length_pair(wd[qi], np.where(
                    np.isin(np.arange(len(D)), qi), -999, wd), rng)
                X = emb[D[qi]]
            else:
                qi = rng.choice(len(H), N1, replace=False)
                pairs = length_pair(wh[qi], wd, rng)
                X = emb[H[qi]]
            dj = np.array([j for _, j in pairs])
            excl = np.union1d(dj, qi) if null else dj
            rest = np.setdiff1d(np.arange(len(D)), excl)
            if len(rest) < 33:
                continue
            nfit = min(len(rest), 120)
            Rfit = emb[D[rest[rng.choice(len(rest), nfit, replace=False)]]]
            Xd = emb[D[dj]]
            dh.append(G.deff(X) - G.deff(Xd))
            dl.append(G.leakage(X, Rfit) - G.leakage(Xd, Rfit))
        if dh:
            rows.append(dict(genre=genre, prompt=prompt,
                             deff=np.mean(dh), leak=np.mean(dl)))
    g = pd.DataFrame(rows).groupby(["genre", "prompt"])[["deff", "leak"]].mean()
    return float(g["deff"].mean()), float(g["leak"].mean())


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    raw = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")

    df = raw[fullsent_mask(raw["text"]).values]
    keep = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby(["genre", "prompt", "label"]).indices.items():
        idx = np.asarray(idx)
        keep[idx[greedy_keep(emb[df.index.values[idx]], DEDUP)]] = True
    df = df[keep]

    obs = stacked_pass(df, emb, seed=0, null=False)
    nulls = np.array([stacked_pass(df, emb, seed=2000 + b, null=True)
                      for b in range(B)])
    print(f"[{backend}] STACKED frame (n={N1}):")
    print(f"  observed:  deff {obs[0]:+.3f}  leak {obs[1]:+.4f}")
    for j, m in enumerate(("deff", "leak")):
        v = nulls[:, j]
        print(f"  null {m}: mean {v.mean():+.4f}  sd {v.std():.4f}  "
              f"max|.| {np.abs(v).max():.4f}  observed = "
              f"{abs(obs[j]) / max(np.abs(v).max(), 1e-9):.1f}x max-null")
    pd.DataFrame(nulls, columns=["deff", "leak"]).to_csv(
        DIR / f"stacked_null_{backend}.csv", index=False)

    if "--skip-jack" in sys.argv:
        return
    # writer jackknife on the story side (main pipeline, writer deleted)
    meta = pd.read_csv(REPO / "data" / "coauthor_metadata.csv")
    worker = dict(zip(meta["session_id"].astype(str), meta["worker_id"].astype(str)))
    story = raw[raw["genre"] == "creative"].copy()
    story["worker"] = story["session"].astype(str).map(worker)
    writers = sorted(story["worker"].dropna().unique())

    def story_mean(sub_df):
        vals = []
        for (genre, prompt), sub in sub_df.groupby(["genre", "prompt"]):
            rng = np.random.default_rng(G.SEED)
            out = G.run_prompt(sub, emb, rng, None, ("Human",), 40)
            if out is None:
                continue
            res, n = out
            vals.append(res["Human"]["deff"] - res["control"]["deff"])
        return np.mean(vals)

    full = story_mean(story)
    jk = np.array([story_mean(story[story["worker"] != w_]) for w_ in writers])
    nw = len(writers)
    se = np.sqrt((nw - 1) / nw * ((jk - jk.mean()) ** 2).sum())
    print(f"\n[{backend}] WRITER JACKKNIFE (story, {nw} writers): "
          f"deff excess {full:+.3f}, jackknife SE {se:.3f}, "
          f"95% CI [{full - 1.96 * se:+.3f}, {full + 1.96 * se:+.3f}]")


if __name__ == "__main__":
    main()
