"""Writer-cluster bootstrap for the story-side Human excess.

Resamples the 57 story writers with replacement; a writer drawn twice
contributes their sessions twice. Within each story prompt group the pipeline
is replayed and the replicate statistic is the across-group mean of the
Human excess. Percentile CIs over replicates.

Usage: python writer_bootstrap.py st|openai [B]
Output: results/replay/writer_boot_{backend}.npy
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import geometry_forms as G

REPO = Path(__file__).resolve().parent
REPS = 5


def excess_one_group(idx_D, idx_H, emb, rng, n):
    de, le = [], []
    for _ in range(REPS):
        p = rng.permutation(len(idx_D))
        half1, half2 = idx_D[p[: len(idx_D) // 2]], idx_D[p[len(idx_D) // 2:]]
        if len(half1) < n or len(half2) < n or len(idx_H) < n:
            return None
        try:
            Rfit = emb[half1[rng.choice(len(half1), n, replace=False)]]
            C = emb[half2[rng.choice(len(half2), n, replace=False)]]
            H = emb[idx_H[rng.choice(len(idx_H), n, replace=False)]]
            base = G.leakage(C, Rfit)
            de.append(G.deff(H) - G.deff(C))
            le.append(G.leakage(H, Rfit) - base)
        except np.linalg.LinAlgError:
            continue
    if not de:
        return None
    return float(np.mean(de)), float(np.mean(le))


def main():
    backend = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")
    meta = pd.read_csv(REPO / "data" / "coauthor_metadata.csv")
    s2w = dict(zip(meta["session_id"], meta["worker_id"]))

    story = df[df["genre"] == "creative"].copy()
    story["writer"] = story["session"].map(s2w)
    story = story.dropna(subset=["writer"])
    writers = story["writer"].unique()
    print(f"[{backend}] story side: {len(writers)} writers, "
          f"{story['session'].nunique()} sessions, B={B}, reps={REPS}")

    groups = {}
    for prompt, sub in story.groupby("prompt"):
        lab = sub["label"].values
        groups[prompt] = dict(
            writer=sub["writer"].values,
            D=sub.index.values[lab == "Dismissed"],
            H=sub.index.values[lab == "Human"],
            wD=sub["writer"].values[lab == "Dismissed"],
            wH=sub["writer"].values[lab == "Human"],
        )

    rng = np.random.default_rng(G.SEED)
    boots_d, boots_l = [], []
    for b in range(B):
        draw = rng.choice(writers, len(writers), replace=True)
        cnt = pd.Series(draw).value_counts()
        gd, gl = [], []
        for prompt, gr in groups.items():
            repD = np.repeat(gr["D"], [cnt.get(w, 0) for w in gr["wD"]])
            repH = np.repeat(gr["H"], [cnt.get(w, 0) for w in gr["wH"]])
            if len(repD) < 4 * 40 or len(repH) < 40:
                continue
            n = min(G.N_MATCH, len(repD) // 4, len(repH))
            out = excess_one_group(repD, repH, emb, rng, n)
            if out is not None:
                gd.append(out[0]); gl.append(out[1])
        if gd:
            boots_d.append(np.mean(gd)); boots_l.append(np.mean(gl))
        if (b + 1) % 50 == 0:
            print(f"  {b+1}/{B} done")

    bd, bl = np.array(boots_d), np.array(boots_l)
    np.save(REPO / "results" / "replay" / f"writer_boot_{backend}.npy",
            np.vstack([bd, bl]))
    for name, x in [("deff excess", bd), ("leak excess", bl)]:
        lo, hi = np.percentile(x, [2.5, 97.5])
        print(f"  {name}: mean {x.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P(<=0) {float((x <= 0).mean()):.3f}")


if __name__ == "__main__":
    main()
