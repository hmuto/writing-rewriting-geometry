"""Two-way (writer x prompt) cluster bootstrap, story side.

Resamples the 57 writers AND the 10 story prompts with replacement; the
replicate statistic is the mean over sampled prompts of the Human excess
(deff, raw leakage, orientation component).

--nodup: a writer drawn k>0 times contributes each sentence once, so no
sentence is duplicated inside a geometric sample (duplicates lower deff).
This drops writer weights and understates variance.
--wdistinct: keeps writer weights as sampling probabilities but draws
distinct sentences (weighted sampling without replacement).

Usage: python writer_prompt_bootstrap.py st|openai [B] [--nodup | --wdistinct] [--vs-mixed]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import geometry_forms as G
from orientation_frames import tail_share

REPO = Path(__file__).resolve().parent
REPS = 5


def one(idx_D, idx_H, emb, rng, n, wD=None, wH=None, idx_M=None, wM=None):
    """wD/wH: optional per-sentence weights (writer multiplicities). With
    weights, samples are drawn without replacement with probability
    proportional to weight, so writer weights are kept but no sentence
    appears twice in a sample. idx_M: if given, the comparison set is drawn
    from the rewritten (Mixed) pool instead of the held-out reference half,
    so the statistic is Write minus Rewrite."""
    def draw(idx, w, sub):
        if w is None:
            return idx[sub[rng.choice(len(sub), n, replace=False)]]
        ww = w[sub].astype(float)
        return idx[sub[rng.choice(len(sub), n, replace=False, p=ww / ww.sum())]]

    de, lo, orx = [], [], []
    allH = np.arange(len(idx_H))
    for _ in range(REPS):
        p = rng.permutation(len(idx_D))
        s1, s2 = p[: len(idx_D) // 2], p[len(idx_D) // 2:]
        if len(s1) < n or len(s2) < n or len(idx_H) < n or (
                idx_M is not None and len(idx_M) < n):
            return None
        try:
            Rfit = emb[draw(idx_D, wD, s1)]
            C = (emb[draw(idx_D, wD, s2)] if idx_M is None
                 else emb[draw(idx_M, wM, np.arange(len(idx_M)))])
            H = emb[draw(idx_H, wH, allH)]
            de.append(G.deff(H) - G.deff(C))
            lo.append(G.leakage(H, Rfit) - G.leakage(C, Rfit))
            orx.append((G.leakage(H, Rfit) - tail_share(H)) -
                       (G.leakage(C, Rfit) - tail_share(C)))
        except np.linalg.LinAlgError:
            continue
    if not de:
        return None
    return np.mean(de), np.mean(lo), np.mean(orx)


def main():
    backend = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 200
    nodup = "--nodup" in sys.argv
    wdist = "--wdistinct" in sys.argv
    vsmix = "--vs-mixed" in sys.argv
    df = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    emb = np.load(REPO / "results" / "replay" / f"emb_{backend}.npy")
    meta = pd.read_csv(REPO / "data" / "coauthor_metadata.csv")
    s2w = dict(zip(meta["session_id"], meta["worker_id"]))
    story = df[df["genre"] == "creative"].copy()
    story["writer"] = story["session"].map(s2w)
    story = story.dropna(subset=["writer"])
    writers = story["writer"].unique()
    prompts = story["prompt"].unique()
    print(f"[{backend}] 2-way bootstrap: {len(writers)} writers x "
          f"{len(prompts)} prompts, B={B}")

    groups = {}
    for prompt, sub in story.groupby("prompt"):
        lab = sub["label"].values
        groups[prompt] = dict(
            D=sub.index.values[lab == "Dismissed"],
            H=sub.index.values[lab == "Human"],
            M=sub.index.values[lab == "Mixed"],
            wD=sub["writer"].values[lab == "Dismissed"],
            wH=sub["writer"].values[lab == "Human"],
            wM=sub["writer"].values[lab == "Mixed"])

    rng = np.random.default_rng(G.SEED)
    boots = []
    for b in range(B):
        wd = rng.choice(writers, len(writers), replace=True)
        cnt = pd.Series(wd).value_counts()
        pd_draw = rng.choice(prompts, len(prompts), replace=True)
        gd = []
        for prompt in pd_draw:
            gr = groups[prompt]
            kD = np.array([cnt.get(w, 0) for w in gr["wD"]])
            kH = np.array([cnt.get(w, 0) for w in gr["wH"]])
            kM = np.array([cnt.get(w, 0) for w in gr["wM"]])
            wD = wH = wM = None
            if nodup or wdist:
                repD, repH, repM = gr["D"][kD > 0], gr["H"][kH > 0], gr["M"][kM > 0]
                if wdist:
                    wD, wH, wM = kD[kD > 0], kH[kH > 0], kM[kM > 0]
            else:
                repD, repH = np.repeat(gr["D"], kD), np.repeat(gr["H"], kH)
                repM = np.repeat(gr["M"], kM)
            if len(repD) < 160 or len(repH) < 40 or (vsmix and len(repM) < 40):
                continue
            n = min(G.N_MATCH, len(repD) // 4, len(repH))
            if vsmix:
                n = min(n, len(repM))
            out = one(repD, repH, emb, rng, n, wD, wH,
                      repM if vsmix else None, wM if vsmix else None)
            if out is not None:
                gd.append(out)
        if gd:
            boots.append(np.mean(gd, axis=0))
        if (b + 1) % 50 == 0:
            print(f"  {b+1}/{B}")
    A = np.array(boots)
    np.save(REPO / "results" / "replay" / f"boot2_{backend}{'_nodup' if nodup else ''}{'_wdistinct' if wdist else ''}{'_vsmixed' if vsmix else ''}.npy", A)
    for i, name in enumerate(["deff", "raw leak", "orientation"]):
        x = A[:, i]
        lo, hi = np.percentile(x, [2.5, 97.5])
        print(f"  {name}: mean {x.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(<=0) {float((x <= 0).mean()):.3f}")


if __name__ == "__main__":
    main()
