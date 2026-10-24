"""Frame-matched null calibrations and the cross-context
redundancy diagnostic.

Part A: null calibration inside the within-session frame. The group-level
null ceiling (0.59 in effective dimension) was computed at matched n up to
200; comparing it with within-session excesses at n=12 mixes frames. Here
the full within-session pipeline is rerun B times with a second held-out
draw of the session's own Dismissed sentences in place of the writer's,
giving the null distribution of the frame itself.

Part B: the same for the all-controls-stacked frame (length-paired D-vs-D).

Part C: the cross-context redundancy diagnostic --- how much of the
control's own dimension at small n comes from spanning many contexts:
control draws across sessions versus within one session, at the same n.

Usage: python check_within_session_null.py [st|openai]
"""
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
N1 = 12
B = 30


def within_session_pass(df, emb, seed, null=False):
    """One full pass of the within-session analysis; returns group means."""
    rows = []
    for (genre, prompt, sess), sub in df.groupby(["genre", "prompt", "session"]):
        idx = sub.index.values
        lab = sub["label"].values
        H = idx[lab == "Human"]
        D = idx[lab == "Dismissed"]
        need = 48 + (N1 if null else 0)
        if len(H) < N1 or len(D) < need:
            continue
        rng = np.random.default_rng((zlib.crc32(str(sess).encode()) + seed) % (2**31))
        dh, dl = [], []
        for _ in range(10):
            p = rng.permutation(len(D))
            nfit = min(len(D) - N1 * (2 if null else 1), 120)
            Rfit = emb[D[p[:nfit]]]
            ctrl = emb[D[p[nfit:nfit + N1]]]
            base = G.leakage(ctrl, Rfit)
            if null:
                X = emb[D[p[nfit + N1:nfit + 2 * N1]]]
            else:
                X = emb[H[rng.choice(len(H), N1, replace=False)]]
            dh.append(G.deff(X) - G.deff(ctrl))
            dl.append(G.leakage(X, Rfit) - base)
        rows.append(dict(genre=genre, prompt=prompt,
                         deff=np.mean(dh), leak=np.mean(dl)))
    d = pd.DataFrame(rows)
    g = d.groupby(["genre", "prompt"])[["deff", "leak"]].mean()
    return float(g["deff"].mean()), float(g["leak"].mean())


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")

    # Part A: within-session frame null
    obs = within_session_pass(df, emb, seed=0, null=False)
    nulls = np.array([within_session_pass(df, emb, seed=1000 + b, null=True)
                      for b in range(B)])
    print(f"[{backend}] WITHIN-SESSION frame (n={N1}):")
    print(f"  observed:  deff {obs[0]:+.3f}  leak {obs[1]:+.4f}")
    for j, m in enumerate(("deff", "leak")):
        v = nulls[:, j]
        print(f"  null {m}: mean {v.mean():+.4f}  sd {v.std():.4f}  "
              f"max|.| {np.abs(v).max():.4f}  "
              f"observed = {abs(obs[j]) / max(np.abs(v).max(), 1e-9):.1f}x max-null, "
              f"{abs(obs[j] - v.mean()) / max(v.std(), 1e-9):.1f} null-SD from its mean")
    pd.DataFrame(nulls, columns=["deff", "leak"]).to_csv(
        DIR / f"within_session_null_{backend}.csv", index=False)

    # Part C: cross-context redundancy of the control at small n
    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        idx = sub.index.values
        lab = sub["label"].values
        D = idx[lab == "Dismissed"]
        sess = sub["session"].values
        ds = sess[lab == "Dismissed"]
        ok = [s for s, c in zip(*np.unique(ds, return_counts=True)) if c >= N1]
        if len(D) < 4 * N1 or len(ok) < 5:
            continue
        a, b_ = [], []
        for _ in range(30):
            a.append(G.deff(emb[D[rng.choice(len(D), N1, False)]]))
            s = ok[rng.integers(len(ok))]
            pool = D[ds == s]
            b_.append(G.deff(emb[pool[rng.choice(len(pool), N1, False)]]))
        rows.append(dict(prompt=prompt, mixed=np.mean(a), onesess=np.mean(b_)))
    d = pd.DataFrame(rows)
    print(f"\n[{backend}] CONTROL CONTEXT-SPAN at n={N1} ({len(d)} groups): "
          f"cross-session {d['mixed'].mean():.2f}  one-session "
          f"{d['onesess'].mean():.2f}  drop {(d['mixed']-d['onesess']).mean():.2f}")


if __name__ == "__main__":
    main()
