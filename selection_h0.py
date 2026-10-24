"""Selection check with human-character-free Accepted sentences only.

Accepted sentences are final-document text and may contain up to 20% human
characters (81% contain none; mean share 1.7%). Here the Accepted condition
is restricted to frac_h == 0 and the deff / orientation contrast is
recomputed.

Usage: python selection_h0.py st|openai
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

from orientation_frames import decompose
import geometry_forms as G

backend = sys.argv[1]
df = pd.read_csv("results/replay/sentences.csv")
emb = np.load(f"results/replay/emb_{backend}.npy")
keep = ~((df["label"] == "Accepted") & (df["frac_h"] > 0))
rows = []
for (genre, prompt), sub in df[keep].groupby(["genre", "prompt"]):
    rng = np.random.default_rng(G.SEED)
    out = decompose(sub, emb, ["Accepted"], G.N_MATCH, G.N_MIN, rng)
    if out is None:
        continue
    res, n = out
    rows.append(dict(prompt=prompt, n=n, **res["Accepted"]))
d = pd.DataFrame(rows)
line = f"[{backend}] pure-selection (frac_h=0) groups={len(d)}"
for m in ("d_deff", "dL_orient"):
    x = d[m].values
    t, p = stats.ttest_rel(x, np.zeros(len(x)))
    line += f"  {m} {x.mean():+.3f} (p={p:.2g}, {int((x>0).sum())}/{len(x)})"
print(line)
