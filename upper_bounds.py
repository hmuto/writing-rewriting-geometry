"""One-sided 95% upper bounds for the dimensional excess of choosing and
rewriting (Supplementary table of upper bounds).

Across prompt groups, the one-sided 95% upper confidence bound of each
condition's dimensional excess is computed for each analysis frame and
compared with the human excess in the same frame. Reads the geometry_*.csv
files written by geometry_forms.py, sibling_free.py and near_duplicates.py.

Usage: python upper_bounds.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parent / "results" / "replay"

FRAMES = [
    ("main", "geometry_{b}.csv"),
    ("length-matched (4-set)", "geometry_{b}_lenmatch.csv"),
    ("complete-sentence", "geometry_{b}_fullsent.csv"),
    ("sibling-free", "geometry_{b}_sibfree.csv"),
    ("menu-capped", "geometry_{b}_menucap.csv"),
    ("near-dup collapsed", "geometry_{b}_dedup95.csv"),
]


def upper95(x):
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() + stats.t.ppf(0.95, len(x) - 1) * se


def main():
    for b, bname in [("st", "MiniLM"), ("openai", "OpenAI")]:
        print(f"=== {bname}: one-sided 95% upper bounds, deff excess ===")
        for fname, tmpl in FRAMES:
            f = R / tmpl.format(b=b)
            if not f.exists():
                continue
            d = pd.read_csv(f)
            piv = d.pivot_table(index=["genre", "prompt"], columns="cond", values="deff")
            if "control" not in piv or "Human" not in piv:
                continue
            hum = (piv["Human"] - piv["control"]).mean()
            out = []
            for c in ["Accepted", "Mixed"]:
                if c not in piv:
                    continue
                x = (piv[c] - piv["control"]).dropna()
                out.append(f"{c}: mean {x.mean():+.2f}, UB {upper95(x):+.2f} (g={len(x)})")
            print(f"  {fname:24s} Human {hum:+.2f} | " + " | ".join(out))
        print()


if __name__ == "__main__":
    main()
