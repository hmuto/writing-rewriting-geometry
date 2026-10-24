"""Null calibration for the excess measures of the main analysis.

For each prompt group, draw three disjoint samples of the matched size from
Dismissed: R_fit, control, and a pseudo-condition. The pseudo-condition is
measured exactly like a real condition (leakage and effective dimension
minus the control's). Averaging over N_REP redraws per group and then across
the 20 groups replicates the headline statistic under the null; repeating
the pipeline B times gives its null distribution.

Usage: python null_calibration.py [st|openai] [B]
Output: results/replay/null_calibration_{backend}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geometry_forms import deff, leakage, N_MATCH, N_REP, SEED

REPO = Path(__file__).resolve().parent


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    DIR = REPO / "results" / "replay"
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")

    # Per-group Dismissed row positions and the matched n actually used in the
    # main analysis (min over conditions), read from the stored results.
    main_res = pd.read_csv(DIR / f"geometry_{backend}.csv")
    n_by_prompt = main_res.groupby("prompt")["n"].first()

    groups = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        if prompt not in n_by_prompt.index:
            continue
        D = sub.index.values[(sub["label"] == "Dismissed").values]
        n = int(n_by_prompt[prompt])
        if len(D) < 3 * n:
            n = len(D) // 3
        groups.append((prompt, D, n))
    print(f"[{backend}] {len(groups)} groups, B={B} null replications")

    rng = np.random.default_rng(SEED)
    null_leak, null_deff = [], []
    for b in range(B):
        g_leak, g_deff = [], []
        for prompt, D, n in groups:
            leak_acc, deff_acc = [], []
            for _ in range(N_REP):
                p = rng.permutation(len(D))
                Rfit = emb[D[p[:n]]]
                ctrl = emb[D[p[n:2 * n]]]
                pseu = emb[D[p[2 * n:3 * n]]]
                base = leakage(ctrl, Rfit)
                leak_acc.append(leakage(pseu, Rfit) - base)
                deff_acc.append(deff(pseu) - deff(ctrl))
            g_leak.append(np.mean(leak_acc))
            g_deff.append(np.mean(deff_acc))
        null_leak.append(np.mean(g_leak))
        null_deff.append(np.mean(g_deff))
        if (b + 1) % 50 == 0:
            print(f"  {b+1}/{B}")

    null_leak = np.array(null_leak)
    null_deff = np.array(null_deff)
    out = pd.DataFrame({"null_leak": null_leak, "null_deff": null_deff})
    out.to_csv(DIR / f"null_calibration_{backend}.csv", index=False)

    for name, arr in [("leakage", null_leak), ("deff", null_deff)]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"{name:8} null mean={arr.mean():+.5f}  sd={arr.std(ddof=1):.5f}  "
              f"95% band=[{lo:+.5f}, {hi:+.5f}]  max|null|={np.abs(arr).max():.5f}")


if __name__ == "__main__":
    main()
