"""Collect every check of the writing-rewriting contrast into one table.

Reads the saved prompt-level estimates and reports, for each check and
embedding model, the mean difference in effective dimension (writing minus
rewriting, or writing minus a rewriting subtype), its 95% t interval across
prompt groups, the paired t-test p-value, and the number of positive groups.

Output: results/replay/wr_checks_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

D = Path(__file__).resolve().parent / "results" / "replay"


def summ(x):
    x = pd.Series(x).dropna()
    ci = stats.t.ppf(.975, len(x) - 1) * stats.sem(x)
    return dict(mean=x.mean(), lo=x.mean() - ci, hi=x.mean() + ci,
                p=stats.ttest_1samp(x, 0)[1], pos=int((x > 0).sum()), groups=len(x))


def geo(e, tag="", col="deff", genre=None):
    d = pd.read_csv(D / f"geometry_{e}{tag}.csv")
    if genre:
        d = d[d.genre == genre]
    g = d.groupby(["prompt", "cond"])[col].mean().unstack()
    return g["Human"] - g["Mixed"]


rows = []
for e in ("st", "openai"):
    add = lambda name, x: rows.append(dict(check=name, model=e, **summ(x)))
    add("main", geo(e))
    add("story prompts", geo(e, genre="creative"))
    add("opinion prompts", geo(e, genre="argumentative"))
    add("near-duplicates removed (0.95)", geo(e, "_dedup95"))
    add("one sentence per menu", geo(e, "_menucap"))
    add("complete sentences only", geo(e, "_fullsent"))
    nl = pd.read_csv(D / f"nonlinear_dimension_{e}.csv")
    for c, name in (("twonn", "TwoNN"), ("lb", "Levina-Bickel")):
        g = nl.groupby(["prompt", "cond"])[c].mean().unstack()
        add(name, g["Human"] - g["Mixed"])
    pr = pd.read_csv(D / f"pr_bias_corrected_{e}.csv")
    add("bias-corrected PR, main n", pr[pr["size"] == "main"].wr_corr)
    add("bias-corrected PR, n=50", pr[pr["size"] == "n50"].wr_corr)
    add("plug-in PR, n=50", pr[pr["size"] == "n50"].wr_plug)
    sm = pd.read_csv(D / f"session_matched_{e}.csv")
    add("same sessions, one each", sm[sm.m == 1].write_rewrite)
    add("same sessions, two each", sm[sm.m == 2].write_rewrite)
    st = pd.read_csv(D / f"mixed_subtype_geometry_{e}.csv")
    for s in ("completion", "interleaved", "append"):
        add(f"writing - {s}", st[st.subtype == s].human_sub)
    add("interleaved - completion", st[st.subtype == "interleaved_minus_completion"].human_sub)
    mo = pd.read_csv(D / f"mixed_origins_geometry_{e}.csv")
    add("selection (originals - reference)", mo.selection)
    add("editing (final - originals)", mo.editing)
    u = pd.read_csv(D / f"union_deff_{e}.csv").groupby(["prompt", "cond"]).u_deff.mean().unstack()
    add("union with reference", u["Human"] - u["Mixed"])

out = pd.DataFrame(rows)
out.to_csv(D / "wr_checks_summary.csv", index=False)
pd.set_option("display.width", 160)
print(out.round({"mean": 2, "lo": 2, "hi": 2}).to_string(index=False,
      formatters={"p": lambda v: f"{v:.2g}"}))
