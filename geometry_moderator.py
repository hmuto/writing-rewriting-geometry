"""Engagement moderators (exploratory; Supplementary Section S.VIII).

Session-level moderators:
    dismiss_rate  refused / everything the model offered
    edit_rate     reworked / model text kept
    human_share   own sentences / sentences in the document
    n_sent        sentences in the session (control for length)

Sessions are split into quartiles of the moderator within each prompt group,
and the human sentences of each quartile are measured at matched sample size.
The test is the trend across quartiles, paired across prompt groups. Each
repetition splits the prompt's sessions in half: one half builds the
reference and its control, the other supplies the human sentences.

Usage: python geometry_moderator.py [st|openai] [replay]
Output: results/replay/geometry_moderator_{moderator}_{backend}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
MODS = ["dismiss_rate", "edit_rate", "human_share"]
QN = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
N_MATCH = 100
N_MIN = 50
N_REP = 30


def session_moderators(df):
    c = (df.groupby(["session", "genre", "prompt", "label"]).size()
           .unstack(fill_value=0).reset_index())
    for col in ("Accepted", "Dismissed", "Human", "Mixed"):
        if col not in c:
            c[col] = 0
    kept = c["Accepted"] + c["Mixed"]
    c["dismiss_rate"] = c["Dismissed"] / (c["Dismissed"] + kept).replace(0, np.nan)
    c["edit_rate"] = c["Mixed"] / kept.replace(0, np.nan)
    c["human_share"] = c["Human"] / (c["Human"] + kept).replace(0, np.nan)
    c["n_sent"] = c[["Accepted", "Human", "Mixed"]].sum(axis=1)
    return c


def run_prompt(sub, emb, rng, quart):
    """Measure human sentences of each quartile against a session-disjoint reference."""
    sess = sub["session"].values
    lab = sub["label"].values
    row = sub.index.values
    uniq = np.unique(sess)
    if len(uniq) < 16:
        return None
    hum = {q: np.flatnonzero((lab == "Human") &
                             np.isin(sess, quart[quart == q].index.values))
           for q in QN}
    dis = np.flatnonzero(lab == "Dismissed")
    if min(len(v) for v in hum.values()) < N_MIN or len(dis) < 4 * N_MIN:
        return None

    acc = {c: {m: [] for m in ("deff", "leak", "oor")} for c in ["control"] + QN}
    n_used = []
    for _ in range(N_REP):
        p = rng.permutation(len(uniq))
        ref_s = set(uniq[p[:len(p) // 2]])            # reference half
        ref = dis[[s in ref_s for s in sess[dis]]]
        meas = {q: v[[s not in ref_s for s in sess[v]]] for q, v in hum.items()}
        if len(ref) < 4 * N_MIN or min(len(v) for v in meas.values()) < N_MIN:
            continue
        n = min(N_MATCH, len(ref) // 2, *(len(v) for v in meas.values()))
        n_used.append(n)
        q2 = rng.permutation(len(ref))
        pick = lambda a: row[a[rng.choice(len(a), n, replace=False)]]
        Rfit = emb[pick(ref[q2[:len(q2) // 2]])]
        sel = {"control": pick(ref[q2[len(q2) // 2:]])}
        for q in QN:
            sel[q] = pick(meas[q])
        base = G.leakage(emb[sel["control"]], Rfit)
        for c in ["control"] + QN:
            X = emb[sel[c]]
            acc[c]["deff"].append(G.deff(X))
            acc[c]["leak"].append(G.leakage(X, Rfit) - base)
            acc[c]["oor"].append(G.oor(X, Rfit)[0])

    if not n_used:
        return None
    return ({c: {m: float(np.mean(v)) for m, v in d.items()}
             for c, d in acc.items()}, int(np.mean(n_used)))


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "replay"
    DIR = REPO / "results" / dataset
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)
    mod = session_moderators(df).set_index("session")

    print(f"[{dataset} x {backend}] session moderators, "
          f"{len(mod)} sessions, matched n<={N_MATCH}, reps={N_REP}")
    print(mod[MODS + ["n_sent"]].describe().loc[
        ["mean", "std", "25%", "50%", "75%"]].round(3).to_string())

    for m in MODS:
        rows = []
        for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
            s = mod.loc[mod.index.intersection(sub["session"].unique()), m].dropna()
            if s.nunique() < 4 or len(s) < 16:
                continue
            try:
                q = pd.qcut(s, 4, labels=QN, duplicates="drop")
            except ValueError:
                continue
            if q.nunique() < 4:
                continue
            rng = np.random.default_rng(G.SEED)
            out = run_prompt(sub, emb, rng, q)
            if out is None:
                continue
            res, n = out
            span = (s.groupby(q, observed=True).mean().reindex(QN))
            for c in res:
                rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c,
                                 mod_value=span.get(c, np.nan), **res[c]))
        d = pd.DataFrame(rows)
        if d.empty:
            print(f"\n### {m}: no prompt group met the sample-size requirement")
            continue
        d.insert(0, "moderator", m)
        d.to_csv(DIR / f"geometry_moderator_{m}_{backend}.csv", index=False)

        print(f"\n{'='*76}\n### moderator: {m}   "
              f"({d['prompt'].nunique()} prompt groups, n={d['n'].min()}-{d['n'].max()})")
        vals = d[d.cond.isin(QN)].groupby("cond", observed=True)["mod_value"].mean()
        print("   quartile means: " + "  ".join(f"{q}={vals.get(q, np.nan):.2f}"
                                                for q in QN))
        for meas, name in (("leak", "directional novelty"),
                           ("deff", "effective dimension")):
            piv = d.pivot_table(index="prompt", columns="cond", values=meas)
            print(f"\n  {name} (excess over control)")
            for q in QN:
                ex = piv[q] - piv["control"]
                t, p = stats.ttest_1samp(ex, 0)
                print(f"    {q:10} {ex.mean():+.3f}  t({len(ex)-1})={t:5.2f} "
                      f"p={p:.3g}  ({(ex > 0).sum()}/{len(ex)} positive)")
            x = np.arange(4)
            sl = [np.polyfit(x, [piv.loc[pr, q] - piv.loc[pr, "control"]
                                for q in QN], 1)[0] for pr in piv.index]
            t, p = stats.ttest_1samp(sl, 0)
            print(f"    trend: slope/quartile={np.mean(sl):+.4f} "
                  f"t({len(sl)-1})={t:.2f} p={p:.3g} "
                  f"({sum(v > 0 for v in sl)}/{len(sl)} prompts rising)")


if __name__ == "__main__":
    main()
