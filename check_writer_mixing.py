"""Is the writer's excess a mixture-of-writers artifact?

The Human pool mixes ~57 writers while the reference is one generator, so a
mixture of writer-specific distributions could raise effective dimension and
leakage without any single writer contributing either. Two designs test this.

Part A holds the reference fixed and changes only HOW the Human sample is
drawn, at one small matched n:
    mixed     n sentences drawn across all sessions (the paper's design)
    1session  all n sentences from ONE randomly chosen session
    1writer   all n sentences from ONE randomly chosen writer (story side,
              where the released metadata carries worker ids)
If the excess needs the mixture, the 1session/1writer variants collapse to
the control; if it is writer-intrinsic, they stay positive.

Part B goes fully within-session: each session's Human sentences are measured
against that same session's Dismissed suggestions (fit/hold split inside the
session), so writer AND context are held constant.

Usage: python check_writer_mixing.py [st|openai]
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
N_SMALL = [8, 12]        # per-variant sample sizes (limited by one session)
N_FIT = 200              # reference-fit size, fixed across variants
N_REP = 50
SEED = 42


def load():
    df = pd.read_csv(DIR / "sentences.csv")
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)
    meta = pd.read_csv(REPO / "data" / "coauthor_metadata.csv")
    worker = dict(zip(meta["session_id"].astype(str),
                      meta["worker_id"].astype(str)))
    df["worker"] = df["session"].astype(str).map(worker)
    return backend, df, emb


def measures(X, Rfit, base_leak):
    return dict(deff=G.deff(X), leak=G.leakage(X, Rfit) - base_leak)


# --------------------------------------------------------------------------- #
def part_a(df, emb, n1):
    """Same reference, same n; only the Human draw changes."""
    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(SEED)
        idx = sub.index.values
        lab = sub["label"].values
        H = idx[lab == "Human"]
        D = idx[lab == "Dismissed"]
        if len(D) < 2 * N_FIT or len(H) < 4 * n1:
            continue
        hs = sub.loc[H, "session"].values
        hw = sub.loc[H, "worker"].values
        sess_ok = [s for s, c in zip(*np.unique(hs, return_counts=True)) if c >= n1]
        wrt_ok = [w for w, c in zip(*np.unique(hw[pd.notna(hw)].astype(str),
                                               return_counts=True)) if c >= n1]
        if len(sess_ok) < 5:
            continue
        acc = {v: {"deff": [], "leak": []} for v in
               ("control", "mixed", "1session", "1writer")}
        for _ in range(N_REP):
            p = rng.permutation(len(D))
            Rfit = emb[D[p[:N_FIT]]]
            hold = D[p[N_FIT:]]
            ctrl = emb[hold[rng.choice(len(hold), n1, replace=False)]]
            base = G.leakage(ctrl, Rfit)
            sel = {"control": ctrl,
                   "mixed": emb[H[rng.choice(len(H), n1, replace=False)]]}
            s = sess_ok[rng.integers(len(sess_ok))]
            pool = H[hs == s]
            sel["1session"] = emb[pool[rng.choice(len(pool), n1, replace=False)]]
            if wrt_ok:
                w = wrt_ok[rng.integers(len(wrt_ok))]
                pool = H[hw.astype(str) == w]
                sel["1writer"] = emb[pool[rng.choice(len(pool), n1, replace=False)]]
            for v, X in sel.items():
                m = measures(X, Rfit, base)
                acc[v]["deff"].append(m["deff"])
                acc[v]["leak"].append(m["leak"])
        for v, d in acc.items():
            if d["deff"]:
                rows.append(dict(genre=genre, prompt=prompt, n=n1, variant=v,
                                 n_sess=len(sess_ok), n_wrt=len(wrt_ok),
                                 deff=np.mean(d["deff"]), leak=np.mean(d["leak"])))
    return pd.DataFrame(rows)


def report_a(d, n1):
    print(f"\n{'='*74}\nPART A  (n={n1} per draw, fit={N_FIT}, reps={N_REP})")
    piv = d.pivot_table(index=["genre", "prompt"], columns="variant",
                        values=["deff", "leak"])
    for m in ("deff", "leak"):
        print(f"\n  {m}: control={piv[(m,'control')].mean():.4f}")
        for v in ("mixed", "1session", "1writer"):
            if (m, v) not in piv:
                continue
            x = (piv[(m, v)] - piv[(m, "control")]).dropna()
            t, p = stats.ttest_1samp(x, 0)
            print(f"    {v:9} - control: {x.mean():+.4f}  t({len(x)-1})={t:5.2f} "
                  f"p={p:.3g}  ({(x>0).sum()}/{len(x)} groups up)")
        for v in ("1session", "1writer"):
            if (m, v) not in piv:
                continue
            x = (piv[(m, "mixed")] - piv[(m, v)]).dropna()
            t, p = stats.ttest_1samp(x, 0)
            print(f"    mixed - {v:9}    : {x.mean():+.4f}  t({len(x)-1})={t:5.2f} "
                  f"p={p:.3g}   <- mixture contribution")


# --------------------------------------------------------------------------- #
def part_b(df, emb, n1=12, min_dis=48):
    """Fully within-session: H vs the same session's Dismissed."""
    rows = []
    for (genre, prompt, sess), sub in df.groupby(["genre", "prompt", "session"]):
        idx = sub.index.values
        lab = sub["label"].values
        H = idx[lab == "Human"]
        D = idx[lab == "Dismissed"]
        if len(H) < n1 or len(D) < min_dis:
            continue
        rng = np.random.default_rng(zlib.crc32(str(sess).encode()) % (2**31))
        dh, dl = [], []
        for _ in range(20):
            p = rng.permutation(len(D))
            nfit = min(len(D) - n1, 120)
            Rfit = emb[D[p[:nfit]]]
            ctrl = emb[D[p[nfit:nfit + n1]]]
            base = G.leakage(ctrl, Rfit)
            X = emb[H[rng.choice(len(H), n1, replace=False)]]
            dh.append(G.deff(X) - G.deff(ctrl))
            dl.append(G.leakage(X, Rfit) - base - 0.0)
        rows.append(dict(genre=genre, prompt=prompt, session=sess,
                         deff_ex=np.mean(dh), leak_ex=np.mean(dl)))
    return pd.DataFrame(rows)


def report_b(d):
    print(f"\n{'='*74}\nPART B  (within-session, n=12, {len(d)} sessions)")
    g = d.groupby(["genre", "prompt"])[["deff_ex", "leak_ex"]].mean()
    for m in ("deff_ex", "leak_ex"):
        x = g[m].dropna()
        t, p = stats.ttest_1samp(x, 0)
        up = (x > 0).sum()
        print(f"  {m:8} mean={x.mean():+.4f}  t({len(x)-1})={t:5.2f} p={p:.3g} "
              f"({up}/{len(x)} groups up; "
              f"{(d[m]>0).sum()}/{len(d)} sessions up)")


# --------------------------------------------------------------------------- #
def main():
    backend, df, emb = load()
    nw = df.loc[df["worker"].notna(), ["session", "worker"]].drop_duplicates()
    print(f"[{backend}] sessions with worker id: {nw['session'].nunique()}  "
          f"unique writers (story side): {nw['worker'].nunique()}")
    out = []
    for n1 in N_SMALL:
        d = part_a(df, emb, n1)
        d.to_csv(DIR / f"writer_mixing_parta_{backend}_n{n1}.csv", index=False)
        report_a(d, n1)
        out.append(d)
    b = part_b(df, emb)
    b.to_csv(DIR / f"writer_mixing_partb_{backend}.csv", index=False)
    report_b(b)


if __name__ == "__main__":
    main()
