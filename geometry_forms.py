"""Three geometric forms of diversity expansion, applied to human-LLM co-writing.

Transposes the tail-reach / dimensional-expansion / directional-novelty
decomposition to CoAuthor. The reference distribution is what GPT-3 offered and
the writer threw away (Dismissed); the conditions are what the writer did:

    control   held-out half of Dismissed      (calibrates every measure at zero)
    Accepted  suggestions the writer took     -> human SELECTION
    Mixed     LLM text the writer rewrote     -> human EDITING
    Human     sentences the writer typed      -> human GENERATION

Accepted and Dismissed are both GPT-3 text and are disjoint, so Accepted vs the
held-out control isolates the writer's choice from the writer's words.

Measures (all at matched sample size, averaged over resamples):
    OOR   fraction of a condition lying farther than eps from the reference,
          eps = 2 x median nearest-neighbour distance inside the reference
    deff  participation-ratio effective dimension  (sum l)^2 / sum l^2
    dL    subspace leakage: own-centred variance outside the reference's top-k
          principal subspace, minus the held-out reference's own leakage

The unit of generalization is the prompt group (13 creative, 8 argumentative).

Usage: python geometry_forms.py [st|openai] [replay] [--lenmatch] [--binw N]
       [--nomixed] [--decorrelate] [--fullsent] [--menucap]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent
K_PCA = 20
N_MATCH = 200          # matched sample size per condition
N_MIN = 60             # skip a prompt x condition cell below this
N_REP = 20
SEED = 42
CONDS = ["control", "Accepted", "Mixed", "Human"]


# --------------------------------------------------------------------------- #
def deff(X):
    """Participation-ratio effective dimension of a centred point set."""
    Xc = X - X.mean(0)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    return lam.sum() ** 2 / (lam ** 2).sum()


def oor(X, R):
    """Fraction of X farther than eps from R; eps from R's own NN distances."""
    dr = 1.0 - (R @ R.T)
    np.fill_diagonal(dr, np.inf)
    eps = 2.0 * np.median(dr.min(axis=1))
    return float(((1.0 - (X @ R.T)).min(axis=1) > eps).mean()), eps


def leakage(X, R_fit, k=K_PCA):
    """Own-centred variance of X outside the top-k principal subspace of R_fit."""
    Rc = R_fit - R_fit.mean(0)
    V = np.linalg.svd(Rc, full_matrices=False)[2][:k]      # k x d
    Xc = X - X.mean(0)
    tot = (Xc ** 2).sum()
    return float(1.0 - ((Xc @ V.T) ** 2).sum() / tot) if tot > 0 else np.nan


def radius(X, R):
    """Mean distance to the reference centroid, in units of the reference's own."""
    c = R.mean(0)
    return float(np.linalg.norm(X - c, axis=1).mean() /
                 np.linalg.norm(R - c, axis=1).mean())


# --------------------------------------------------------------------------- #
def run_prompt(sub, emb, rng, per_session, conds=("Accepted", "Mixed", "Human"),
               n_min=N_MIN):
    """Return per-condition measures for one prompt group, or None if too small.

    per_session caps how many sentences one session may contribute to a sample.
    GPT-3 offers five suggestions per batch from the same context, so the
    Dismissed pool is clustered; capping breaks most of that dependence.
    """
    sess = sub["session"].values
    lab = sub["label"].values
    pos = {c: np.flatnonzero(lab == c) for c in list(conds) + ["Dismissed"]}
    row = sub.index.values

    def thin(p):
        """Keep at most per_session rows per session, in random order."""
        if per_session is None:
            return p
        seen, keep = {}, []
        for j in rng.permutation(len(p)):
            s = sess[p[j]]
            if seen.get(s, 0) < per_session:
                seen[s] = seen.get(s, 0) + 1
                keep.append(p[j])
        return np.array(keep, dtype=int)

    def cap(p):
        return len(thin(p)) if per_session else len(p)

    def draw(p, n):
        p = thin(p)
        return None if len(p) < n else row[p[rng.choice(len(p), n, replace=False)]]

    D = pos["Dismissed"]
    if cap(D) < 4 * n_min:
        return None
    n = min(N_MATCH, cap(D) // 4, *(cap(pos[c]) for c in conds))
    if n < n_min:
        return None
    conds = list(conds)

    all_conds = ["control"] + conds
    acc = {c: {m: [] for m in ("oor", "deff", "leak", "rad")} for c in all_conds}
    for _ in range(N_REP):
        # half of Dismissed fits the subspace, half calibrates every measure
        p = rng.permutation(len(D))
        sel = {"Rfit": draw(D[p[:len(D) // 2]], n),
               "control": draw(D[p[len(D) // 2:]], n)}
        for c in conds:
            sel[c] = draw(pos[c], n)
        if any(v is None for v in sel.values()):
            continue
        Rfit = emb[sel["Rfit"]]
        base = leakage(emb[sel["control"]], Rfit)
        for c in all_conds:
            X = emb[sel[c]]
            acc[c]["oor"].append(oor(X, Rfit)[0])
            acc[c]["deff"].append(deff(X))
            acc[c]["leak"].append(leakage(X, Rfit) - base)
            acc[c]["rad"].append(radius(X, Rfit))

    if not acc[all_conds[-1]]["oor"]:
        return None
    return {c: {m: float(np.mean(v)) for m, v in d.items()}
            for c, d in acc.items()}, n


def length_match(sub, rng, conds, bin_w=4):
    """Give every set the same word-count distribution.

    Rewritten sentences are longer than the others by construction (a writer who
    adds to a suggestion lengthens it), so length is a candidate explanation for
    the Mixed condition. Binning word counts and keeping the same number of
    sentences per bin from every set removes it.
    """
    w = sub["text"].astype(str).str.split().str.len().values
    b = (w // bin_w).astype(int)
    lab = sub["label"].values
    sets = list(conds) + ["Dismissed"]
    keep = []
    for bb in np.unique(b):
        m = b == bb
        pools = [np.flatnonzero(m & (lab == c)) for c in sets]
        k = min(len(p) for p in pools)
        if k == 0:
            continue
        keep += [rng.choice(p, k, replace=False) for p in pools]
    return sub.iloc[np.sort(np.concatenate(keep))] if keep else sub.iloc[:0]


# --------------------------------------------------------------------------- #
def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "replay"
    decor = "--decorrelate" in sys.argv
    lenm = "--lenmatch" in sys.argv
    binw = 4
    if "--binw" in sys.argv:
        binw = int(sys.argv[sys.argv.index("--binw") + 1])
    fullsent = "--fullsent" in sys.argv
    menucap = "--menucap" in sys.argv
    DIR = REPO / "results" / dataset
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)
    if menucap:
        # Five suggestions share one context, so cap the
        # Dismissed pool at ONE sentence per suggestion menu. Row alignment
        # with the embedding matrix is preserved (boolean filtering only).
        mid = pd.read_csv(DIR / "menu_ids.csv")["menu_id"].fillna("").values
        assert len(mid) == len(df)
        rng0 = np.random.default_rng(SEED)
        keep = np.ones(len(df), dtype=bool)
        dis = df["label"].values == "Dismissed"
        import collections
        by_menu = collections.defaultdict(list)
        for j in np.flatnonzero(dis):
            by_menu[mid[j]].append(j)
        for m, idxs in by_menu.items():
            if len(idxs) > 1:
                chosen = rng0.choice(len(idxs))
                for k, j in enumerate(idxs):
                    if k != chosen:
                        keep[j] = False
        df = df[keep]
    if fullsent:
        # Symmetric complete-sentence filter: every set keeps
        # only sentences that start like a sentence and end with terminal
        # punctuation. Row positions are preserved so embeddings still align.
        import re as _re
        ok = df["text"].astype(str).str.strip().map(
            lambda t: len(t.split()) >= 3
            and bool(_re.match(r'^["\u201c\(\'A-Z0-9]', t))
            and bool(_re.search(r'[.!?"\u201d\')\]]$', t)))
        df = df[ok.values]

    # Under decorrelation a session may contribute at most two sentences, which
    # shrinks every pool; Mixed is the smallest, so it is dropped for that check.
    nomixed = "--nomixed" in sys.argv
    per_session = 2 if decor else None
    conds = ["Accepted", "Human"] if (decor or nomixed) else ["Accepted", "Mixed", "Human"]
    n_min = 40 if decor else N_MIN
    if lenm:
        n_min = 40
    print(f"[{dataset} x {backend}]  per_session={per_session}  lenmatch={lenm}  "
          f"binw={binw}  k={K_PCA}  matched n<={N_MATCH}  reps={N_REP}  conds={conds}")

    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(SEED)
        if lenm:
            sub = length_match(sub, rng, conds, bin_w=binw)
            if len(sub) == 0:
                continue
        out = run_prompt(sub, emb, rng, per_session, conds, n_min)
        if out is None:
            continue
        res, n = out
        for c in res:
            rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c, **res[c]))
    d = pd.DataFrame(rows)
    if d.empty:
        print("no prompt group met the sample-size requirement")
        return
    suffix = ('_dec' if decor else '') + ('_lenmatch' if lenm else '')
    if lenm and binw != 4:
        suffix += f'_bw{binw}'
    if nomixed and not decor:
        suffix += '_nomixed'
    if fullsent:
        suffix += '_fullsent'
    if menucap:
        suffix += '_menucap'
    d.to_csv(DIR / f"geometry_{backend}{suffix}.csv", index=False)
    global CONDS
    CONDS = ["control"] + conds

    for genre in ("creative", "argumentative", "ALL"):
        g = d if genre == "ALL" else d[d["genre"] == genre]
        prompts = g["prompt"].nunique()
        if prompts < 3:
            continue
        print(f"\n{'='*74}\n[{genre}]  {prompts} prompt groups, "
              f"matched n = {g['n'].min()}-{g['n'].max()}")
        piv = g.pivot_table(index="prompt", columns="cond",
                            values=["oor", "deff", "leak", "rad"])
        name = {"oor": "out-of-reference rate", "deff": "effective dimension",
                "leak": "subspace leakage (adj.)", "rad": "radius ratio"}
        for m in ("oor", "deff", "leak", "rad"):
            line = "  ".join(f"{c}={piv[(m, c)].mean():.3f}" for c in CONDS)
            print(f"\n  {name[m]:26} {line}")
            for c in CONDS[1:]:
                a, b = piv[(m, c)], piv[(m, "control")]
                t, p = stats.ttest_rel(a, b)
                print(f"    {c:9} vs control: {a.mean()-b.mean():+.3f}  "
                      f"t({len(a)-1})={t:.2f}  p={p:.3g}  "
                      f"({(a > b).sum()}/{len(a)} prompts up)")
            t, p = stats.ttest_rel(piv[(m, "Human")], piv[(m, "Accepted")])
            print(f"    Human vs Accepted : "
                  f"{piv[(m,'Human')].mean()-piv[(m,'Accepted')].mean():+.3f}  "
                  f"t={t:.2f}  p={p:.3g}")


if __name__ == "__main__":
    main()
