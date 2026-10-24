"""Supplementary figures (session thirds, residual spectrum, moderators).
Reads saved result CSVs, plus the OpenAI embeddings for the residual
spectrum. Figures 1-3 of the main text come from make_paper_figures.py.

Usage: python make_figures.py            # writes figs/fig4-fig6 (.png, .pdf)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
OUT = REPO / "figs"
OUT.mkdir(exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                        "Liberation Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})
C = {"st": "#3a6ea5", "openai": "#c9622a", "grey": "#8a8a8a", "hl": "#2b7a4b"}
ENC = {"st": "MiniLM (384-d)", "openai": "OpenAI (1536-d)"}
MEAS = [("oor", "Out-of-reference rate"),
        ("deff", "Effective dimension"),
        ("leak", "Orientation difference\n(orientation component)")]


def ci(a):
    a = np.asarray(a, float)
    return stats.t.ppf(.975, len(a) - 1) * stats.sem(a)


def excess(d, meas, conds):
    """Per-prompt excess of each condition over its control."""
    p = d.pivot_table(index="prompt", columns="cond", values=meas)
    return pd.DataFrame({c: p[c] - p["control"] for c in conds if c in p})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figs/{name}.png / .pdf")


# --------------------------------------------------------------------------- #




def fig4():
    """Human directional excess persists; accepted excess declines."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharex=True,
                             sharey=True, layout="constrained")
    parts = ["early", "mid", "late"]
    for ax, be in zip(axes, ("st", "openai")):
        d = pd.read_csv(DIR / f"temporal_orient_full_{be}.csv")
        ok = d.groupby("prompt")["tertile"].nunique() == 3
        d = d[d["prompt"].isin(ok[ok].index)]
        piv = d.pivot_table(index="prompt", columns=["cond", "tertile"],
                            values="dL_orient")
        n_groups = len(piv)
        ex = {c: np.array([piv[(c, t)].mean() for t in (0, 1, 2)])
              for c in ("Human", "Accepted")}
        er = {c: np.array([ci(piv[(c, t)]) for t in (0, 1, 2)])
              for c in ("Human", "Accepted")}
        x = np.arange(3)
        margin = 0.020
        ax.axhspan(ex["Human"][0] - margin, ex["Human"][0] + margin,
                   color=C["hl"], alpha=.10, zorder=1,
                   label="equivalence margin (fixed 0.020)")
        for c, col, mk in (("Human", C["hl"], "o"), ("Accepted", C["openai"], "s")):
            ax.errorbar(x, ex[c], yerr=er[c], color=col, lw=2, marker=mk, ms=5,
                        capsize=3, zorder=3,
                        label="Write (Human)" if c == "Human" else "Choose (Accepted)")
        ax.axhline(0, color="#444", lw=.9)
        ax.set_xticks(x)
        ax.set_xticklabels(["early", "mid", "late"])
        ax.set_title(ENC[be])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False,
               fontsize=8)
    fig.supylabel("orientation difference, excess over control",
                  fontsize=9)
    fig.suptitle("The writer's orientation difference persists during the session  "
                 f"(story prompts, {n_groups} groups)",
                 fontsize=10.5)
    save(fig, "fig4_temporal")


def fig5():
    """Residual spectrum: the new directions are diffuse, and so is the control's."""
    K = 20
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / "emb_openai.npy")
    rng = np.random.default_rng(42)

    def spectrum(genre, source):
        res = []
        for _, sub in df[df.genre == genre].groupby("prompt"):
            D = sub.index[sub.label == "Dismissed"].values
            X = sub.index[sub.label == source].values
            if source == "control":
                p = rng.permutation(len(D))
                D, X = D[p[:len(p) // 2]], D[p[len(p) // 2:]]
            if len(D) < 4 * K or len(X) < 50:
                continue
            Rc = emb[D] - emb[D].mean(0)
            V = np.linalg.svd(Rc, full_matrices=False)[2][:K]
            Xc = emb[X] - emb[X].mean(0)
            res.append(Xc - (Xc @ V.T) @ V)
        R = np.vstack(res)
        R -= R.mean(0)
        lam = np.linalg.eigvalsh(R.T @ R)[::-1]
        return lam / lam.sum(), R.shape[1] - K

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True,
                             layout="constrained")
    for ax, genre in zip(axes, ("creative", "argumentative")):
        for source, col, lab in (("Human", C["hl"], "human residual"),
                                 ("control", C["grey"], "LLM residual (control)")):
            s, dim = spectrum(genre, source)
            ax.plot(np.arange(1, 41), s[:40], color=col, lw=1.8, label=lab)
        ax.axhline(1 / dim, color=C["openai"], ls="--", lw=1.1,
                   label=f"isotropic ({1/dim:.2%})")
        ax.set_yscale("log")
        ax.set_xlabel("residual component (rank)")
        ax.set_title({"creative": "story prompts",
                      "argumentative": "opinion prompts"}[genre])
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.supylabel("share of residual variance", fontsize=9)
    fig.suptitle("Human and model residuals show similar variance spectra",
                 fontsize=10.5)
    save(fig, "fig5_residual_spectrum")


def fig6():
    """Expansion does not depend on the writer's engagement style."""
    mods = [("dismiss_rate", "dismissal rate"), ("edit_rate", "editing rate"),
            ("human_share", "own-writing share")]
    QN = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    # No guide lines from Fig. 2: this analysis uses a session-disjoint
    # reference (half the sessions), which raises leakage systematically, so the
    # two levels are not comparable. Only the flatness within a panel is.
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.0), sharey=True,
                             layout="constrained")
    for ax, (m, nice) in zip(axes, mods):
        for be in ("st", "openai"):
            f = DIR / f"geometry_moderator_{m}_{be}.csv"
            if not f.exists():
                continue
            e = excess(pd.read_csv(f), "leak", QN)
            x = np.arange(4)
            ax.errorbar(x, [e[q].mean() for q in QN],
                        yerr=[ci(e[q]) for q in QN], color=C[be], lw=1.8,
                        marker="o", ms=4, capsize=3, label=ENC[be])
        ax.axhline(0, color="#444", lw=.9)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Q1\nlow", "Q2", "Q3", "Q4\nhigh"])
        ax.set_xlabel(nice)
        ax.set_xlim(-.35, 3.35)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.supylabel("raw subspace leakage, excess over control", fontsize=9)
    fig.suptitle("Raw subspace leakage across levels of writing activity",
                 fontsize=10.5)
    save(fig, "fig6_moderators")






if __name__ == "__main__":
    for f in (fig4, fig5, fig6):
        print(f.__name__)
        f()
