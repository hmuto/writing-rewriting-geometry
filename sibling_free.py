"""Sibling-free reference: remove the menu siblings of chosen suggestions.

Every menu offers five candidates. When the writer takes one, its four
same-context siblings enter the Dismissed reference, so Accepted sentences
have same-menu neighbours in the reference by construction and Human
sentences do not. This check drops from the reference every Dismissed
sentence that came from a menu the writer accepted from (identified by
suggestion-open event index, as in menu_ids.py), then reruns the standard
pipeline.

Usage: python sibling_free.py parse          # build accepted_menus.csv once
       python sibling_free.py [st|openai]    # run the analysis
       python sibling_free.py [st|openai] --menucap   # also one Dismissed per menu
Output: results/replay/geometry_{backend}_sibfree[_menucap].csv
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G
import temporal as T

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
MENUS = DIR / "accepted_menus.csv"


def parse():
    """Collect the menu ids (sid:event-index) where a selection occurred."""
    rows = []
    for fp in sorted(glob.glob(str(T.DATA_DIR / "*.jsonl"))):
        sid = Path(fp).stem
        try:
            events = [json.loads(l) for l in open(fp)]
        except Exception:
            continue
        bid, has_batch, accepted = None, False, False
        for i, e in enumerate(events):
            name = e.get("eventName")
            if name == "suggestion-open":
                if has_batch and accepted:
                    rows.append(f"{sid}:{bid}")
                c = e.get("currentSuggestions") or []
                has_batch, accepted = bool(isinstance(c, list) and c), False
                bid = i
            elif name == "suggestion-select":
                if isinstance(e.get("currentHoverIndex"), int) and has_batch:
                    accepted = True
            elif name == "suggestion-close":
                if has_batch and accepted:
                    rows.append(f"{sid}:{bid}")
                has_batch, accepted = False, False
        if has_batch and accepted:
            rows.append(f"{sid}:{bid}")
    pd.Series(rows, name="menu_id").to_csv(MENUS, index=False)
    print(f"accepted menus: {len(rows)}  -> {MENUS}")


def main(backend):
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)
    mid = pd.read_csv(DIR / "menu_ids.csv")["menu_id"].fillna("").values
    assert len(mid) == len(df)
    acc_menus = set(pd.read_csv(MENUS)["menu_id"].astype(str))
    dis = df["label"].values == "Dismissed"
    sib = dis & np.isin(mid, list(acc_menus))
    print(f"[{backend}] Dismissed rows {dis.sum()}, siblings of accepted menus "
          f"{sib.sum()} ({sib.mean()/dis.mean():.1%} of Dismissed) -> dropped")
    keep = ~sib
    if "--menucap" in sys.argv:
        # additionally keep one sentence per surviving menu
        rng0 = np.random.default_rng(G.SEED)
        import collections
        by_menu = collections.defaultdict(list)
        for j in np.flatnonzero(dis & keep):
            by_menu[mid[j]].append(j)
        for m_, idxs in by_menu.items():
            if len(idxs) > 1:
                chosen = rng0.choice(len(idxs))
                for k_, j in enumerate(idxs):
                    if k_ != chosen:
                        keep[j] = False
        print(f"  + menucap: Dismissed kept {int((dis & keep).sum())}")
    df2 = df[keep]

    rows = []
    for (genre, prompt), sub in df2.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(G.SEED)
        out = G.run_prompt(sub, emb, rng, None, ("Accepted", "Mixed", "Human"),
                           G.N_MIN)
        if out is None:
            continue
        res, n = out
        for c in res:
            rows.append(dict(genre=genre, prompt=prompt, n=n, cond=c, **res[c]))
    d = pd.DataFrame(rows)
    suf = "_menucap" if "--menucap" in sys.argv else ""
    d.to_csv(DIR / f"geometry_{backend}_sibfree{suf}.csv", index=False)

    piv = d.pivot_table(index="prompt", columns="cond",
                        values=["oor", "deff", "leak"])
    print(f"{d['prompt'].nunique()} prompt groups, matched n = "
          f"{d['n'].min()}-{d['n'].max()}")
    for m in ("oor", "deff", "leak"):
        print(f"\n  {m} (excess over held-out control)")
        for c in ("Accepted", "Mixed", "Human"):
            x = (piv[(m, c)] - piv[(m, "control")]).dropna()
            t, p = stats.ttest_1samp(x, 0)
            print(f"    {c:9} {x.mean():+.4f}  t({len(x)-1})={t:5.2f} p={p:.3g} "
                  f"({(x>0).sum()}/{len(x)} groups up)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "st"
    if arg == "parse":
        parse()
    else:
        main(arg)
