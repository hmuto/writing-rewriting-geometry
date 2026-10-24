"""Is the Human dimensional excess just scatter from incoherent or disfluent text?

Two per-sentence covariates:
    coh   cosine between a sentence and the centroid of its session's final
          document (Human/Accepted/Mixed sentences, excluding the sentence
          itself). Dismissed suggestions are compared with the same centroid,
          since they were generated from that session's text. Low values mean
          off-topic relative to the writer's own document.
    lppl  GPT-2 log-perplexity of the sentence (fluency / typo proxy).

The Human set is then resampled within each prompt group so that its joint
(coh x lppl) quintile distribution matches the Dismissed reference, and the
geometry pipeline is rerun. If the excess were produced by off-topic or
disfluent sentences, it should disappear under this matching.

Usage: python coherence_check.py st|openai
Output: results/replay/ppl_gpt2.npy (cached), coherence_check_{backend}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import geometry_forms as G

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
PPL = DIR / "ppl_gpt2.npy"
NQ = 5


def gpt2_logppl(texts, batch=64):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    out = np.empty(len(texts))
    order = np.argsort([len(t) for t in texts])
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            ids = order[s:s + batch]
            enc = tok([tok.bos_token + texts[i] for i in ids], return_tensors="pt",
                      padding=True, truncation=True, max_length=128).to(dev)
            logits = model(**enc).logits[:, :-1]
            tgt = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            nll = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), tgt, reduction="none")
            out[ids] = ((nll * mask).sum(1) / mask.sum(1)).cpu().numpy()
    return out


def coherence(df, emb):
    coh = np.full(len(df), np.nan)
    doc = df["label"].isin(["Human", "Accepted", "Mixed"]).values
    for _, idx in df.groupby("session").indices.items():
        d = idx[doc[idx]]
        if len(d) < 3:
            continue
        tot = emb[d].sum(0)
        for j in idx:
            c = tot - emb[j] if doc[j] else tot
            k = len(d) - 1 if doc[j] else len(d)
            c = c / k
            coh[j] = emb[j] @ c / (np.linalg.norm(c) + 1e-12)
    return coh


def match_to_reference(sub, rng):
    """Resample Human so its joint covariate bins follow the Dismissed shares."""
    lab = sub["label"].values
    b = sub["bin"].values
    D, H = lab == "Dismissed", lab == "Human"
    share = pd.Series(b[D]).value_counts(normalize=True)
    hcount = pd.Series(b[H]).value_counts()
    n = min(hcount.get(k, 0) / p for k, p in share.items() if p > 0)
    keep = list(np.flatnonzero(~H))
    for k, p in share.items():
        m = int(np.floor(n * p))
        pool = np.flatnonzero(H & (b == k))
        if m > 0:
            keep += list(rng.choice(pool, m, replace=False))
    return sub.iloc[np.sort(keep)]


def main():
    backend = sys.argv[1]
    df = pd.read_csv(DIR / "sentences.csv")
    emb = np.load(DIR / f"emb_{backend}.npy")
    assert len(df) == len(emb)
    if PPL.exists():
        lppl = np.load(PPL)
    else:
        lppl = gpt2_logppl(df["text"].astype(str).tolist())
        np.save(PPL, lppl)
    df["lppl"] = lppl
    df["coh"] = coherence(df, emb)

    print(f"[{backend}] medians by label")
    print(df.groupby("label")[["coh", "lppl"]].median().round(3).to_string())

    df = df[df["coh"].notna()].copy()
    ref = df[df["label"] == "Dismissed"]
    qc = np.quantile(ref["coh"], np.linspace(0, 1, NQ + 1)[1:-1])
    qp = np.quantile(ref["lppl"], np.linspace(0, 1, NQ + 1)[1:-1])
    df["bin"] = np.digitize(df["coh"], qc) * NQ + np.digitize(df["lppl"], qp)

    rows = []
    for variant in ("unmatched", "matched"):
        res = []
        for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
            rng = np.random.default_rng(G.SEED)
            if variant == "matched":
                sub = match_to_reference(sub, rng)
            out = G.run_prompt(sub, emb, rng, None, ["Human"], G.N_MIN)
            if out is None:
                continue
            r, n = out
            res.append(dict(prompt=prompt, n=n, d=r["Human"]["deff"] - r["control"]["deff"]))
        r = pd.DataFrame(res)
        t, p = stats.ttest_1samp(r["d"], 0)
        half = stats.t.ppf(0.975, len(r) - 1) * r["d"].std(ddof=1) / np.sqrt(len(r))
        rows.append(dict(variant=variant, groups=len(r), n_min=r["n"].min(),
                         n_max=r["n"].max(), mean=r["d"].mean(),
                         ci_lo=r["d"].mean() - half, ci_hi=r["d"].mean() + half, pos=int((r["d"] > 0).sum()), t=t, p=p))
        if variant == "matched":
            h = df[df["label"] == "Human"]
            print(f"  Human coh/lppl medians before matching: "
                  f"{h['coh'].median():.3f} / {h['lppl'].median():.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(DIR / f"coherence_check_{backend}.csv", index=False)
    print("Human - control effective dimension across prompt groups")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
