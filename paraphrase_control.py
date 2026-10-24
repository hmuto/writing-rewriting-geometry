"""Paraphrase control.

Every sentence, model and human alike, is rewritten by one model into the
same plain register, and the pipeline is rerun. Survival of the excess is
evidence for a content reading; disappearance is ambiguous, because a
paraphraser also moves content toward its own distribution.

Stages:
  para     sample per-group pools (Dismissed<=800, Accepted<=400, Human<=400),
           paraphrase each unique sentence once with GPT-5 (minimal
           reasoning), checkpointing to results/replay/paraphrase.csv
           (calls the OpenAI API; the shipped paraphrase.csv is the one used)
  embed    embed the paraphrases with MiniLM and text-embedding-3-small
  analyze  rerun the pipeline on the paraphrased pools and report the excess
           over the paraphrased held-out control

Usage: python paraphrase_control.py {para|embed|analyze} [st|openai]
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from geometry_forms import deff, leakage, N_REP, SEED

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
PARA = DIR / "paraphrase.csv"
MODEL = "gpt-5"
CAPS = {"Dismissed": 800, "Accepted": 400, "Human": 400}
SYSTEM = ("You rewrite sentences. Rewrite the given sentence in plain, "
          "neutral English. Preserve the meaning completely. Change the "
          "wording and the sentence structure where natural. Output only "
          "the rewritten sentence.")


def sample_pools():
    df = pd.read_csv(DIR / "sentences.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        for label, cap in CAPS.items():
            pool = sub[sub.label == label]
            take = pool if len(pool) <= cap else pool.sample(cap, random_state=42)
            for i, r in take.iterrows():
                rows.append((i, prompt, genre, label, r.text))
    out = pd.DataFrame(rows, columns=["row", "prompt", "genre", "label", "text"])
    print(f"sampled {len(out)} sentences "
          f"({out.label.value_counts().to_dict()})")
    return out


def stage_para():
    from openai import OpenAI
    client = OpenAI()
    want = sample_pools()
    if PARA.exists():
        done = pd.read_csv(PARA)
        done_rows = set(done["row"])
        print(f"resume: {len(done_rows)} already paraphrased")
    else:
        done = pd.DataFrame(columns=["row", "prompt", "genre", "label",
                                     "text", "para"])
        done_rows = set()
    todo = want[~want.row.isin(done_rows)].reset_index(drop=True)
    print(f"to paraphrase: {len(todo)}")

    def work(r):
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, max_completion_tokens=120,
                    reasoning_effort="minimal",
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": str(r.text)}])
                p = (resp.choices[0].message.content or "").strip()
                if p:
                    return (r.row, r.prompt, r.genre, r.label, r.text, p)
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None

    buf, n_ok, n_fail = [], 0, 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(work, r): r.row for r in todo.itertuples()}
        for k, f in enumerate(as_completed(futs)):
            res = f.result()
            if res is None:
                n_fail += 1
            else:
                buf.append(res)
                n_ok += 1
            if len(buf) >= 500:
                done = pd.concat([done, pd.DataFrame(buf, columns=done.columns)])
                done.to_csv(PARA, index=False)
                buf = []
                print(f"  checkpoint {n_ok} ok / {n_fail} fail "
                      f"({k+1}/{len(todo)})", flush=True)
    if buf:
        done = pd.concat([done, pd.DataFrame(buf, columns=done.columns)])
        done.to_csv(PARA, index=False)
    print(f"done: {n_ok} ok, {n_fail} fail, total {len(done)}")


def stage_embed():
    df = pd.read_csv(PARA)
    texts = df["para"].astype(str).tolist()
    # MiniLM
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    e = m.encode(texts, batch_size=256, show_progress_bar=True,
                 normalize_embeddings=True)
    np.save(DIR / "paraphrase_emb_st.npy", e.astype(np.float32))
    # OpenAI
    from openai import OpenAI
    client = OpenAI()
    out = []
    for i in range(0, len(texts), 512):
        chunk = [t[:4000] for t in texts[i:i + 512]]
        r = client.embeddings.create(model="text-embedding-3-small",
                                     input=chunk)
        out.extend(d.embedding for d in r.data)
        print(f"  openai {i + len(chunk)}/{len(texts)}", flush=True)
    e2 = np.array(out, dtype=np.float32)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    np.save(DIR / "paraphrase_emb_openai.npy", e2)
    print("embeddings saved")


def stage_analyze(backend):
    from scipy import stats
    df = pd.read_csv(PARA).reset_index(drop=True)
    emb = np.load(DIR / f"paraphrase_emb_{backend}.npy")
    assert len(df) == len(emb)
    orig_emb = np.load(DIR / f"emb_{backend}.npy")

    rows = []
    for (genre, prompt), sub in df.groupby(["genre", "prompt"]):
        rng = np.random.default_rng(SEED)
        D = sub.index.values[(sub.label == "Dismissed").values]
        pools = {c: sub.index.values[(sub.label == c).values]
                 for c in ("Accepted", "Human")}
        n = min(200, len(D) // 4, *(len(p) for p in pools.values()))
        if n < 40:
            continue
        acc = {c: {"deff": [], "leak": []}
               for c in ("control", "Accepted", "Human")}
        ctrl_orig_deff = []
        for _ in range(N_REP):
            p = rng.permutation(len(D))
            half = len(D) // 2
            Rfit = emb[rng.choice(D[p[:half]], n, replace=False)]
            ctrl_idx = rng.choice(D[p[half:]], n, replace=False)
            ctrl = emb[ctrl_idx]
            base = leakage(ctrl, Rfit)
            acc["control"]["deff"].append(deff(ctrl))
            acc["control"]["leak"].append(0.0)
            ctrl_orig_deff.append(deff(orig_emb[df.loc[ctrl_idx, "row"].values]))
            for c in ("Accepted", "Human"):
                X = emb[rng.choice(pools[c], n, replace=False)]
                acc[c]["deff"].append(deff(X))
                acc[c]["leak"].append(leakage(X, Rfit) - base)
        rec = dict(genre=genre, prompt=prompt, n=n,
                   ctrl_orig_deff=float(np.mean(ctrl_orig_deff)))
        for c in acc:
            rec[f"{c}_deff"] = float(np.mean(acc[c]["deff"]))
            rec[f"{c}_leak"] = float(np.mean(acc[c]["leak"]))
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(DIR / f"paraphrase_geometry_{backend}.csv", index=False)

    print(f"[paraphrase control x {backend}]  {len(res)} prompt groups")
    print(f"  control deff: original texts {res.ctrl_orig_deff.mean():.1f} -> "
          f"paraphrased {res.control_deff.mean():.1f} "
          f"(homogenization gauge)")
    for c in ("Accepted", "Human"):
        d_leak = res[f"{c}_leak"]
        d_deff = res[f"{c}_deff"] - res["control_deff"]
        t1, p1 = stats.ttest_1samp(d_leak, 0)
        t2, p2 = stats.ttest_1samp(d_deff, 0)
        print(f"  {c:9} leak {d_leak.mean():+.4f} (p={p1:.2g}, "
              f"{(d_leak > 0).sum()}/{len(res)} up)   "
              f"deff {d_deff.mean():+.3f} (p={p2:.2g}, "
              f"{(d_deff > 0).sum()}/{len(res)} up)")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "para":
        stage_para()
    elif stage == "embed":
        stage_embed()
    elif stage == "analyze":
        stage_analyze(sys.argv[2] if len(sys.argv) > 2 else "st")
