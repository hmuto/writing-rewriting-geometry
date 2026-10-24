"""Regenerate CoAuthor suggestion pools with GPT-5 (modern-reference check).

For sampled sessions, reconstruct the document exactly as it stood at each
moment the writer asked GPT-3 for suggestions, give the same context to GPT-5,
and collect five continuations per query. The result is a "what a modern
assistant would have offered" reference pool, against which the 2021 writers'
sentences can be measured with the same three-forms pipeline.

Design (fixed):
  * up to SESS_PER_GROUP sessions per prompt group (seed 42),
  * up to QUERIES_PER_SESSION suggestion-open events per session, evenly spaced
    so all session thirds are covered,
  * n=5 completions per context, reasoning_effort="minimal",
  * temperature is NOT settable on this model (fixed at its default); CoAuthor
    used 0.30/0.75 with GPT-3 -- recorded as a limitation.

Usage: python gpt5_regen.py [--pilot]
Output: results/gpt5ref/gpt5_suggestions.csv
"""
import glob
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import temporal as T
import replay_extract as R

OUT_DIR = T.REPO / "results" / "gpt5ref"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "gpt5_suggestions.csv"

MODEL = "gpt-5"
SESS_PER_GROUP = 15
QUERIES_PER_SESSION = 10
N_SUGG = 5
SEED = 42
SENT_END = re.compile(r"(?<=[.!?])\s+")
SYSTEM = ("You are completing a piece of writing. Given the text so far, "
          "write one natural next sentence. Output only that sentence, "
          "with no preamble and no quotation marks.")

_usage = {"in": 0, "out": 0}
_lock = threading.Lock()


def contexts_for_session(fp):
    """(tertile, document text) at each sampled suggestion-open event."""
    events = [json.loads(l) for l in open(fp)]
    idxs = [i for i, e in enumerate(events)
            if e.get("eventName") == "suggestion-open"
            and isinstance(e.get("currentSuggestions"), list)
            and e["currentSuggestions"]]
    if not idxs:
        return []
    if len(idxs) > QUERIES_PER_SESSION:      # evenly spaced: cover all thirds
        pick = np.linspace(0, len(idxs) - 1, QUERIES_PER_SESSION).round().astype(int)
        idxs = [idxs[i] for i in sorted(set(pick))]
    out = []
    for i in idxs:
        text, _, _ = R.replay(events[:i])
        text = text.strip()
        if len(text) < 20:
            continue
        tert = min(int(i * T.N_TERTILES / max(len(events), 1)), T.N_TERTILES - 1)
        out.append((tert, text[-6000:]))     # cap pathological docs, keep tail
    return out


def query(client, doc):
    r = client.chat.completions.create(
        model=MODEL, n=N_SUGG, max_completion_tokens=80,
        reasoning_effort="minimal",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": doc}])
    with _lock:
        _usage["in"] += r.usage.prompt_tokens
        _usage["out"] += r.usage.completion_tokens
    out = []
    for ch in r.choices:
        s = (ch.message.content or "").strip().strip('"')
        s = SENT_END.split(s)[0].strip()     # first sentence only
        if len(s) >= T.MIN_TEXT_LENGTH:
            out.append(s)
    return out


def main():
    pilot = "--pilot" in sys.argv
    from openai import OpenAI
    client = OpenAI()
    rng = np.random.default_rng(SEED)

    # session -> (prompt, genre) from the corrected dataset
    meta = (pd.read_csv(T.REPO / "results" / "replay" / "sentences.csv")
              [["session", "prompt", "genre"]].drop_duplicates("session"))
    by_group = meta.groupby(["genre", "prompt"])["session"].apply(list)

    picked = []
    for (genre, prompt), sess in by_group.items():
        k = min(SESS_PER_GROUP, len(sess))
        for s in rng.choice(sess, size=k, replace=False):
            picked.append((s, prompt, genre))
    if pilot:
        picked = picked[:2]
    print(f"sessions: {len(picked)}  (pilot={pilot})")

    jobs = []
    for sid, prompt, genre in tqdm(picked, desc="contexts"):
        fp = T.DATA_DIR / f"{sid}.jsonl"
        if not fp.exists():
            continue
        for tert, doc in contexts_for_session(fp):
            jobs.append((sid, prompt, genre, tert, doc))
    print(f"contexts: {len(jobs)}  -> ~{len(jobs)} calls x {N_SUGG} completions")

    rows, fails = [], 0
    def work(j):
        sid, prompt, genre, tert, doc = j
        for attempt in range(3):
            try:
                return [(sid, prompt, genre, tert, "GPT5", s)
                        for s in query(client, doc)]
            except Exception:
                if attempt == 2:
                    return None
        return None

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="gpt-5"):
            got = f.result()
            if got is None:
                fails += 1
            else:
                rows += got

    df = pd.DataFrame(rows, columns=["session", "prompt", "genre",
                                     "tertile", "label", "text"])
    df.to_csv(OUT, index=False)
    print(f"\nfailed calls: {fails}/{len(jobs)}")
    print(f"suggestions kept: {len(df)}  "
          f"({df.groupby('genre').size().to_dict()})")
    print(f"tokens: in={_usage['in']:,} out={_usage['out']:,}")
    print(f"per prompt group:\n"
          f"{df.groupby(['genre','prompt']).size().describe().round(0).to_string()}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
