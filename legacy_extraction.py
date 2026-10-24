"""Legacy direct extraction (Supplementary Sections S.I and S.III).

This reproduces the earlier extraction that the paper corrects. It joins the
writer's raw text-insert events and re-splits them into sentences, without
replaying deletions, and infers the genre from the prompt's opening words.
Both steps are known to be wrong: abandoned keystrokes survive as typos, and
the genre rule puts two opinion prompts (132 sessions) in the story group.
It is kept only so that the "direct extraction" and "inferred genre label"
numbers of the supplement can be rebuilt.

Usage: python legacy_extraction.py
       python fullscale_embed.py st fullscale       (also: openai)
       python fullscale_analyze.py st fullscale     (also: openai)
Output: results/fullscale/sentences.csv
  columns = session, prompt, genre, tertile, label, text
Needs the raw CoAuthor logs in data/coauthor-v1.0/.
"""
import glob
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data" / "coauthor-v1.0"
OUT_DIR = REPO / "results" / "fullscale"
OUT = OUT_DIR / "sentences.csv"
MIN_TEXT_LENGTH = 20
N_TERTILES = 3
LABELS = (("acc", "Accepted"), ("dis", "Dismissed"), ("hum", "Human"))

# The genre rule of the earlier analysis (superseded by the metadata labels).
ARGUMENTATIVE_RE = re.compile(
    r"^\s*(should|is|are|how|do|does|can|will|would|has|have|was|were)\b.{0,200}\?",
    re.IGNORECASE | re.DOTALL,
)


def classify_prompt(p: str) -> str:
    if not p:
        return "unknown"
    return "argumentative" if ARGUMENTATIVE_RE.search(p) else "creative"


def prompt_key(p: str) -> str:
    return " ".join((p or "").split())[:80]


def split_sentences(text, min_length=MIN_TEXT_LENGTH):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = []
    buf = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        buf = (buf + " " + sent).strip() if buf else sent
        if len(buf) >= min_length:
            out.append(buf)
            buf = ""
    if buf and len(buf) >= 10:
        out.append(buf)
    return out


def extract_per_tertile(filepath: Path):
    """Return a dict: tertile_index (0/1/2) -> {"acc": [...], "dis": [...], "hum_raw": str}"""
    with open(filepath) as f:
        events = [json.loads(line) for line in f]

    # Get system-initialize prompt
    prompt = ""
    for e in events[:3]:
        if e.get("eventName") == "system-initialize":
            prompt = (e.get("currentDoc") or "").strip()
            break

    total = len(events)
    tertile_boundaries = [total // 3, 2 * total // 3]

    def tertile_of(idx):
        if idx < tertile_boundaries[0]:
            return 0
        elif idx < tertile_boundaries[1]:
            return 1
        else:
            return 2

    buckets = {i: {"acc": [], "dis": [], "hum_raw": []} for i in range(N_TERTILES)}

    current_batch = None
    current_batch_selected_idx = None
    current_batch_tertile = None

    def flush_batch():
        nonlocal current_batch, current_batch_selected_idx, current_batch_tertile
        if current_batch is None:
            return
        for cand in current_batch:
            if cand.get("index") == current_batch_selected_idx:
                continue
            text = (cand.get("trimmed") or cand.get("original") or "").strip()
            if text:
                buckets[current_batch_tertile]["dis"].append(text)
        current_batch = None
        current_batch_selected_idx = None
        current_batch_tertile = None

    for idx, e in enumerate(events):
        name = e.get("eventName")
        t = tertile_of(idx)

        if name == "suggestion-open":
            flush_batch()
            cands = e.get("currentSuggestions") or []
            if isinstance(cands, list) and cands:
                current_batch = cands
                current_batch_selected_idx = None
                current_batch_tertile = t

        elif name == "suggestion-select":
            hover = e.get("currentHoverIndex")
            if isinstance(hover, int) and current_batch:
                for cand in current_batch:
                    if cand.get("index") == hover:
                        text = (cand.get("trimmed") or cand.get("original") or "").strip()
                        if text:
                            buckets[current_batch_tertile]["acc"].append(text)
                        current_batch_selected_idx = hover
                        break

        elif name == "suggestion-close":
            flush_batch()

        elif name == "text-insert":
            if e.get("eventSource") == "user":
                td = e.get("textDelta")
                if isinstance(td, dict):
                    for op in td.get("ops", []):
                        if isinstance(op, dict) and isinstance(op.get("insert"), str):
                            buckets[t]["hum_raw"].append(op["insert"])

    flush_batch()

    # Convert hum_raw to sentences per tertile
    for i in range(N_TERTILES):
        joined = "".join(buckets[i]["hum_raw"])
        buckets[i]["hum"] = split_sentences(joined)
        del buckets[i]["hum_raw"]

    return prompt, buckets


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for fp in sorted(glob.glob(str(DATA_DIR / "*.jsonl"))):
        sid = Path(fp).stem
        try:
            prompt, buckets = extract_per_tertile(Path(fp))
        except Exception:
            continue
        genre = classify_prompt(prompt)
        if genre not in ("creative", "argumentative"):
            continue
        pk = prompt_key(prompt)
        for t in range(N_TERTILES):
            for key, label in LABELS:
                for txt in buckets[t][key]:
                    if len(txt) >= MIN_TEXT_LENGTH:
                        rows.append((sid, pk, genre, t, label, txt))
    df = pd.DataFrame(
        rows, columns=["session", "prompt", "genre", "tertile", "label", "text"])
    before = len(df)
    df = df.drop_duplicates(subset=["prompt", "label", "text"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"rows: {before} -> {len(df)} after dedup; sessions {df.session.nunique()}, "
          f"prompts {df.prompt.nunique()}")
    print(df.groupby(["genre", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
