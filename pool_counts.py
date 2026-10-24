"""Staged accounting for the Dismissed pool (Supplementary Section S.I).

Stages: candidates offered (metadata num_query x 5) -> minus accepted
events (num_selected) -> minus empty or short candidates (MIN_TEXT_LENGTH
filter, counted by replaying suggestion batches) -> minus prompt material
and exact duplicates (= final sentences.csv count). Needs the raw logs.

Usage: python pool_counts.py
"""

from pathlib import Path

import pandas as pd

import temporal as T
from menu_ids import dismissed_with_menu
import json
import glob

REPO = Path(__file__).resolve().parent


def main():
    meta = pd.read_csv(REPO / "data" / "coauthor_metadata.csv")
    offered = int(meta["num_query"].sum()) * 5
    selected = int(meta["num_selected"].sum())
    print(f"candidates offered (num_query x 5): {offered}")
    print(f"accepted events (num_selected):     {selected}")
    print(f"dismissed events (offered - sel):   {offered - selected}")

    n_postfilter = 0
    files = sorted(glob.glob(str(REPO / "data" / "coauthor-v1.0" / "*.jsonl")))
    print(f"replaying {len(files)} session logs for the length filter...")
    for i, f in enumerate(files):
        sid = Path(f).stem
        with open(f) as fh:
            events = [json.loads(l) for l in fh]
        out = dismissed_with_menu(sid, events, len(events))
        n_postfilter += len(out)
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(files)}")
    print(f"after empty/short filter:           {n_postfilter}")

    sent = pd.read_csv(REPO / "results" / "replay" / "sentences.csv")
    final = int((sent["label"] == "Dismissed").sum())
    print(f"after prompt-removal + dedup:       {final}")


if __name__ == "__main__":
    main()
