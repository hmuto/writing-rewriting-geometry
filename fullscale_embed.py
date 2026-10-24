"""Embed the sentence set. Cached to .npy next to sentences.csv.

Usage:
  python fullscale_embed.py st     [replay|gpt5ref|fullscale]  # MiniLM (local)
  python fullscale_embed.py openai [replay|gpt5ref|fullscale]  # text-embedding-3-small
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
DATASET = sys.argv[2] if len(sys.argv) > 2 else "replay"
DIR = REPO / "results" / DATASET
SENT = DIR / "sentences.csv"


def embed_st(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return m.encode(texts, batch_size=256, show_progress_bar=True,
                    normalize_embeddings=True).astype(np.float32)


def embed_openai(texts):
    from openai import OpenAI
    from tqdm import tqdm
    client = OpenAI()
    out = []
    B = 256
    for i in tqdm(range(0, len(texts), B), desc="openai"):
        batch = [t[:8000] for t in texts[i:i + B]]
        r = client.embeddings.create(model="text-embedding-3-small", input=batch)
        out.extend(d.embedding for d in r.data)
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "st"
    out = DIR / f"emb_{backend}.npy"
    df = pd.read_csv(SENT)
    texts = df["text"].astype(str).tolist()
    if out.exists():
        arr = np.load(out)
        if arr.shape[0] == len(texts):
            print(f"cached: {out.name} {arr.shape}")
            return
    print(f"embedding {len(texts)} sentences with {backend} ...")
    arr = embed_st(texts) if backend == "st" else embed_openai(texts)
    np.save(out, arr)
    print(f"saved {out.name} {arr.shape}")


if __name__ == "__main__":
    main()
