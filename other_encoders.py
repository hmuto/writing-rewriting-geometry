"""Embeddings from encoders without a contrastive objective (T5, GPT-2).

MiniLM and text-embedding-3-small are both trained contrastively. This
script embeds the same sentences with the t5-base encoder and with GPT-2
(last hidden layer, attention-mask mean pooling, at most 64 tokens, L2
normalization), so the main pipeline can be rerun on them.

Usage: python other_encoders.py [gpt2|t5]  # writes results/replay/emb_<name>.npy
       python geometry_forms.py <name>     # then run the main pipeline on it
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys

from transformers import AutoTokenizer, GPT2Model, T5EncoderModel

REPO = Path(__file__).resolve().parent
DIR = REPO / "results" / "replay"
BATCH = 128


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    if name == "t5":
        tok = AutoTokenizer.from_pretrained("t5-base")
        model = T5EncoderModel.from_pretrained("t5-base").to(device).eval()
        dim = model.config.d_model
    else:
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        model = GPT2Model.from_pretrained("gpt2").to(device).eval()
        dim = model.config.n_embd

    df = pd.read_csv(DIR / "sentences.csv")
    texts = df["text"].astype(str).tolist()
    out = np.zeros((len(texts), dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            b = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                    truncation=True, max_length=64)
            b = {k: v.to(device) for k, v in b.items()}
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1)
            out[i:i + BATCH] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
            if i % (BATCH * 100) == 0:
                print(f"{i}/{len(texts)}", flush=True)
    out /= np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)
    np.save(DIR / f"emb_{name}.npy", out)
    print(f"saved {out.shape} -> emb_{name}.npy")


if __name__ == "__main__":
    main()
