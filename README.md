# Writing and rewriting in AI-assisted text: analysis code and derived data

Code and derived data for:

> "Distinct Geometric Signatures of Writing and Rewriting in AI-Assisted Text." (Under review, 2026.)

The paper replays the edit logs of the CoAuthor corpus (1,447 human–GPT-3
co-writing sessions), classifies each final sentence by its share of
human-typed characters (choosing, rewriting, or mainly human writing), and
compares equal-sized sentence sets with unused model suggestions by
effective dimension, orientation, and an out-of-reference rate.

The Supplementary Material (Sections S.I–S.XI) is
`supplementary/supplementary.pdf`.

## Contents

- `data/coauthor_metadata.csv` — session metadata from the CoAuthor release
  (CC-BY, Lee et al. 2022): official prompt codes and, for story sessions,
  pseudonymous writer identifiers.
- `results/replay/` — the replayed sentence table (`sentences.csv`) and every
  per-prompt-group result table used in the paper.
- `results/gpt5ref/` — the regenerated GPT-5 suggestion pools and their results.
- `results/fullscale/` — the earlier direct extraction, kept only to rebuild
  the comparison in Supplementary Sections S.I and S.III.
- `results/sampling_sensitivity/` — fixed-design and session-contribution
  checks (Supplementary S.XI).
- `supplementary/supplementary.pdf` — the Supplementary Material.

Not included: the raw CoAuthor logs (download from
https://coauthor.stanford.edu/ and place the session files at
`data/coauthor-v1.0/*.jsonl`) and the embedding matrices (`.npy`, recomputed by
`fullscale_embed.py`; the MiniLM backend runs locally, the OpenAI backend needs
an API key).

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export OPENAI_API_KEY=...   # only for OpenAI embeddings, GPT-5 pools, paraphrases
```

The random seed is 42 throughout.

## Quick check from the shipped tables

These run in seconds and need neither the raw logs nor the embeddings:

```bash
python make_paper_figures.py   # Figures 1-3 of the main text -> figs/
python wr_checks_summary.py    # checks of the writing-rewriting contrast (S.II)
python upper_bounds.py         # upper bounds for choosing and rewriting (S.V)
```

## Full rerun

`rerun_all.sh` reruns every analysis in order. Steps that need the raw logs
run only with `RAW=1`; steps that call the OpenAI API run only with `API=1`.
Without them, the script starts from the shipped `sentences.csv` and needs the
embeddings in `results/replay/`.

```bash
sh rerun_all.sh                 # from shipped sentence tables
RAW=1 API=1 sh rerun_all.sh     # everything, including extraction and API steps
```

## Where each result comes from

Unless stated otherwise, each script takes `st` (MiniLM) or `openai`
(text-embedding-3-small) and writes to `results/replay/`.

**Extraction and pools (Method; S.I)**
- `replay_extract.py` — replays each session's edit stream (retain / insert /
  delete) with per-character authorship and writes `sentences.csv`. Needs raw logs.
- `menu_ids.py` — suggestion-menu identifier of each dismissed candidate.
- `pool_counts.py` — staged accounting of the dismissed pool (S.I).
- `legacy_extraction.py` — the earlier direct extraction (S.I, S.III), then
  `fullscale_embed.py` / `fullscale_analyze.py` with `fullscale`.
- `fullscale_embed.py` — sentence embeddings for `replay`, `gpt5ref` or `fullscale`.

**Main comparisons (Results)**
- `geometry_forms.py` — the three operations against the reference: effective
  dimension, raw leakage, out-of-reference rate. Flags: `--lenmatch`,
  `--nomixed`, `--fullsent`, `--menucap`, `--decorrelate`.
- `spectral_null.py` — orientation component and its decomposition (S.V).
- `geometry_dose.py` (`--fullsent`) and `dose_orientation.py` — human-typed
  share bins.
- `make_paper_figures.py` — Figures 1–3; `make_figures.py` — Figures S1–S3.

**Checks of the writing–rewriting contrast (S.II)**
- `session_matched.py`, `mixed_subtypes.py` (raw logs) and
  `mixed_subtype_geometry.py` (character arrangements; the CSV labels
  `completion` and `append` are the human–suggestion and suggestion–human
  arrangements of the paper), `mixed_origins.py` (original suggestions;
  `extract` needs raw logs, `embed openai` the API), `pr_bias_corrected.py`,
  `nonlinear_dimension.py` (TwoNN, Levina–Bickel, saturation),
  `union_dimension.py`, `wr_checks_summary.py`.

**Scalar benchmark (S.III)** — `fullscale_analyze.py`, `scalar_sensitivity.py`
(the 64 variants).

**Out-of-reference radius (S.IV)** — `radius_sweep.py`.

**Robustness inventory (S.V)**
- Null calibrations and small designs: `null_calibration.py`,
  `check_writer_mixing.py`, `check_all_controls.py`,
  `check_within_session_null.py`, `check_stacked_null.py`.
- Reference and pool variants: `near_duplicates.py`, `sibling_free.py`
  (`parse` needs raw logs), `menu_matched.py`, `upper_bounds.py`.
- Orientation: `orient_k_sweep.py` (k = 10–40), `principal_angles.py`.
- Estimators and encoders: `other_encoders.py` (T5, GPT-2; then
  `geometry_forms.py t5|gpt2`), `sentence_regression.py`.
- Sessions and writers: `session_span_vendi.py`, `rewrite_clustering.py`,
  `writer_bootstrap.py`, `writer_prompt_bootstrap.py` (`--nodup`,
  `--wdistinct`, `--vs-mixed`), `standard_bootstrap.py`, `seed_sweep.py`.
- Paraphrase control: `paraphrase_control.py` (`para` needs the API; the
  shipped `paraphrase.csv` is the one used).

**Session thirds (S.VI)** — `geometry_temporal.py`, `orientation_frames.py temporal`.

**Residual spectrum (S.VII)** — `residual_spectrum.py`.

**Engagement moderators (S.VIII)** — `geometry_moderator.py`.

**GPT-5 reference (Results; S.IX)** — `gpt5_regen.py` (API), `gpt5_analyze.py`,
`pool_lenmatch.py`, `orientation_frames.py gpt5`, `union_dimension.py gpt5`.

**Writing process and proxy matching (S.X)** — `process_flags.py` (raw logs),
`process_delta.py`, `process_robust.py`, `coherence_check.py`,
`position_control.py`, `selection_h0.py`, `orientation_frames.py pure`.

**Fixed-design and session-contribution checks (S.XI)** —
`sampling_sensitivity.py REPO OUTPUT`.

## Licenses

Code: MIT (see `LICENSE`). Derived data: CC-BY 4.0, as a derivative of the
CoAuthor corpus (Mina Lee, Percy Liang, and Qian Yang, 2022,
https://coauthor.stanford.edu/); cite CoAuthor when you use it.
