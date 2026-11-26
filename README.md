# Multimodal Deepfake Detection & Explainability

Complementary voice and face deepfake verification with abstention (`REAL` / `FAKE` / `UNCERTAIN`) and stream-level plain-language explanations, evaluated on a 998-clip speaker-disjoint subset of **FakeAVCeleb v1.2** (CSET 346: Natural Language Processing, Project PJ_07).

## Overview

Audio-video deepfakes may manipulate voice (`RVFA`), face (`FVRA`), or both (`FVFA`), leaving unimodal detectors without relevant evidence. This project:
1. Extracts frozen **wav2vec 2.0-base** speech embeddings (13-layer mean, 768-d), frozen **CLIP ViT-B/32** face-crop embeddings (3 sampled frames, 512-d), and **Whisper-base** transcripts.
2. Trains separate class-balanced logistic regression classifiers for voice and face, averages their probabilities, and tunes a validation abstention band (`REAL` / `FAKE` / `UNCERTAIN`).
3. Evaluates modality baselines, fusion strategies, leave-one-manipulation-type-out generalization, ablations, error patterns, explanation fidelity/deletion tests, frame-count variants, and demographic slice rates.
4. Packages a CLI and Gradio web demo in `system/`.

## Key Results (5-Fold Speaker-Disjoint Out-of-Fold, 998 Clips / 500 Speakers)

| Metric | Value |
| --- | --- |
| Voice-only ROC-AUC | 0.817 |
| Face-only ROC-AUC | 0.746 |
| Fused ROC-AUC | 0.900 |
| Coverage (answered clips) | 83.8% (162 `UNCERTAIN`) |
| Accuracy on answered clips | 86.6% |

- **Per-category answered accuracy**: `RVRA` (real) 86.5%, `RVFA` (cloned voice) 99.2%, `FVRA` (face swap) 56.1%, `FVFA` (both) 100.0%.
- **Transcripts**: evaluated via CLIP/TF-IDF features and excluded from the final scorer because they lowered fusion AUC.

See [`results_final/FINAL_NUMBERS.md`](results_final/FINAL_NUMBERS.md) and [`ieee_paper/main.tex`](ieee_paper/main.tex) for full tables, sanity checks, and limitations.

## Repository Structure

- `01-extract-features.ipynb` — Feature extraction notebook (Kaggle/dataset preprocessing).
- `experiment1/` – `experiment6/` — Experiment notebooks (`02-exp1-baselines.ipynb` through `07-exp6-explanations.ipynb`) and saved results.
- `exp_local/` — Local Python runner (`run_experiments.py`) and scripts `02_exp1_baselines.py` through `09_final_pack.py` + `make_features_variant.py`.
- `system/` — Deployable verifier (`verifier_core.py`, `extract.py`, `extract_local.py`, `verify.py`, `app.py`, `final_system.joblib`, `requirements.txt`).
- `results_final/`, `results_kaggle_env/`, `results_local/`, `results_local_8f/` — Consolidated figures, CSV metrics, and final numbers.
- `ieee_paper/` — IEEE conference paper (`main.tex`, Word build script `build_word_paper.py`, and compiled `.docx`).
- `*.pdf` — Project specification (`PJ_07.pdf`) and experiment/handoff reports.

> **Note on excluded data:** The raw `FakeAVCeleb_v1.2/` dataset (~6.3 GB), Python `.venv/`, and cached `.npz` feature matrices are excluded via `.gitignore`. FakeAVCeleb requires accepting the official non-commercial research access terms before obtaining media.

## Quickstart

### 1. Run the Verifier Demo (`system/`)

Requires Python, `ffmpeg` on `PATH`, and dependencies from `system/requirements.txt`:

```bash
pip install -r system/requirements.txt
python system/verify.py path/to/clip.mp4
python system/app.py
```

The first run downloads the pretrained wav2vec 2.0, CLIP, and Whisper checkpoints (~1 GB).

### 2. Run Experiments Locally (`exp_local/`)

Once you have extracted `features.npz` and `meta.csv` into a local features directory (e.g. `features_local/`):

```bash
cd exp_local
python run_experiments.py --features ..\features_local --out ..\results_local --install-model ..\system
```

Useful flags: `--only 4 5 6`, `--skip-neural`, `--quick`.

## Limitations & Ethics

- Evaluated only on a 998-clip subset of FakeAVCeleb v1.2; cross-dataset generalization is untested.
- Face-swap-only clips (`FVRA`) remain the weakest category.
- Intended as a research prototype for human-in-the-loop review, not standalone forensic evidence.
