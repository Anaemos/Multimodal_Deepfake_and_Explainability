# Multimodal deepfake verifier (voice + face) — first showable system

Takes one short video, returns **REAL / FAKE / UNCERTAIN** with a plain-language explanation.

Pipeline: video -> frames + audio -> frozen encoders (CLIP for the face crop, wav2vec2 for the voice, Whisper for a transcript shown to the reader)
-> two logistic regressions (voice, face) averaged -> verdict (UNCERTAIN = low confidence) -> explanation.

## Get the trained model
Run `07_exp6_explanations.ipynb` on Kaggle (input: your `FakeAVCeleb-features` dataset). It writes `final_system.joblib`.
Download it into this folder, next to `app.py`.

## Run
    pip install -r requirements.txt        # and install ffmpeg
    python verify.py clip.mp4              # command line
    python app.py                          # web demo (Gradio)

The first run downloads CLIP, wav2vec2 and Whisper (about 1 GB). An RTX 4060 is plenty; CPU also works, slower.

## Files
- `verifier_core.py` training, scoring, UNCERTAIN rule, explanation (same code the notebook uses)
- `extract.py` video -> features, identical preprocessing to the Kaggle extraction notebook
- `verify.py`, `app.py` command line and web demo

## Limits
Trained and evaluated only on FakeAVCeleb. Needs a visible face and speech; about 10 s of audio is used. Face-swap-only fakes are the weakest case.
