"""Check that extract.py reproduces the features stored from the Kaggle run, for ONE clip.
   python check_parity.py <path-to-clip.mp4> <folder-with-features.npz-and-meta.csv>"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import Extractor

clip, feat_dir = sys.argv[1], sys.argv[2]
Z = np.load(os.path.join(feat_dir, "features.npz"))
meta = pd.read_csv(os.path.join(feat_dir, "meta.csv")).sort_values("clip_id").reset_index(drop=True)
norm = lambda p: str(p).replace("\\", "/").split("FakeAVCeleb_v1.2/")[-1]
hit = meta.index[meta.path.map(norm) == norm(clip)]
if len(hit) == 0:
    sys.exit("This clip is not in meta.csv (it was not part of the 998-clip subset). Pick one that is, e.g. one listed in exp5_hard_cases.csv.")
i = hit[0]
f = Extractor(with_transcript=False).features(clip)
V_ref, F_ref = Z["audio_layers"][i].mean(0), Z["image"][i]
cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
print(f"clip_id {meta.clip_id[i]}  category {meta.category[i]}")
print(f"voice features cosine similarity with Kaggle: {cos(f['V'], V_ref):.5f}   (want > 0.999)")
print(f"face  features cosine similarity with Kaggle: {cos(f['F'], F_ref):.5f}   (want > 0.999)")
print(f"face_rate local {f['face_rate']:.2f} vs Kaggle {meta.face_rate[i]:.2f}")
