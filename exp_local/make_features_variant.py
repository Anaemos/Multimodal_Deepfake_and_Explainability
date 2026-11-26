"""Make a copy of the local features whose face feature is the mean of N evenly spaced frames (default 8).

    python make_features_variant.py --src ..\features_local --out ..\features_local_8f --frames 8

The experiment scripts read the key `image`, so running them on the new folder evaluates that face feature
with everything else unchanged. Voice, text, clip order and metadata are copied as they are.
"""
import argparse, shutil
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=8)
a = ap.parse_args()

Z = np.load(Path(a.src) / "features.npz")
fr = Z["image_frames"]
nf = fr.shape[1]
idx = np.array([nf // 2]) if a.frames == 1 else np.linspace(0, nf - 1, a.frames).round().astype(int)
img = fr[:, idx].mean(1)
img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "features.npz", clip_id=Z["clip_id"], image=img.astype(np.float32), audio_layers=Z["audio_layers"], text=Z["text"])
shutil.copy(Path(a.src) / "meta.csv", out / "meta.csv")
print(f"wrote {out}  (face = mean of {len(idx)} frames: indices {idx.tolist()})")
