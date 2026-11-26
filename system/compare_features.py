"""Compare the Kaggle features with the locally extracted ones, clip by clip.

    python compare_features.py ..\features_kaggle ..\features_local

Voice (wav2vec2) and transcript features should be (almost) identical: same clips, same code.
Face features legitimately differ (3 frames vs the mean of 16 frames), so only their similarity is reported.
"""
import sys, numpy as np, pandas as pd

kd, ld = sys.argv[1], sys.argv[2]
K, L = np.load(kd + "/features.npz"), np.load(ld + "/features.npz")
mk = pd.read_csv(kd + "/meta.csv").sort_values("clip_id").reset_index(drop=True)
ml = pd.read_csv(ld + "/meta.csv").sort_values("clip_id").reset_index(drop=True)
print("clips:", len(mk), len(ml), "| same clip_id order:", bool((K["clip_id"] == L["clip_id"]).all()),
      "| same speaker:", bool((mk.speaker.values == ml.speaker.values).all()), "| same label:", bool((mk.label.values == ml.label.values).all()))
print("ok_aud kaggle/local:", int(mk.ok_aud.sum()), int(ml.ok_aud.sum()), "| ok_img:", int(mk.ok_img.sum()), int(ml.ok_img.sum()))

def cos(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

def report(name, c):
    q = np.quantile(c, [0, 0.01, 0.1, 0.5])
    print(f"{name:34s} cosine  min {q[0]:.4f} | 1% {q[1]:.4f} | 10% {q[2]:.4f} | median {q[3]:.4f} | clips below 0.999: {(c < 0.999).sum()}")

Ak, Al = K["audio_layers"], L["audio_layers"]
report("voice: mean over 13 layers", cos(Ak.mean(1), Al.mean(1)))
for layer in (0, 6, 12):
    report(f"voice: layer {layer}", cos(Ak[:, layer], Al[:, layer]))
report("face: Kaggle 3-frame vs local mean", cos(K["image"], L["image"]))
report("text: CLIP of transcript", cos(K["text"], L["text"]))
same = (mk.transcript.fillna("").astype(str).str.strip().values == ml.transcript.fillna("").astype(str).str.strip().values)
print(f"identical Whisper transcripts: {same.sum()} of {len(same)}")

c = cos(Ak.mean(1), Al.mean(1))
worst = np.argsort(c)[:5]
print("\n5 clips whose voice features differ most:")
for i in worst:
    print(f"  clip {mk.clip_id[i]:4d} {mk.category[i]}  cosine {c[i]:.4f}  {str(mk.path[i])[-60:]}")
