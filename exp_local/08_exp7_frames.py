# %% [markdown]
# # 08 · Experiment 7 — does using more video frames help? (local features vs Kaggle features)
#
# **Question.** The Kaggle features average the face embedding of **3 frames** per clip. The local extraction stored **16 frames** per clip. Does the face stream (our weakest part) improve with more frames,
# and does adding *how much the face changes from frame to frame* help (face swaps can flicker)?
#
# **Controls.** Same 998 clips, same order, same speaker-disjoint splits (seeds 42-46), same classifiers (logistic regression, C chosen on validation AUC) as Exp. 1-6. Only the face feature changes.
# The voice model is identical in every row, so differences come from the face stream alone.
#
# **Variants (all use frames from the same 16 per clip)**
# | Name | Face feature |
# |---|---|
# | mean K | average embedding of K evenly spaced frames, K = 1, 3, 8, 16 |
# | mean K + temporal | mean (512) + std over frames (512) + mean absolute frame-to-frame difference (512), K = 8, 16 |
# | detected-only 16 | mean over only the frames where a face was found (falls back to all frames if none) |
# | Kaggle 3-frame (reference) | the original Kaggle feature file, if `BASE_DIR` is given |
#
# **How we decide.** The winner is chosen on **validation** AUC (mean over splits), never on test. Test numbers are reported for all rows. Differences below about 0.01 AUC are within split-to-split noise (see the std column).
#
# **Run:** `python run_experiments.py --features ..\features_local --baseline ..\features_kaggle --out ..\local_results --only 7`

# %%
import os, glob, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
warnings.filterwarnings("ignore")
try:
    display
except NameError:
    display = lambda x: print(x.to_string() if hasattr(x, "to_string") else x)

SEED, N_SPLITS = 42, 5
CS = [1e-3, 1e-2, 1e-1, 1, 10]
OUT = os.environ.get("EXP_OUT", "exp7"); os.makedirs(OUT, exist_ok=True)
FEAT_DIR, BASE_DIR = os.environ.get("FEAT_DIR"), os.environ.get("BASE_DIR")

def load(d):
    Z = np.load(os.path.join(d, "features.npz"))
    meta = pd.read_csv(os.path.join(d, "meta.csv")).sort_values("clip_id").reset_index(drop=True)
    assert (Z["clip_id"] == meta.clip_id.values).all()
    return Z, meta

Z, meta = load(FEAT_DIR)
if "image_frames" not in Z.files:
    print("features.npz has no per-frame embeddings (image_frames). Run extract_local.py first. SKIPPING Experiment 7.")
    raise SystemExit(0)
keep = ((meta.ok_img == 1) & (meta.ok_aud == 1)).values
meta = meta[keep].reset_index(drop=True)
FR, FOUND = Z["image_frames"][keep], Z["face_found"][keep]            # (N, 16, 512), (N, 16)
V = Z["audio_layers"][keep].mean(1)
y, groups, cats = meta.label.values, meta.speaker.values, meta.category.values
NF = FR.shape[1]
base = None
if BASE_DIR:
    Zb, mb = load(BASE_DIR)
    kb = ((mb.ok_img == 1) & (mb.ok_aud == 1)).values
    mb = mb[kb].reset_index(drop=True)
    assert len(mb) == len(meta) and (mb.speaker.values == groups).all() and (mb.label.values == y).all(), "baseline and local features do not describe the same clips"
    base = Zb["image"][kb]
print(f"clips {len(y)} | frames per clip {NF} | baseline features: {'yes' if base is not None else 'no'}")

# %% [markdown]
# ## 1 · Face feature variants

# %%
def pick(K):
    return np.array([NF // 2]) if K == 1 else np.linspace(0, NF - 1, K).round().astype(int)

def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def mean_k(K):
    return l2(FR[:, pick(K)].mean(1))

def temporal(K):
    X = FR[:, pick(K)]
    return np.hstack([l2(X.mean(1)), X.std(1), np.abs(np.diff(X, axis=1)).mean(1)])

def detected_only():
    w = FOUND.astype(np.float32)
    none = w.sum(1) == 0
    w[none] = 1.0
    return l2((FR * w[..., None]).sum(1) / w.sum(1, keepdims=True))

VARIANTS = {}
if base is not None:
    VARIANTS["Kaggle 3-frame (reference)"] = base
for K in (1, 3, 8, 16):
    if K <= NF:
        VARIANTS[f"mean {K}"] = mean_k(K)
for K in (8, 16):
    if K <= NF:
        VARIANTS[f"mean {K} + temporal"] = temporal(K)
VARIANTS[f"detected-only {NF}"] = detected_only()
print({k: v.shape[1] for k, v in VARIANTS.items()})

# %% [markdown]
# ## 2 · Evaluate (5 speaker-disjoint splits)

# %%
def grouped_split(seed, val=0.15, test=0.15):
    idx = np.arange(len(y))
    trv, te = next(GroupShuffleSplit(1, test_size=test, random_state=seed).split(idx, groups=groups))
    a, b = next(GroupShuffleSplit(1, test_size=val / (1 - test), random_state=seed).split(trv, groups=groups[trv]))
    return trv[a], trv[b], te

def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in CS:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
        a = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if best is None or a > best[0]:
            best = (a, clf)
    return best[1]

def val_threshold(yv, pv):
    grid = np.linspace(0.05, 0.95, 91)
    sc = np.array([balanced_accuracy_score(yv, (pv >= t).astype(int)) for t in grid])
    cand = grid[sc >= sc.max() - 1e-12]
    return float(cand[np.argmin(np.abs(cand - 0.5))])

def stream(X, tr, va, te):
    sc = StandardScaler().fit(X[tr])
    clf = fit_lr(sc.transform(X[tr]), y[tr], sc.transform(X[va]), y[va])
    return clf.predict_proba(sc.transform(X[va]))[:, 1], clf.predict_proba(sc.transform(X[te]))[:, 1]

rows = []
for s in range(N_SPLITS):
    tr, va, te = grouped_split(SEED + s)
    vv, vt = stream(V, tr, va, te)
    for name, X in VARIANTS.items():
        fv, ft = stream(X, tr, va, te)
        mv, mt = (vv + fv) / 2, (vt + ft) / 2
        thr = val_threshold(y[va], mv)
        yp = (mt >= thr).astype(int)
        r = dict(split=s, variant=name, dims=X.shape[1], face_val_auc=roc_auc_score(y[va], fv), face_test_auc=roc_auc_score(y[te], ft),
                 fused_val_auc=roc_auc_score(y[va], mv), fused_test_auc=roc_auc_score(y[te], mt), fused_test_acc=accuracy_score(y[te], yp))
        for c in ("RVRA", "RVFA", "FVRA", "FVFA"):
            m = cats[te] == c
            r[f"acc_{c}"] = float((yp[m] == y[te][m]).mean()) if m.any() else np.nan
        rows.append(r)
    print(f"split {s} done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/exp7_long.csv", index=False)
num = [c for c in df.columns if c not in ("split", "variant")]
mean, std = df.groupby("variant", sort=False)[num].mean(), df.groupby("variant", sort=False)[num].std()
summ = mean.round(3).join(std[["face_test_auc", "fused_test_auc"]].round(3).add_suffix("_std"))
summ.to_csv(f"{OUT}/exp7_summary.csv")
display(summ[["dims", "face_val_auc", "face_test_auc", "face_test_auc_std", "fused_val_auc", "fused_test_auc", "fused_test_auc_std", "fused_test_acc", "acc_RVRA", "acc_RVFA", "acc_FVRA", "acc_FVFA"]])

ref = "Kaggle 3-frame (reference)" if base is not None else "mean 3"
pr = []
for v in VARIANTS:
    if v == ref:
        continue
    for col in ("face_test_auc", "fused_test_auc"):
        d = df[df.variant == v].set_index("split")[col] - df[df.variant == ref].set_index("split")[col]
        pr.append(dict(variant=v, metric=col, vs=ref, mean_diff=d.mean(), std=d.std(), splits_better=int((d > 0).sum())))
paired = pd.DataFrame(pr).round(3)
paired.to_csv(f"{OUT}/exp7_paired_vs_reference.csv", index=False)
display(paired)
best = summ.fused_val_auc.idxmax()
print(f"\nBest variant by VALIDATION fused AUC: {best}  (validation {summ.fused_val_auc[best]:.3f}, test {summ.fused_test_auc[best]:.3f})")

# %% [markdown]
# ## 3 · Figure

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for a, col, title in ((ax[0], "face_test_auc", "Face stream alone"), (ax[1], "fused_test_auc", "Voice + face (late mean)")):
    ks = [k for k in (1, 3, 8, 16) if f"mean {k}" in summ.index]
    a.errorbar(ks, [summ.loc[f"mean {k}", col] for k in ks], [summ.loc[f"mean {k}", col + "_std"] for k in ks], marker="o", capsize=3, label="mean of K frames", c="#2a6fdb")
    kt = [k for k in (8, 16) if f"mean {k} + temporal" in summ.index]
    if kt:
        a.errorbar(kt, [summ.loc[f"mean {k} + temporal", col] for k in kt], [summ.loc[f"mean {k} + temporal", col + "_std"] for k in kt], marker="s", capsize=3, label="+ frame-to-frame change", c="#e08a1e")
    if base is not None:
        a.axhline(summ.loc["Kaggle 3-frame (reference)", col], c="#777", ls="--", lw=1)
        a.text(ks[-1], summ.loc["Kaggle 3-frame (reference)", col], " Kaggle 3-frame", va="bottom", ha="right", fontsize=8, color="#555")
    a.set_xscale("log", base=2); a.set_xticks(ks); a.set_xticklabels(ks)
    a.set_xlabel("frames per clip"); a.set_ylabel("test ROC-AUC (mean ± std, 5 splits)"); a.set_title(title, pad=10); a.legend(fontsize=8, loc="lower right")
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/exp7_frames.png", dpi=150); plt.show()
print(sorted(os.listdir(OUT)))
