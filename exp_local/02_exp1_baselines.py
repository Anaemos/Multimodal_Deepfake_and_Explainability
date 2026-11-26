# %% [markdown]
# # 02 · Experiment 1 — single-modality baselines vs. multimodal fusion
#
# **Research question (Exp. 1):** does combining voice and face-image evidence beat each modality alone?
#
# Systems compared (all trained on the frozen embeddings from `01_extract_features`, logistic regression head, `C` tuned on the validation set):
#
# | System | Input |
# |---|---|
# | Audio only | Wav2Vec2 (best layer chosen on validation) |
# | Image only | CLIP ViT-B/32 on face crops |
# | Text only (CLIP) | CLIP-text embedding of the Whisper transcript |
# | Text only (TF-IDF) | word 1–2-gram TF-IDF of the Whisper transcript |
# | **Audio + Image** | early fusion (concatenation) — the project's core voice + image setting |
# | **Audio + Image + Text** | early fusion of all three streams |
#
# Metrics: accuracy, precision, recall, F1 (FAKE = positive class), macro-F1, ROC-AUC, PR-AUC, confusion matrices,
# per-category accuracy (RVRA / RVFA / FVRA / FVFA) and a repeated speaker-grouped re-split for mean ± std.
#
# Works on Kaggle (reads `features.npz` + `meta.csv` from your saved output/dataset) and locally.

# %%
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, roc_curve, classification_report)

warnings.filterwarnings("ignore")
try:
    display
except NameError:
    display = lambda x: print(x.to_string() if hasattr(x, "to_string") else x)

SEED = 42
N_RESPLITS = 5
CS = [1e-3, 1e-2, 1e-1, 1, 10]
OUT = os.environ.get("EXP_OUT", "/kaggle/working/exp1")
os.makedirs(OUT, exist_ok=True)

def find(name):
    cands = []
    if os.environ.get("FEAT_DIR"):
        cands.append(os.path.join(os.environ["FEAT_DIR"], name))
    cands.append(os.path.join("/kaggle/working/features", name))
    cands += glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"{name} not found. Run 01_extract_features first or add its output as an input dataset.")

FEAT_NPZ, META_CSV = find("features.npz"), find("meta.csv")
print("features:", FEAT_NPZ, "\nmeta:    ", META_CSV)

# %% [markdown]
# ## 1 · Load features

# %%
Z = np.load(FEAT_NPZ)
meta = pd.read_csv(META_CSV).sort_values("clip_id").reset_index(drop=True)
assert (Z["clip_id"] == meta.clip_id.values).all(), "features.npz and meta.csv are out of sync"

keep = ((meta.ok_img == 1) & (meta.ok_aud == 1)).values
print(f"dropping {int((~keep).sum())} clips with failed image/audio decode")
meta = meta[keep].reset_index(drop=True)
A_LAYERS, IMG, TXT = Z["audio_layers"][keep], Z["image"][keep], Z["text"][keep]
texts = meta.transcript.fillna("").astype(str).values
y = meta.label.values                       # 1 = FAKE
groups = meta.speaker.values
cats = meta.category.values

print("clips:", len(meta), "| REAL:", int((y == 0).sum()), "| FAKE:", int((y == 1).sum()))
print(pd.crosstab(meta.split, meta.category).to_string())

tr0 = np.where(meta.split == "train")[0]
va0 = np.where(meta.split == "val")[0]
te0 = np.where(meta.split == "test")[0]
print(f"\nmain split  train={len(tr0)}  val={len(va0)}  test={len(te0)}  (split by speaker)")
print("majority-class accuracy on test:", round(max((y[te0] == 0).mean(), (y[te0] == 1).mean()), 3), "(anything near this is no better than guessing)")

# %% [markdown]
# ## 2 · Pipeline: features → scaling → logistic regression (C tuned on validation only)

# %%
def metrics(yt, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(acc=accuracy_score(yt, yh), precision=precision_score(yt, yh, zero_division=0),
                recall=recall_score(yt, yh, zero_division=0), f1=f1_score(yt, yh, zero_division=0),
                macro_f1=f1_score(yt, yh, average="macro", zero_division=0),
                roc_auc=roc_auc_score(yt, p), pr_auc=average_precision_score(yt, p))

def scale(tr, *others):
    sc = StandardScaler().fit(tr)
    return (sc.transform(tr),) + tuple(sc.transform(o) for o in others)

def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in CS:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
        auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    return best[2], best[1], best[0]

def build(tr, va, te):
    """Return {system: (Xtr, Xva, Xte)} and the chosen audio layer. All fitting uses train only; layer choice uses val only."""
    def audio_mat(layer):
        return A_LAYERS.mean(1) if layer == "mean" else A_LAYERS[:, layer]

    best_layer, best_auc = None, -1
    for layer in ["mean"] + list(range(A_LAYERS.shape[1])):
        M = audio_mat(layer)
        a_tr, a_va = scale(M[tr], M[va])
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(a_tr, y[tr])
        auc = roc_auc_score(y[va], clf.predict_proba(a_va)[:, 1])
        if auc > best_auc:
            best_layer, best_auc = layer, auc

    M = audio_mat(best_layer)
    A = scale(M[tr], M[va], M[te])
    I = scale(IMG[tr], IMG[va], IMG[te])
    T = scale(TXT[tr], TXT[va], TXT[te])
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(texts[tr])
    TF = (tf.transform(texts[tr]), tf.transform(texts[va]), tf.transform(texts[te]))
    fuse = lambda parts: tuple(np.hstack([p[i] for p in parts]) for i in range(3))

    feats = {
        "Audio only (Wav2Vec2)": A,
        "Image only (CLIP, face crop)": I,
        "Text only (CLIP of ASR)": T,
        "Text only (TF-IDF of ASR)": TF,
        "Audio + Image (early fusion)": fuse([A, I]),
        "Audio + Image + Text (early fusion)": fuse([A, I, T]),
    }
    return feats, best_layer

def run_split(tr, va, te):
    feats, layer = build(tr, va, te)
    rows, probs = [], {}
    for name, (Xtr, Xva, Xte) in feats.items():
        clf, C, _ = fit_lr(Xtr, y[tr], Xva, y[va])
        p = clf.predict_proba(Xte)[:, 1]
        rows.append(dict(system=name, C=C, **metrics(y[te], p)))
        probs[name] = p
    return pd.DataFrame(rows).set_index("system"), probs, layer

# %% [markdown]
# ## 3 · Main result (fixed speaker-disjoint split)

# %%
res, probs, layer = run_split(tr0, va0, te0)
print("audio layer selected on validation:", layer, "(0 = CNN features, 12 = last transformer layer, 'mean' = average over layers)\n")
display(res.round(3))
res.round(4).to_csv(f"{OUT}/exp1_main_results.csv")

# %% [markdown]
# ## 4 · Confusion matrices and ROC curves

# %%
names = list(probs)
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
for ax, n in zip(axes.ravel(), names):
    cm = confusion_matrix(y[te0], (probs[n] >= 0.5).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["REAL", "FAKE"]); ax.set_yticklabels(["REAL", "FAKE"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(n, fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/exp1_confusion_matrices.png", dpi=160); plt.show()

plt.figure(figsize=(6.5, 5.5))
for n in names:
    fpr, tpr, _ = roc_curve(y[te0], probs[n])
    plt.plot(fpr, tpr, label=f"{n} (AUC {res.loc[n, 'roc_auc']:.3f})", lw=2 if "+" in n else 1.2)
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("Exp. 1 — ROC on held-out speakers")
plt.legend(fontsize=8, loc="lower right"); plt.tight_layout(); plt.savefig(f"{OUT}/exp1_roc.png", dpi=160); plt.show()

# %% [markdown]
# ## 5 · Where does each modality help? Accuracy per FakeAVCeleb category
#
# RVRA = real (so this column is the true-negative rate); RVFA = only the voice is fake; FVRA = only the face is fake; FVFA = both.
# A voice-only detector *should* struggle on FVRA and an image-only detector on RVFA — this table is the cleanest evidence for the "multimodal helps" claim.

# %%
rows = {}
for n in names:
    pred = (probs[n] >= 0.5).astype(int)
    rows[n] = {c: (pred[cats[te0] == c] == y[te0][cats[te0] == c]).mean() if (cats[te0] == c).any() else np.nan
               for c in ["RVRA", "RVFA", "FVRA", "FVFA"]}
bycat = pd.DataFrame(rows).T
display(bycat.round(3))
print("test clips per category:", pd.Series(cats[te0]).value_counts().to_dict())
bycat.round(4).to_csv(f"{OUT}/exp1_per_category_accuracy.csv")

# %% [markdown]
# ## 6 · Class-wise report for the core voice + image system

# %%
core = "Audio + Image (early fusion)"
rep = classification_report(y[te0], (probs[core] >= 0.5).astype(int), target_names=["REAL", "FAKE"], digits=3)
print(core, "\n", rep)
with open(f"{OUT}/exp1_classwise_{core.split(' (')[0].replace(' + ', '_').replace(' ', '')}.txt", "w") as f:
    f.write(rep)

# %% [markdown]
# ## 7 · Robustness: repeat with 5 different speaker-disjoint splits (mean ± std)
#
# One split can be lucky. Because the heads train in seconds, re-run everything (including the audio-layer choice) on several random speaker-grouped splits.

# %%
def grouped_split(seed, val=0.15, test=0.15):
    idx = np.arange(len(y))
    trv, te = next(GroupShuffleSplit(1, test_size=test, random_state=seed).split(idx, groups=groups))
    a, b = next(GroupShuffleSplit(1, test_size=val / (1 - test), random_state=seed).split(trv, groups=groups[trv]))
    return trv[a], trv[b], te

runs = []
for s in range(N_RESPLITS):
    tr, va, te = grouped_split(SEED + s)
    r, _, lyr = run_split(tr, va, te)
    r["seed"] = SEED + s
    runs.append(r.reset_index())
allr = pd.concat(runs)
summary = allr.groupby("system")[["acc", "f1", "macro_f1", "roc_auc", "pr_auc"]].agg(["mean", "std"]).round(3)
summary = summary.loc[res.index]
display(summary)
summary.to_csv(f"{OUT}/exp1_resplit_mean_std.csv")

# %% [markdown]
# ## 8 · Read-out template for the paper (fill in from the numbers above — don't trust any claim you can't see in the tables)
#
# * **Does fusion beat the best single modality?** Compare *Audio + Image* with the better of *Audio only* / *Image only* on F1 and ROC-AUC, on both the main split and the mean ± std table. If the gap is within one std, say it is not significant.
# * **Which modality carries the signal?** Use the per-category table: RVFA tests the voice, FVRA tests the face, FVFA both.
# * **Text:** transcripts of real and cloned speech are often near-identical, so text-only is expected to sit near chance. If it does, that is a legitimate finding — it motivates using text for *cross-modal consistency* (CLIP transcript–image similarity) rather than as a standalone detector.
# * **Caveat to state:** frozen encoders + linear head measure how linearly separable the embeddings are, not the ceiling of each modality. Real and fake clips can also differ in compression/resolution, so high image-only scores may partly reflect dataset artefacts — say so in the threats-to-validity section.
