"""09 - Final results pack: everything the guidelines ask for that Exps 1-7 did not already give.

Produces (in OUT):  charts (PNG), tables (CSV) and FINAL_NUMBERS.md
  1. 3-way confusion matrix (REAL / FAKE / UNCERTAIN) + class-wise precision/recall/F1 + coverage
  2. Per-category verdict table (RVRA / RVFA / FVRA / FVFA)
  3. Sanity checks: genuine pairs, native fakes, DELIBERATELY MISMATCHED / recombined voice-face pairs
  4. Bias audit: accuracy / false-alarm / miss rate by ethnicity and sex (from the dataset folder names)
  5. Difficult cases (wrong or UNCERTAIN, ranked) with the system's own explanation
  6. Kaggle-vs-local comparison chart (from the Exp 6 stability files, if present)
All numbers are out-of-fold (5-fold speaker-disjoint CV, same partition rule as Exp 6, seed 42): the model that scores a clip never saw its speaker.

Env: FEAT_DIR (folder with features.npz + meta.csv), SYSTEM_DIR (folder with verifier_core.py), EXP_OUT, RESULTS_ROOT (optional, for the comparison chart)
"""
import os, re, sys, json, glob, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
warnings.filterwarnings("ignore")
try:
    import tabulate  # noqa: F401
except ImportError:   # fall back to plain text tables if 'tabulate' is not installed
    pd.DataFrame.to_markdown = lambda self, index=True, **k: "```\n" + self.to_string(index=index) + "\n```"

FEAT = os.environ.get("FEAT_DIR", "../features_kaggle")
SYSD = os.environ.get("SYSTEM_DIR", "../system")
OUT = os.environ.get("EXP_OUT", "../results_final")
ROOT = os.environ.get("RESULTS_ROOT", "..")
SEED, K = 42, 5
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SYSD)
from verifier_core import Verifier

# palette: one hue per meaning, used consistently
C_REAL, C_FAKE, C_UNC, C_INK, C_GRID = "#2a7f62", "#c0392b", "#9aa0a6", "#222222", "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 130})

# ----------------------------------------------------------------- data
Z = np.load(os.path.join(FEAT, "features.npz"))
meta = pd.read_csv(os.path.join(FEAT, "meta.csv")).sort_values("clip_id").reset_index(drop=True)
assert (Z["clip_id"] == meta.clip_id.values).all()
keep = ((meta.ok_img == 1) & (meta.ok_aud == 1)).values
meta = meta[keep].reset_index(drop=True)
V, F = Z["audio_layers"][keep].mean(1), Z["image"][keep]
y, groups, cats = meta.label.values, meta.speaker.values, meta.category.values
n = len(y)
transcript = meta.transcript.fillna("").astype(str).values
face_rate = meta.face_rate.values

def parse_demo(p):
    parts = re.split(r"[\\/]", str(p))
    for i, s in enumerate(parts):
        if s in ("RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio") and i + 2 < len(parts):
            return parts[i + 1], parts[i + 2]
    return "unknown", "unknown"
dem = [parse_demo(p) for p in meta.path]
meta["ethnicity"], meta["sex"] = [d[0] for d in dem], [d[1] for d in dem]
meta["file"] = [re.split(r"[\\/]", str(p))[-1] for p in meta.path]

def speaker_folds(seed, k=K):
    spk = np.unique(groups)
    perm = np.random.RandomState(seed).permutation(len(spk))
    fold_of = {s: i % k for i, s in zip(range(len(spk)), spk[perm])}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]

# ----------------------------------------------------------------- out-of-fold verdicts
models, folds_of = {}, np.zeros(n, int)
pv, pf, pm = np.zeros(n), np.zeros(n), np.zeros(n)
for k, (trv, te) in enumerate(speaker_folds(SEED)):
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED + k).split(trv, groups=groups[trv]))
    m = Verifier().fit(V, F, y, groups, tr=trv[a], va=trv[b]); models[k] = m; folds_of[te] = k
    pv[te], pf[te], pm[te] = m.score(V[te], F[te])
verdict = np.array([str(models[folds_of[i]].decide(pm[i])) for i in range(n)])
expl = [models[folds_of[i]].explain(pv[i], pf[i], face_rate[i], transcript[i]) for i in range(n)]
conf = np.array([e["confidence"] for e in expl])
meta.assign(fold=folds_of, p_voice=pv, p_face=pf, fused=pm, verdict=verdict, confidence=conf).to_csv(f"{OUT}/final_oof_predictions.csv", index=False)
ans = verdict != "UNCERTAIN"
pred_fake = verdict == "FAKE"
correct = (pred_fake == (y == 1))
N = {}   # master numbers
N["clips"], N["real"], N["fake"] = int(n), int((y == 0).sum()), int((y == 1).sum())
N["speakers"] = int(len(np.unique(groups)))
N["auc_voice"], N["auc_face"], N["auc_fused"] = [round(float(roc_auc_score(y, s)), 3) for s in (pv, pf, pm)]
N["coverage"] = round(float(ans.mean()), 3)
N["accuracy_answered"] = round(float(correct[ans].mean()), 3)
N["accuracy_all_uncertain_as_wrong"] = round(float((correct & ans).mean()), 3)
N["uncertain_clips"] = int((~ans).sum())

# ----------------------------------------------------------------- 1. 3-way confusion + class-wise
T = pd.crosstab(pd.Series(np.where(y == 1, "FAKE (true)", "REAL (true)"), name="truth"), pd.Series(verdict, name="verdict"))
T = T.reindex(index=["REAL (true)", "FAKE (true)"], columns=["REAL", "FAKE", "UNCERTAIN"], fill_value=0)
T.to_csv(f"{OUT}/final_confusion_3way.csv")
rows = []
for cls, lab in (("REAL", 0), ("FAKE", 1)):
    tp = int(((verdict == cls) & (y == lab)).sum()); pred = int((verdict == cls).sum()); true = int((y == lab).sum())
    unc = int(((verdict == "UNCERTAIN") & (y == lab)).sum())
    prec = tp / pred if pred else np.nan; rec_ans = tp / (true - unc) if true - unc else np.nan; rec_all = tp / true
    rows.append(dict(**{"class": cls}, support=true, predicted=pred, true_positives=tp, precision=prec,
                     recall_on_answered=rec_ans, recall_incl_uncertain=rec_all,
                     f1_on_answered=(2 * prec * rec_ans / (prec + rec_ans)) if prec and rec_ans else np.nan, uncertain=unc))
CW = pd.DataFrame(rows).round(3); CW.to_csv(f"{OUT}/final_classwise.csv", index=False)
N["classwise"] = CW.to_dict("records")
N["confusion_3way"] = T.to_dict()

fig, ax = plt.subplots(figsize=(5.6, 3.4))
M = T.values.astype(float); rowp = M / M.sum(1, keepdims=True)
ax.imshow(rowp, cmap="Greys", vmin=0, vmax=1.0)
for i in range(2):
    for j in range(3):
        ax.text(j, i, f"{int(M[i, j])}\n{rowp[i, j]:.0%}", ha="center", va="center", color="white" if rowp[i, j] > .55 else C_INK, fontsize=11)
ax.set_xticks(range(3), T.columns); ax.set_yticks(range(2), ["REAL", "FAKE"])
ax.set_xlabel("system verdict"); ax.set_ylabel("true label"); ax.set_title(f"Final system, 3-way confusion matrix (n={n}, out-of-fold)")
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_confusion_3way.png"); plt.close(fig)

# ROC
fig, ax = plt.subplots(figsize=(4.6, 4.2))
for nm, s, c, ls in (("voice only", pv, "#7a5aa6", "-"), ("face only", pf, "#d68a1f", "-"), ("fused (system)", pm, C_INK, "-")):
    fpr, tpr, _ = roc_curve(y, s); ax.plot(fpr, tpr, color=c, lw=2.2 if "fused" in nm else 1.5, ls=ls, label=f"{nm}  AUC {roc_auc_score(y, s):.3f}")
ax.plot([0, 1], [0, 1], color="#bbb", lw=1, ls="--"); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("ROC, out-of-fold, speaker-disjoint"); ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_roc.png"); plt.close(fig)

# ----------------------------------------------------------------- 2. per category
order = ["RVRA", "RVFA", "FVRA", "FVFA"]
names = {"RVRA": "real video + real audio", "RVFA": "cloned voice", "FVRA": "face swap", "FVFA": "face swap + cloned voice"}
rows = []
for c in order:
    s = cats == c
    if not s.any(): continue
    d = dict(category=c, meaning=names[c], clips=int(s.sum()))
    for v in ("REAL", "FAKE", "UNCERTAIN"): d[v] = int((verdict[s] == v).sum())
    want = "REAL" if c == "RVRA" else "FAKE"
    d["correct_of_answered"] = round(float((verdict[s & ans] == want).mean()), 3) if (s & ans).any() else np.nan
    d["correct_of_all"] = round(float((verdict[s] == want).mean()), 3)
    rows.append(d)
PC = pd.DataFrame(rows); PC.to_csv(f"{OUT}/final_per_category.csv", index=False); N["per_category"] = PC.to_dict("records")
fig, ax = plt.subplots(figsize=(7, 3.4))
left = np.zeros(len(PC))
for v, c in (("REAL", C_REAL), ("FAKE", C_FAKE), ("UNCERTAIN", C_UNC)):
    w = (PC[v] / PC.clips).values; ax.barh(range(len(PC)), w, left=left, color=c, label=v, height=.62)
    for i, (l, x) in enumerate(zip(left, w)):
        if x > .06: ax.text(l + x / 2, i, f"{x:.0%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    left += w
ax.set_yticks(range(len(PC)), [f"{c}\n{names[c]} (n={k})" for c, k in zip(PC.category, PC.clips)], fontsize=9); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1], ["0%", "50%", "100%"]); ax.set_title("What the system says, by true clip type")
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.12))
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_verdict_by_category.png"); plt.close(fig)

# ----------------------------------------------------------------- 3. sanity checks with recombined pairs
rng = np.random.RandomState(7)
sane = []
def pair(idxA, idxB, fold_te, label):
    """voice from clip a (idxA), face from clip b (idxB); only pairs of DIFFERENT speakers, both inside the same test fold."""
    for a in idxA:
        cand = [b for b in idxB if groups[b] != groups[a]]
        if not cand: continue
        b = cand[rng.randint(len(cand))]
        m = models[folds_of[a]]
        _, _, fz = m.score(V[a], F[b]); sane.append(dict(test=label, a=a, b=b, fused=float(fz[0]), verdict=str(m.decide(fz[0]))))
tests = {  # name: (voice source category, face source category, what a sensible system should say)
    "genuine clip (own voice + own face)": None,
    "real voice + real face of ANOTHER person (mismatched identity)": ("RVRA", "RVRA", "REAL?"),
    "cloned voice + real face": ("RVFA", "RVRA", "FAKE"),
    "real voice + swapped face": ("RVRA", "FVRA", "FAKE"),
    "cloned voice + swapped face": ("RVFA", "FVRA", "FAKE"),
}
for k in range(K):
    te = np.where(folds_of == k)[0]
    pool = {c: te[cats[te] == c] for c in order}
    for name, spec in tests.items():
        if spec is None:
            for i in pool["RVRA"]: sane.append(dict(test=name, a=i, b=i, fused=float(pm[i]), verdict=str(verdict[i])))
        else:
            pair(pool[spec[0]], pool[spec[1]], te, name)
S = pd.DataFrame(sane)
rows = []
for name, spec in tests.items():
    s = S[S.test == name]
    rows.append(dict(test=name, pairs=len(s), REAL=int((s.verdict == "REAL").sum()), FAKE=int((s.verdict == "FAKE").sum()), UNCERTAIN=int((s.verdict == "UNCERTAIN").sum()),
                     expected="REAL" if spec is None or spec[2] == "REAL?" else "FAKE"))
SC = pd.DataFrame(rows)
SC["share_FAKE"] = (SC.FAKE / SC.pairs).round(3); SC["share_REAL"] = (SC.REAL / SC.pairs).round(3); SC["share_UNCERTAIN"] = (SC.UNCERTAIN / SC.pairs).round(3)
SC.to_csv(f"{OUT}/final_sanity_checks.csv", index=False); N["sanity"] = SC.to_dict("records")
fig, ax = plt.subplots(figsize=(8, 3.6))
left = np.zeros(len(SC))
for v, c in (("REAL", C_REAL), ("FAKE", C_FAKE), ("UNCERTAIN", C_UNC)):
    w = (SC[v] / SC.pairs).values; ax.barh(range(len(SC)), w, left=left, color=c, label=v, height=.62)
    for i, (l, x) in enumerate(zip(left, w)):
        if x > .06: ax.text(l + x / 2, i, f"{x:.0%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    left += w
ax.set_yticks(range(len(SC)), [f"{t}\n(n={p})" for t, p in zip(SC.test, SC.pairs)], fontsize=8.5); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1], ["0%", "50%", "100%"]); ax.set_title("Sanity checks: genuine, fake and recombined pairs")
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.1))
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_sanity_checks.png"); plt.close(fig)

# ----------------------------------------------------------------- 4. bias audit
def audit(col):
    rows = []
    for g, s in meta.groupby(col).groups.items():
        s = np.array(list(s)); r = s[y[s] == 0]; f = s[y[s] == 1]
        ra, fa = r[ans[r]], f[ans[f]]
        rows.append(dict(group=g, clips=len(s), speakers=len(np.unique(groups[s])), real=len(r), fake=len(f),
                         accuracy_answered=correct[s[ans[s]]].mean() if ans[s].any() else np.nan,
                         false_alarm_rate=(verdict[ra] == "FAKE").mean() if len(ra) else np.nan,
                         miss_rate=(verdict[fa] == "REAL").mean() if len(fa) else np.nan,
                         uncertain_rate=1 - ans[s].mean()))
    return pd.DataFrame(rows).round(3)
BE, BS = audit("ethnicity"), audit("sex")
BE.to_csv(f"{OUT}/final_bias_ethnicity.csv", index=False); BS.to_csv(f"{OUT}/final_bias_sex.csv", index=False)
N["bias_ethnicity"], N["bias_sex"] = BE.to_dict("records"), BS.to_dict("records")
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw=dict(width_ratios=[len(BE), len(BS) + 1]))
for ax, B, ttl in ((axs[0], BE, "by ethnicity (folder label)"), (axs[1], BS, "by sex (folder label)")):
    x = np.arange(len(B)); w = .38
    ax.bar(x - w / 2, B.false_alarm_rate, w, color=C_REAL, label="false-alarm rate (real called FAKE)")
    ax.bar(x + w / 2, B.miss_rate, w, color=C_FAKE, label="miss rate (fake called REAL)")
    ax.set_xticks(x, [f"{g}\n{c} clips" for g, c in zip(B.group, B.clips)], fontsize=8); ax.set_ylim(0, 1); ax.set_title(ttl)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_bias_audit.png"); plt.close(fig)

# ----------------------------------------------------------------- 5. difficult cases
dist = np.abs(pm - np.array([models[f].cutoff for f in folds_of]))
wrong = ans & ~correct
hard = pd.DataFrame(dict(file=meta.file, category=cats, speaker=groups, ethnicity=meta.ethnicity, sex=meta.sex, truth=np.where(y == 1, "FAKE", "REAL"),
                         verdict=verdict, confidence=conf, p_voice=pv.round(3), p_face=pf.round(3), fused=pm.round(3),
                         suspected=[e["suspected"] for e in expl], caveats=["; ".join(e["caveats"]) for e in expl],
                         face_rate=face_rate.round(2), transcript_words=[len(t.split()) for t in transcript],
                         kind=np.where(wrong, "WRONG", np.where(~ans, "UNCERTAIN", "right")), distance_to_cutoff=dist.round(3)))
hard["why_hint"] = np.select(
    [hard.kind == "right", (hard.category == "FVRA") & (hard.kind != "right"), (hard.category == "RVFA") & (hard.kind != "right"),
     (hard.truth == "REAL") & (hard.kind != "right")],
    ["", "face swap with real voice: only the face stream can see it", "cloned voice with real face: only the voice stream can see it", "real clip scored fake-like by one stream"], "other")
hard = hard[hard.kind != "right"].assign(wrongness=lambda d: np.where(d.kind == "WRONG", 1, 0) * (1 + d.distance_to_cutoff)).sort_values(["wrongness", "distance_to_cutoff"], ascending=[False, True])
hard.drop(columns="wrongness").to_csv(f"{OUT}/final_difficult_cases.csv", index=False)
N["difficult_cases"] = dict(wrong=int(wrong.sum()), uncertain=int((~ans).sum()), top_wrong_by_category=hard[hard.kind == "WRONG"].category.value_counts().to_dict(),
                            wrong_with_caveat=int((hard[hard.kind == "WRONG"].caveats != "").sum()))

# ----------------------------------------------------------------- 6. Kaggle vs local comparison (from Exp 6 stability files)
comp = {}
for lab, folder in (("Kaggle features (3 frames)", "results_kaggle_env"), ("Local 8 frames", "results_local_8f"), ("Local 16 frames", "results_local")):
    p = os.path.join(ROOT, folder, "exp6", "exp6_stability_auc.csv")
    if os.path.exists(p):
        d = pd.read_csv(p, index_col=0); comp[lab] = d
if comp:
    fig, ax = plt.subplots(figsize=(7, 3.6)); labs = list(comp); x = np.arange(3); w = .26
    cols = ["#444", "#4c8cc0", "#8fb8de"]
    for j, l in enumerate(labs):
        d = comp[l]; ax.bar(x + (j - (len(labs) - 1) / 2) * w, [d.loc[s, "mean AUC"] for s in ("voice", "face", "fused")], w,
                            yerr=[d.loc[s, "std"] for s in ("voice", "face", "fused")], color=cols[j], label=l, capsize=3)
    ax.set_xticks(x, ["voice only", "face only", "fused"]); ax.set_ylim(.6, 1); ax.set_ylabel("pooled out-of-fold AUC")
    ax.set_title("Feature variants, 10 speaker partitions (mean ± std)"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_feature_variants.png"); plt.close(fig)
    N["feature_variants"] = {l: {s: [round(float(d.loc[s, "mean AUC"]), 3), round(float(d.loc[s, "std"]), 3)] for s in ("voice", "face", "fused")} for l, d in comp.items()}

# ----------------------------------------------------------------- numbers file
json.dump(N, open(f"{OUT}/FINAL_NUMBERS.json", "w"), indent=2, default=float)
md = [f"# Final system: out-of-fold numbers\n", f"{N['clips']} clips ({N['real']} real, {N['fake']} fake), {N['speakers']} speakers, 5-fold speaker-disjoint CV.\n",
      f"AUC voice {N['auc_voice']}, face {N['auc_face']}, fused {N['auc_fused']}. Coverage {N['coverage']:.1%}; accuracy on answered clips {N['accuracy_answered']:.1%}; "
      f"{N['uncertain_clips']} clips UNCERTAIN.\n", "## 3-way confusion\n", T.to_markdown(), "\n## Class-wise\n", CW.to_markdown(index=False),
      "\n## Per category\n", PC.to_markdown(index=False), "\n## Sanity checks\n", SC[["test", "pairs", "REAL", "FAKE", "UNCERTAIN", "expected"]].to_markdown(index=False),
      "\n## Bias: ethnicity\n", BE.to_markdown(index=False), "\n## Bias: sex\n", BS.to_markdown(index=False)]
open(f"{OUT}/FINAL_NUMBERS.md", "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md)); print("\nwrote", OUT, sorted(os.listdir(OUT)))
