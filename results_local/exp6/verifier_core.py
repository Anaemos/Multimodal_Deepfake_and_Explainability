"""Core of the multimodal deepfake verifier: train, score, decide (REAL / FAKE / UNCERTAIN) and explain.

Inputs are the frozen-encoder features used in Experiments 1-5:
  V : voice  = mean over the 13 wav2vec2-base layers of the time-averaged hidden states  (768,)
  F : face   = mean CLIP ViT-B/32 embedding of the face crops of 3 frames, re-normalised (512,)
No deep network is trained: two logistic regressions (one per stream) are averaged ("late mean").
"""
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

CS = [1e-3, 1e-2, 1e-1, 1, 10]
ABSTAIN_TARGET = 0.15


def _best_threshold(yv, pv):
    grid = np.linspace(0.05, 0.95, 91)
    sc = np.array([balanced_accuracy_score(yv, (pv >= t).astype(int)) for t in grid])
    cand = grid[sc >= sc.max() - 1e-12]
    return float(cand[np.argmin(np.abs(cand - 0.5))])


def _fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for C in CS:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
        a = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if best is None or a > best[0]:
            best = (a, C, clf)
    return best[2], best[1]


class Verifier:
    STREAMS = ("voice", "face")

    # ------------------------------------------------------------------ training
    def fit(self, V, F, y, groups, tr=None, va=None, seed=42):
        """Train on speaker-disjoint train/val. If tr/va are not given, 15% of the speakers go to validation."""
        n = len(y)
        if tr is None:
            tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(np.arange(n), groups=groups))
        X = {"voice": V, "face": F}
        self.scaler, self.clf, self.C, self.stream_thr = {}, {}, {}, {}
        pva = {}
        for s in self.STREAMS:
            self.scaler[s] = StandardScaler().fit(X[s][tr])
            clf, C = _fit_lr(self.scaler[s].transform(X[s][tr]), y[tr], self.scaler[s].transform(X[s][va]), y[va])
            self.clf[s], self.C[s] = clf, C
            pva[s] = clf.predict_proba(self.scaler[s].transform(X[s][va]))[:, 1]
            self.stream_thr[s] = _best_threshold(y[va], pva[s])
        fused = (pva["voice"] + pva["face"]) / 2
        self.cutoff = _best_threshold(y[va], fused)
        self.band = float(np.quantile(np.abs(fused - self.cutoff), ABSTAIN_TARGET))   # UNCERTAIN half-width
        self.val_auc = float(roc_auc_score(y[va], fused))
        return self

    # ------------------------------------------------------------------ inference
    def stream_scores(self, V, F):
        X = {"voice": np.atleast_2d(V), "face": np.atleast_2d(F)}
        return {s: self.clf[s].predict_proba(self.scaler[s].transform(X[s]))[:, 1] for s in self.STREAMS}

    def score(self, V, F):
        p = self.stream_scores(V, F)
        return p["voice"], p["face"], (p["voice"] + p["face"]) / 2

    def decide(self, fused):
        """REAL / FAKE / UNCERTAIN from the fused score (UNCERTAIN = low confidence = close to the cut-off)."""
        fused = np.asarray(fused)
        d = np.abs(fused - self.cutoff)
        return np.where(d <= self.band, "UNCERTAIN", np.where(fused >= self.cutoff, "FAKE", "REAL"))

    # ------------------------------------------------------------------ explanation
    def explain(self, p_voice, p_face, face_rate=1.0, transcript=None):
        """Turn the scores of ONE clip into a structured, plain-language explanation."""
        p_voice, p_face = float(p_voice), float(p_face)
        fused = (p_voice + p_face) / 2
        verdict = str(self.decide(fused))
        dist = abs(fused - self.cutoff)
        confidence = "low" if verdict == "UNCERTAIN" else ("medium" if dist <= 2 * self.band else "high")

        def stream(name, p):
            thr = self.stream_thr[name]
            flag = p >= thr
            gap = p - thr
            strength = "strongly" if abs(gap) >= 0.3 else "moderately" if abs(gap) >= 0.12 else "weakly"
            return dict(name=name, score=round(p, 3), threshold=round(thr, 3), flags_fake=bool(flag),
                        gap=round(gap, 3), strength=strength)

        sv, sf = stream("voice", p_voice), stream("face", p_face)
        v, f = sv["flags_fake"], sf["flags_fake"]
        pattern = ("both the voice and the face look manipulated" if v and f else
                   "the voice looks synthetic or cloned while the face looks genuine" if v else
                   "the face looks swapped or generated while the voice looks genuine" if f else
                   "neither the voice nor the face shows clear signs of manipulation")
        suspected = "both" if v and f else "voice" if v else "face" if f else "none"
        driver = max((sv, sf), key=lambda s: s["gap"])["name"] if verdict == "FAKE" else \
                 min((sv, sf), key=lambda s: s["gap"])["name"] if verdict == "REAL" else None

        caveats = []
        if verdict == "UNCERTAIN":
            caveats.append("the combined score is too close to the decision boundary to call either way")
        if v != f:
            caveats.append("the voice and face detectors disagree, so the decision rests on averaging two opposing signals")
        if face_rate < 0.67:
            caveats.append(f"a face was located in only {round(face_rate * 3)} of 3 sampled frames, so the face evidence is less reliable")
        if transcript is not None and len(str(transcript).split()) <= 3:
            caveats.append("the speech transcript is very short, so little speech was available")

        head = {"FAKE": "Verdict: FAKE", "REAL": "Verdict: REAL", "UNCERTAIN": "Verdict: UNCERTAIN"}[verdict]
        lines = [f"{head} (confidence: {confidence}; fused score {fused:.2f}, cut-off {self.cutoff:.2f})."]
        lines.append(f"Evidence: {pattern}.")
        for s in (sv, sf):
            lines.append(f"- {s['name'].capitalize()} model: score {s['score']:.2f} vs its threshold {s['threshold']:.2f}, "
                         f"{s['strength']} {'towards fake' if s['flags_fake'] else 'towards real'}.")
        if caveats:
            lines.append("Caveats: " + "; ".join(caveats) + ".")
        if transcript:
            lines.append(f'Words spoken (ASR, not used in the score): "{str(transcript).strip()}"')
        return dict(verdict=verdict, confidence=confidence, fused=round(fused, 3), cutoff=round(self.cutoff, 3),
                    band=round(self.band, 3), voice=sv, face=sf, suspected=suspected, driver=driver,
                    caveats=caveats, text="\n".join(lines))

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
