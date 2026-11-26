"""Command line:  python verify.py path/to/clip.mp4 [--model final_system.joblib] [--no-asr]"""
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verifier_core import Verifier
from extract import Extractor


def verify(video, model, extractor):
    f = extractor.features(video)
    pv, pf, _ = model.score(f["V"], f["F"])
    return model.explain(pv[0], pf[0], f["face_rate"], f["transcript"]), f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_system.joblib"))
    ap.add_argument("--no-asr", action="store_true", help="skip the Whisper transcript (faster; it is not used in the score)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    m = Verifier.load(a.model)
    ex = Extractor(with_transcript=not a.no_asr)
    exp, _ = verify(a.video, m, ex)
    print(json.dumps(exp, indent=2) if a.json else exp["text"])
