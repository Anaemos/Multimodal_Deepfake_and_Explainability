"""Web demo.  pip install gradio  ;  python app.py   (then open the printed local URL)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradio as gr
from verifier_core import Verifier
from extract import Extractor

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = Verifier.load(os.environ.get("MODEL_PATH", os.path.join(HERE, "final_system.joblib")))
EXTRACTOR = Extractor(with_transcript=True)
COLORS = {"REAL": "#2e8b57", "FAKE": "#c0392b", "UNCERTAIN": "#b8860b"}


def run(video):
    if video is None:
        return "<p>Upload a short video clip with a visible face and speech.</p>", "", None
    try:
        f = EXTRACTOR.features(video)
    except Exception as e:
        return f"<p style='color:#c0392b'>{e}</p>", "", None
    pv, pf, _ = MODEL.score(f["V"], f["F"])
    e = MODEL.explain(pv[0], pf[0], f["face_rate"], f["transcript"])
    c = COLORS[e["verdict"]]
    bar = lambda s: (f"<div style='margin:6px 0'><b>{s['name'].capitalize()}</b> fake score {s['score']:.2f} "
                     f"(threshold {s['threshold']:.2f})<div style='background:#ddd;height:10px;border-radius:5px;position:relative'>"
                     f"<div style='width:{int(s['score']*100)}%;background:{'#c0392b' if s['flags_fake'] else '#2e8b57'};height:10px;border-radius:5px'></div>"
                     f"<div style='position:absolute;left:{int(s['threshold']*100)}%;top:-3px;height:16px;border-left:2px solid #333'></div></div></div>")
    html = (f"<div style='font-size:28px;font-weight:700;color:{c}'>{e['verdict']}</div>"
            f"<div>confidence: <b>{e['confidence']}</b> · fused score {e['fused']:.2f} · cut-off {e['cutoff']:.2f}</div>"
            + bar(e["voice"]) + bar(e["face"]))
    return html, e["text"], [c for c in f["face_crops"]]


with gr.Blocks(title="Multimodal deepfake verifier") as demo:
    gr.Markdown("# Multimodal deepfake verifier\nVoice + face, with an explanation. Research prototype trained on FakeAVCeleb; "
                "results on other kinds of video are untested.")
    v = gr.Video(label="Clip (about 10 s, one speaker)")
    b = gr.Button("Verify")
    out_html = gr.HTML()
    out_txt = gr.Textbox(label="Explanation", lines=9)
    gal = gr.Gallery(label="Face crops the model saw", columns=3, height=180)
    b.click(run, v, [out_html, out_txt, gal])

if __name__ == "__main__":
    demo.launch()
