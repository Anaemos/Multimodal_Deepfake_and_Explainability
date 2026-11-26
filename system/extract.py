"""Turn ONE video file into the features the verifier expects (same preprocessing as the Kaggle extraction notebook)."""
import subprocess
import numpy as np

CLIP_ID = "openai/clip-vit-base-patch32"
W2V_ID = "facebook/wav2vec2-base"
WHISPER_ID = "openai/whisper-base"
N_FRAMES, MAX_AUDIO_SEC = 3, 10


class Extractor:
    def __init__(self, with_transcript=True, device=None, whisper_lang="en"):
        import torch, cv2
        from transformers import CLIPModel, CLIPProcessor, Wav2Vec2Model, Wav2Vec2FeatureExtractor, pipeline
        self.torch, self.cv2 = torch, cv2
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip = CLIPModel.from_pretrained(CLIP_ID).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)
        self.w2v = Wav2Vec2Model.from_pretrained(W2V_ID).to(self.device).eval()
        self.w2v_fe = Wav2Vec2FeatureExtractor.from_pretrained(W2V_ID)
        self.asr, self.lang = None, whisper_lang
        if with_transcript:
            dt = torch.float16 if self.device == "cuda" else torch.float32
            try:
                self.asr = pipeline("automatic-speech-recognition", model=WHISPER_ID, dtype=dt, device=0 if self.device == "cuda" else -1)
            except TypeError:
                self.asr = pipeline("automatic-speech-recognition", model=WHISPER_ID, torch_dtype=dt, device=0 if self.device == "cuda" else -1)
        if not hasattr(cv2, "CascadeClassifier"):
            raise ImportError("OpenCV " + cv2.__version__ + " has no CascadeClassifier (OpenCV 5 dropped it). "
                              "Fix:  pip uninstall -y opencv-python-headless  then  pip install \"opencv-python-headless>=4.8,<5\"")
        import shutil
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg was not found on PATH. Install it (winget install Gyan.FFmpeg), then open a NEW terminal.")
        self.face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # ---- helpers (identical logic to 01_extract_features) ----
    def _as_tensor(self, x):
        if self.torch.is_tensor(x):
            return x
        for attr in ("pooler_output", "image_embeds", "text_embeds"):
            if hasattr(x, attr) and getattr(x, attr) is not None:
                return getattr(x, attr)
        return x[0]

    def read_frames(self, path, n=N_FRAMES):
        cv2 = self.cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(total * 0.1, total * 0.9, n).astype(int) if total >= n else [0]
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def crop_face(self, rgb, margin=0.25):
        cv2 = self.cv2
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        h, w = rgb.shape[:2]
        if len(faces) == 0:
            s = min(h, w); y0 = (h - s) // 2; x0 = (w - s) // 2
            return rgb[y0:y0 + s, x0:x0 + s], False
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        m = int(margin * max(fw, fh))
        return rgb[max(0, y - m):min(h, y + fh + m), max(0, x - m):min(w, x + fw + m)], True

    def load_audio(self, path, sr=16000, max_sec=MAX_AUDIO_SEC):
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-t", str(max_sec), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        out = subprocess.run(cmd, capture_output=True)
        if out.returncode != 0 or len(out.stdout) < 3200:
            return None
        return np.frombuffer(out.stdout, dtype=np.float32).copy()

    # ---- main entry ----
    def features(self, video_path):
        torch = self.torch
        frames = self.read_frames(video_path)
        if not frames:
            raise ValueError("Could not read any video frames from this file.")
        crops, found = zip(*[self.crop_face(f) for f in frames])
        with torch.no_grad():
            inp = self.clip_proc(images=list(crops), return_tensors="pt").to(self.device)
            e = self._as_tensor(self.clip.get_image_features(pixel_values=inp["pixel_values"]))
            e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy().mean(0)
        F = e / (np.linalg.norm(e) + 1e-9)
        face_rate = float(np.mean(found))

        wav = self.load_audio(video_path)
        if wav is None:
            raise ValueError("This file has no usable audio track (the verifier needs both voice and face).")
        with torch.no_grad():
            x = self.w2v_fe(wav, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            hs = self.w2v(x, output_hidden_states=True).hidden_states
            A = torch.stack([h.mean(dim=1).squeeze(0) for h in hs]).cpu().numpy()      # (13, 768)
        V = A.mean(0)

        text = ""
        if self.asr is not None:
            kw = {"task": "transcribe"}
            if self.lang:
                kw["language"] = self.lang
            text = self.asr({"raw": wav, "sampling_rate": 16000}, generate_kwargs=kw)["text"].strip()
        return dict(V=V, F=F, face_rate=face_rate, transcript=text, face_crops=crops, frames=frames)
