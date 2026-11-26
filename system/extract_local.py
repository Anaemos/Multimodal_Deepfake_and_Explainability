"""Re-extract features for the SAME clips as the Kaggle run, on this machine, with MORE video frames per clip.

    python extract_local.py --clips ..\\features_kaggle\\meta.csv --dataset ..\\FakeAVCeleb_v1.2 --out ..\\features_local --frames 16

Output (in --out): features.npz and meta.csv with the same layout the experiment scripts read, plus
    image_frames (N, FRAMES, 512)  per-frame CLIP face embeddings (so any number of frames <= FRAMES can be pooled later)
    face_found   (N, FRAMES)       1 if a face was detected in that frame (0 = centre-crop fallback)
`image` is the mean over ALL extracted frames, so the unchanged experiment scripts already see the richer face feature.
Resumable: finished batches are kept in <out>/parts; re-run the same command after an interruption.
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import Extractor


def local_path(kaggle_path, root):
    s = str(kaggle_path).replace("\\", "/")
    rel = s.split("FakeAVCeleb_v1.2/")[-1]
    p = Path(root) / rel
    if p.exists():
        return p
    hits = list(Path(root).rglob(Path(rel).name))            # fallback: same file name somewhere under root
    hits = [h for h in hits if h.parent.name == Path(rel).parent.name] or hits
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="meta.csv of the Kaggle run (defines WHICH clips, in which order)")
    ap.add_argument("--dataset", required=True, help="local FakeAVCeleb_v1.2 folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args()

    meta = pd.read_csv(a.clips).sort_values("clip_id").reset_index(drop=True)
    out = Path(a.out); parts = out / "parts"; parts.mkdir(parents=True, exist_ok=True)
    paths = [local_path(p, a.dataset) for p in meta.path]
    missing = [i for i, p in enumerate(paths) if p is None]
    if missing:
        sys.exit(f"{len(missing)} clips from meta.csv were not found under {a.dataset} (first: {meta.path[missing[0]]})")
    print(f"{len(meta)} clips found locally; {a.frames} frames per clip")

    ex = Extractor(with_transcript=not a.no_asr)
    torch = ex.torch
    t0 = time.time()
    for s in range(0, len(meta), a.batch):
        pf = parts / f"part_{s:06d}.npz"
        if pf.exists():
            continue
        fr, ff, au, ok_i, ok_a, texts = [], [], [], [], [], []
        for i in range(s, min(s + a.batch, len(meta))):
            p = str(paths[i])
            frames = ex.read_frames(p, n=a.frames)
            if frames:
                crops, found = zip(*[ex.crop_face(f) for f in frames])
                with torch.no_grad():
                    inp = ex.clip_proc(images=list(crops), return_tensors="pt").to(ex.device)
                    e = ex._as_tensor(ex.clip.get_image_features(pixel_values=inp["pixel_values"]))
                    e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
                k = a.frames - len(e)                                    # unreadable frames -> repeat the last one
                if k > 0:
                    e = np.vstack([e, np.repeat(e[-1:], k, 0)]); found = list(found) + [found[-1]] * k
                fr.append(e.astype(np.float32)); ff.append(np.array(found, np.int8)); ok_i.append(1)
            else:
                fr.append(np.zeros((a.frames, 512), np.float32)); ff.append(np.zeros(a.frames, np.int8)); ok_i.append(0)
            wav = ex.load_audio(p)
            if wav is None:
                au.append(np.zeros((13, 768), np.float32)); ok_a.append(0); texts.append("")
            else:
                with torch.no_grad():
                    x = ex.w2v_fe(wav, sampling_rate=16000, return_tensors="pt").input_values.to(ex.device)
                    hs = ex.w2v(x, output_hidden_states=True).hidden_states
                    au.append(torch.stack([h.mean(dim=1).squeeze(0) for h in hs]).cpu().numpy().astype(np.float32))
                ok_a.append(1)
                texts.append(ex.asr({"raw": wav, "sampling_rate": 16000}, generate_kwargs={"task": "transcribe", "language": ex.lang})["text"].strip()
                             if ex.asr is not None else "")
        if ex.asr is not None:
            with torch.no_grad():
                inp = ex.clip_proc.tokenizer([t if t else " " for t in texts], padding=True, truncation=True, max_length=77, return_tensors="pt").to(ex.device)
                te = torch.nn.functional.normalize(ex._as_tensor(ex.clip.get_text_features(**inp)), dim=-1).cpu().numpy()
        else:
            te = np.zeros((len(texts), 512), np.float32)
        np.savez(pf, image_frames=np.stack(fr), face_found=np.stack(ff), audio_layers=np.stack(au), text=te,
                 ok_img=np.array(ok_i), ok_aud=np.array(ok_a), transcript=np.array(texts, dtype=object), allow_pickle=True)
        done = min(s + a.batch, len(meta))
        el = time.time() - t0
        print(f"{done}/{len(meta)} clips  ({el / 60:.1f} min elapsed)", flush=True)

    P = [np.load(f, allow_pickle=True) for f in sorted(parts.glob("part_*.npz"))]
    cat = lambda k: np.concatenate([p[k] for p in P])
    frames_arr = cat("image_frames")
    img = frames_arr.mean(1); img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-9)
    meta["ok_img"], meta["ok_aud"] = cat("ok_img"), cat("ok_aud")
    meta["face_rate"] = cat("face_found").mean(1)
    meta["transcript"] = cat("transcript")
    meta["path_local"] = [str(p) for p in paths]
    np.savez(out / "features.npz", clip_id=meta.clip_id.values, image=img, image_frames=frames_arr, face_found=cat("face_found"),
             audio_layers=cat("audio_layers"), text=cat("text"))
    meta.to_csv(out / "meta.csv", index=False)
    print(f"done: {len(meta)} clips -> {out}  | mean face_rate {meta.face_rate.mean():.3f}")


if __name__ == "__main__":
    main()
