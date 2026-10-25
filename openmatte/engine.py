"""Matte engine.

Tracking (hard masks): SAM 2.1 from clicks, or SAM 3 from a text prompt.
Refinement (soft alpha, optional): ViTMatte, MatAnyone2 or VideoMaMa.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

TRACKERS = {
    "sam2-tiny": "facebook/sam2.1-hiera-tiny",
    "sam2-small": "facebook/sam2.1-hiera-small",
    "sam2-base": "facebook/sam2.1-hiera-base-plus",
    "sam2-large": "facebook/sam2.1-hiera-large",
    "sam3": "facebook/sam3",
}
REFINERS = ("none", "vitmatte", "matanyone2", "videomama")
VITMATTE_MODEL = "hustvl/vitmatte-small-composition-1k"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _autocast():
    return torch.autocast(device(), dtype=torch.bfloat16, enabled=device() == "cuda")


# ---------------------------------------------------------------- input / output

def read_frames(src):
    """Return (list of RGB uint8 frames, list of output stems) for an image, a folder of images, or a video."""
    src = Path(src)
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"no images found in {src}")
        return [_read_image(p) for p in files], [p.stem for p in files]
    if src.suffix.lower() in IMAGE_EXTS:
        return [_read_image(src)], [src.stem]
    if src.suffix.lower() in VIDEO_EXTS:
        cap, frames = cv2.VideoCapture(str(src)), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"could not decode video {src}")
        return frames, [f"{i:05d}" for i in range(len(frames))]
    raise ValueError(f"unsupported input: {src}")


def _read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not read {path}")
    if img.dtype != np.uint8:  # 16-bit PNG/TIFF from Nuke
        img = (img.astype(np.float32) / np.iinfo(img.dtype).max * 255).round().astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)


def write_alphas(alphas, stems, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for a, stem in zip(alphas, stems):
        if a is None:
            continue
        p = out_dir / f"{stem}.png"
        cv2.imwrite(str(p), (np.clip(a, 0, 1) * 255).round().astype(np.uint8))
        paths.append(p)
    return paths


def _dump(frames, folder, ext=".png"):
    folder.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(folder / f"{i:05d}{ext}"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------- tracking

def track(frames, tracker="sam2-small", points=(), negatives=(), frame_idx=0, text=None):
    """Hard masks (float32, 0/1) for every frame."""
    if tracker == "sam3":
        if not text:
            raise ValueError("sam3 needs a text prompt (--text)")
        return _track_sam3(frames, text)
    if not points:
        raise ValueError(f"{tracker} needs at least one --point")
    if len(frames) == 1:
        return [_sam2_image(frames[0], TRACKERS[tracker], points, negatives)]
    return _sam2_video(frames, TRACKERS[tracker], points, negatives, frame_idx)


def _prompt(points, negatives):
    coords = np.array(list(points) + list(negatives), dtype=np.float32)
    labels = np.array([1] * len(points) + [0] * len(negatives), dtype=np.int32)
    return coords, labels


def _sam2_image(frame, model, points, negatives):
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(model, device=device())
    coords, labels = _prompt(points, negatives)
    with torch.inference_mode(), _autocast():
        predictor.set_image(frame)
        masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)
    return masks[int(np.argmax(scores))].astype(np.float32)


def _sam2_video(frames, model, points, negatives, frame_idx):
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    predictor = SAM2VideoPredictor.from_pretrained(model, device=device())
    coords, labels = _prompt(points, negatives)
    masks = [None] * len(frames)
    tmp = Path(tempfile.mkdtemp(prefix="openmatte_"))
    try:
        _dump(frames, tmp, ".jpg")  # SAM 2's video loader wants JPEGs named by index
        with torch.inference_mode(), _autocast():
            state = predictor.init_state(video_path=str(tmp))
            predictor.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=1, points=coords, labels=labels)
            for reverse in (False, True):
                for i, _, logits in predictor.propagate_in_video(state, start_frame_idx=frame_idx, reverse=reverse):
                    masks[i] = (logits[0, 0] > 0).float().cpu().numpy()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return masks


def _track_sam3(frames, text):
    """Every object matching `text`, merged into one mask per frame. facebook/sam3 is gated on Hugging Face."""
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    dtype = torch.bfloat16 if device() == "cuda" else torch.float32
    try:
        model = Sam3VideoModel.from_pretrained(TRACKERS["sam3"]).to(device(), dtype=dtype)
        processor = Sam3VideoProcessor.from_pretrained(TRACKERS["sam3"])
    except OSError as e:
        if "gated" not in str(e).lower():
            raise
        raise RuntimeError("SAM 3 weights are gated: request access at https://huggingface.co/facebook/sam3, "
                           "then run `hf auth login` in this environment") from None
    session = processor.init_video_session(video=frames, inference_device=device(), processing_device="cpu",
                                           video_storage_device="cpu", dtype=dtype)
    session = processor.add_text_prompt(inference_session=session, text=text)
    h, w = frames[0].shape[:2]
    masks = [np.zeros((h, w), np.float32) for _ in frames]
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(inference_session=session, max_frame_num_to_track=len(frames)):
            m = processor.postprocess_outputs(session, out)["masks"]
            if len(m):
                masks[out.frame_idx] = m.any(dim=0).float().cpu().numpy()
    return masks


# ---------------------------------------------------------------- refinement

def refine(frames, masks, method="none", frame_idx=0):
    if method == "none":
        return masks
    if method == "vitmatte":
        return _vitmatte(frames, masks)
    if method == "matanyone2":
        return _matanyone2(frames, masks[frame_idx], frame_idx)
    if method == "videomama":
        return _videomama(frames, masks)
    raise ValueError(f"unknown refiner {method!r}")


def trimap(mask, band=None):
    """0 = background, 128 = unknown edge band, 255 = foreground."""
    m = (mask > 0.5).astype(np.uint8)
    band = band or max(3, round(0.01 * min(m.shape)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    t = np.full(m.shape, 128, np.uint8)
    t[cv2.dilate(m, k) == 0] = 0
    t[cv2.erode(m, k) == 1] = 255
    return t


def _vitmatte(frames, masks):
    """Per-frame soft alpha from an automatic trimap (Apache-2.0)."""
    from PIL import Image
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    processor = VitMatteImageProcessor.from_pretrained(VITMATTE_MODEL)
    model = VitMatteForImageMatting.from_pretrained(VITMATTE_MODEL).to(device()).eval()
    alphas = []
    for f, m in zip(frames, masks):
        if m is None or not m.any():
            alphas.append(m)
            continue
        t = trimap(m)
        inputs = processor(images=Image.fromarray(f), trimaps=Image.fromarray(t), return_tensors="pt").to(device())
        with torch.inference_mode():
            a = model(**inputs).alphas[0, 0, : f.shape[0], : f.shape[1]].float().cpu().numpy()
        a[t == 0], a[t == 255] = 0, 1  # keep certain regions exact
        alphas.append(a)
    return alphas


def _external(name):
    """(python, repo_dir) for a refiner that lives in its own clone/environment."""
    repo = os.environ.get(f"OPENMATTE_{name}_DIR")
    if not repo or not Path(repo).is_dir():
        raise RuntimeError(f"set OPENMATTE_{name}_DIR to your clone of the {name.title()} repository (see README)")
    return os.environ.get(f"OPENMATTE_{name}_PYTHON", sys.executable), repo


def _matanyone2(frames, mask, frame_idx):
    """Soft alpha propagated forward from frame_idx; earlier frames are None. S-Lab License: non-commercial.

    Runs in MatAnyone2's own environment: its pinned dependencies conflict with transformers 5.
    """
    python, repo = _external("MATANYONE2")
    tmp = Path(tempfile.mkdtemp(prefix="openmatte_ma_"))
    try:
        _dump(frames[frame_idx:], tmp / "frames")
        cv2.imwrite(str(tmp / "mask.png"), (mask * 255).astype(np.uint8))
        # cwd=repo so its checkpoint lands in <repo>/pretrained_models and is reused
        subprocess.run([python, "-c", "from matanyone2.cli import app; app()", "-i", str(tmp / "frames"),
                        "-m", str(tmp / "mask.png"), "-o", str(tmp / "out"), "--save-image"],
                       cwd=repo, env={**os.environ, "PYTHONPATH": repo}, check=True)
        files = sorted((tmp / "out" / "frames" / "pha").glob("*.png"))
        if len(files) != len(frames) - frame_idx:
            raise RuntimeError(f"MatAnyone2 returned {len(files)} alpha frames, expected {len(frames) - frame_idx}")
        alphas = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for p in files]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return [None] * frame_idx + alphas


def _videomama(frames, masks, chunk=16):
    """VideoMaMa (code CC BY-NC 4.0, weights Stability AI Community License) in 16-frame chunks.

    Runs the script from a clone of cvlab-kaist/VideoMaMa (OPENMATTE_VIDEOMAMA_DIR). Only the SVD parts it
    loads (feature extractor, fp16 image encoder and VAE) and the VideoMaMa UNet are downloaded.
    """
    from huggingface_hub import snapshot_download

    python, repo = _external("VIDEOMAMA")
    svd = snapshot_download("stabilityai/stable-video-diffusion-img2vid-xt", allow_patterns=[
        "model_index.json", "feature_extractor/*", "image_encoder/config.json", "image_encoder/*fp16*",
        "vae/config.json", "vae/*fp16*"])
    ckpt = snapshot_download("SammyLim/VideoMaMa")
    h, w = frames[0].shape[:2]
    tmp = Path(tempfile.mkdtemp(prefix="openmatte_vm_"))
    try:
        starts = range(0, len(frames), chunk)
        for c in starts:  # one "video" per chunk; VideoMaMa repeats the last frame to fill a short chunk
            _dump(frames[c:c + chunk], tmp / "image" / f"c{c:06d}")
            for i, m in enumerate(masks[c:c + chunk]):
                (tmp / "mask" / f"c{c:06d}").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(tmp / "mask" / f"c{c:06d}" / f"{i:05d}.png"), ((m > 0.5) * 255).astype(np.uint8))
        subprocess.run([python, "inference_onestep_folder.py", "--base_model_path", svd,
                        "--unet_checkpoint_path", ckpt,
                        "--image_root_path", str(tmp / "image"), "--mask_root_path", str(tmp / "mask"),
                        "--output_dir", str(tmp / "out"), "--num_frames", str(chunk)],
                       cwd=repo, check=True)
        alphas = []
        for c in starts:
            n = len(frames[c:c + chunk])
            for i in range(n):
                p = tmp / "out" / "results" / f"c{c:06d}" / f"frame_{i:04d}.png"
                a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if a is None:
                    raise RuntimeError(f"VideoMaMa did not produce {p}")
                alphas.append(cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return alphas
