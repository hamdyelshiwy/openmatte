"""End-to-end check: generate the example clip, matte it from one click, compare against ground truth.

Run: python tests/test_example.py   (downloads the SAM 2 small model on first run)
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

import make_example  # noqa: E402
from openmatte.cli import main  # noqa: E402


def iou(a, b):
    a, b = a > 127, b > 127
    return (a & b).sum() / max((a | b).sum(), 1)


def test_ball(min_iou=0.9):
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "ball"
        make_example.main(clip)
        out = Path(tmp) / "alpha"
        x, y = make_example.CLICK
        assert main([str(clip / "frames"), "-o", str(out), "-p", f"{x},{y}"]) == 0
        truths = sorted((clip / "truth").glob("*.png"))
        scores = [iou(cv2.imread(str(out / t.name), 0), cv2.imread(str(t), 0)) for t in truths]
    print(f"IoU min={min(scores):.3f} mean={np.mean(scores):.3f} over {len(scores)} frames")
    assert len(scores) == make_example.N
    assert min(scores) >= min_iou, scores


def check_nuke_output(min_iou=0.9):
    """Score the RGBA frames written by tests/nuke_smoke.py (alpha channel) against ground truth."""
    out = ROOT / "examples" / "out_nuke"
    truths = sorted((ROOT / "examples" / "ball" / "truth").glob("*.png"))
    scores = [iou(cv2.imread(str(out / t.name), cv2.IMREAD_UNCHANGED)[..., 3], cv2.imread(str(t), 0)) for t in truths]
    print(f"Nuke IoU min={min(scores):.3f} mean={np.mean(scores):.3f} over {len(scores)} frames")
    assert min(scores) >= min_iou, scores


if __name__ == "__main__":
    if "--nuke" in sys.argv:
        check_nuke_output()
    else:
        test_ball()
    print("ok")
