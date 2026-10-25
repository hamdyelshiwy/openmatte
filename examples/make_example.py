"""Generate a small synthetic clip (a ball moving over a busy background) plus ground-truth masks.

Everything here is procedurally generated, so it is free to redistribute.
Usage: python examples/make_example.py  ->  examples/ball/frames/*.png, examples/ball/truth/*.png
"""
from pathlib import Path

import cv2
import numpy as np

W, H, N, R = 640, 360, 24, 50
CLICK = (120, 180)  # the ball's centre on frame 0, used in the README command


def main(out=Path(__file__).parent / "ball"):
    rng = np.random.default_rng(7)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 6)
    bg = cv2.normalize(bg, None, 40, 200, cv2.NORM_MINMAX)
    for i in range(0, W, 80):  # stripes so the background is not trivially uniform
        cv2.rectangle(bg, (i, 0), (i + 30, H), (90, 120, 60), -1)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    for f in range(N):
        cx = CLICK[0] + f * 16
        cy = int(CLICK[1] + 60 * np.sin(f / 4))
        img, mask = bg.copy(), np.zeros((H, W), np.uint8)
        cv2.circle(mask, (cx, cy), R, 255, -1)
        cv2.circle(img, (cx, cy), R, (30, 60, 220), -1)
        cv2.circle(img, (cx - 15, cy - 15), R // 3, (120, 160, 255), -1)  # highlight
        cv2.imwrite(str(out / "frames" / f"{f:04d}.png"), img)
        cv2.imwrite(str(out / "truth" / f"{f:04d}.png"), mask)
    print(f"wrote {N} frames to {out}")


if __name__ == "__main__":
    main()
