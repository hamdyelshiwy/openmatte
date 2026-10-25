import argparse
import subprocess
import sys
import time

from . import __version__, engine


def _xy(s):
    try:
        x, y = (float(v) for v in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y in pixels, got {s!r}")
    return x, y


def main(argv=None):
    ap = argparse.ArgumentParser(prog="openmatte", description="AI mattes for images and video: SAM 2.1 / SAM 3 tracking, optional ViTMatte / MatAnyone2 / VideoMaMa refinement.")
    ap.add_argument("input", help="image, folder of images, or video file")
    ap.add_argument("-o", "--output", required=True, help="output folder for alpha PNGs")
    ap.add_argument("-p", "--point", type=_xy, action="append", default=[], help="foreground click X,Y in pixels (top-left origin); repeatable")
    ap.add_argument("-n", "--neg", type=_xy, action="append", default=[], help="background click X,Y; repeatable")
    ap.add_argument("-f", "--frame", type=int, default=0, help="index of the frame the clicks belong to (default 0)")
    ap.add_argument("-t", "--text", help="text prompt, e.g. 'person' (sam3 tracker)")
    ap.add_argument("--tracker", choices=list(engine.TRACKERS),
                    help="default: sam3 with --text, else sam2-small (tested); other sam2 sizes and sam3 are experimental")
    ap.add_argument("--refine", choices=engine.REFINERS, default="none",
                    help="soft alpha: vitmatte (Apache-2.0), matanyone2 / videomama (NON-COMMERCIAL licenses, separate setup)")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)
    tracker = a.tracker or ("sam3" if a.text else "sam2-small")
    if tracker == "sam3" and not a.text:
        ap.error("sam3 needs --text")
    if tracker != "sam3" and not a.point:
        ap.error(f"{tracker} needs at least one --point")

    t = time.time()
    frames, stems = engine.read_frames(a.input)
    if not 0 <= a.frame < len(frames):
        ap.error(f"--frame {a.frame} out of range (input has {len(frames)} frames)")
    h, w = frames[0].shape[:2]
    for x, y in a.point + a.neg:
        if not (0 <= x < w and 0 <= y < h):
            ap.error(f"point {x},{y} is outside the {w}x{h} frame")
    print(f"{len(frames)} frame(s) {w}x{h}, device={engine.device()}, tracker={tracker}, refine={a.refine}", flush=True)

    try:
        masks = engine.track(frames, tracker, a.point, a.neg, a.frame, a.text)
        print(f"tracked in {time.time() - t:.1f}s", flush=True)
        alphas = engine.refine(frames, masks, a.refine, a.frame)
    except RuntimeError as e:  # setup problems (missing access/install) -> one clear line, no traceback
        print(f"error: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:  # external refiner failed; its own output is printed above
        print(f"error: {a.refine} exited with code {e.returncode} (see its output above)", file=sys.stderr)
        return 1

    paths = engine.write_alphas(alphas, stems, a.output)
    print(f"wrote {len(paths)} alpha frame(s) to {a.output} in {time.time() - t:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
