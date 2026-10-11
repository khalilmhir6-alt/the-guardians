"""Extract rover frames from a video into data/frames/ for the pipeline.

The rover reads 640x640 PNG frames named frame_NNNN.png from data/frames/.
This turns any rover video into that format so the trained model can run.

Usage:
    python tools/extract_frames.py <video.mp4> [--count 120] [--size 640]
"""
import argparse
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def extract(video, out_dir, count, size):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] could not open {video}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(total / count)) if count > 0 else 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("frame_*.png"):
        f.unlink()

    print(f"[INFO] {video}: {total} frames @ {fps:.0f} fps, taking every {step}")
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            saved += 1
            frame = cv2.resize(frame, (size, size))
            cv2.imwrite(str(out_dir / f"frame_{saved:04d}.png"), frame)
        i += 1
    cap.release()
    print(f"[INFO] wrote {saved} frames to {out_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="video -> data/frames/ frames")
    ap.add_argument("video")
    ap.add_argument("--count", type=int, default=120,
                    help="target number of frames (default 120 = 60 s @ 2 fps)")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--out", default=str(ROOT / "data" / "frames"))
    args = ap.parse_args()
    return extract(args.video, Path(args.out), args.count, args.size)


if __name__ == "__main__":
    sys.exit(main())
