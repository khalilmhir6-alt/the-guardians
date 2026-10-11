"""Rover side: frames -> detect -> track -> localize -> send.

Runs until Ctrl+C. No silent crashes: per-frame errors are logged and the
loop continues. All tunables come from config.json.
"""
import argparse
import csv
import logging
import math
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config          # noqa: E402
from rover.detector import make_detector        # noqa: E402
from rover.localize import localize_detection   # noqa: E402
from rover.sender import Sender                 # noqa: E402
from rover.tracker import Tracker               # noqa: E402

log = logging.getLogger("crater.rover")


def setup_logging(logs_dir: str):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(),):
        h.setFormatter(fmt)
        root.addHandler(h)
    fh = RotatingFileHandler(Path(logs_dir) / "rover.log",
                             maxBytes=1_000_000, backupCount=3)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def load_frame_paths(frames_dir: str):
    p = Path(frames_dir)
    if not p.exists():
        return []
    return sorted(p.glob("frame_*.png"))


def ensure_frames(frames_dir: str, feed_video: str, count: int = 120,
                  size: int = 640):
    """Return frame paths, extracting them from the sample feed video the
    first time if data/frames/ is empty. Keeps a fresh clone runnable with
    the trained model without shipping 48 MB of PNGs.
    """
    existing = load_frame_paths(frames_dir)
    if existing:
        return existing
    p = Path(frames_dir)
    feed = Path(feed_video)
    if not feed.is_absolute():
        feed = Path(__file__).resolve().parent.parent / feed_video
    if not feed.exists():
        return []
    try:
        import cv2
    except ImportError:
        log.warning("no frames and no opencv to extract from %s", feed)
        return []
    cap = cv2.VideoCapture(str(feed))
    if not cap.isOpened():
        log.warning("could not open feed video %s", feed)
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(total / count)) if count > 0 else 1
    p.mkdir(parents=True, exist_ok=True)
    log.info("data/frames empty - extracting frames from %s", feed)
    i = saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                saved += 1
                cv2.imwrite(str(p / f"frame_{saved:04d}.png"),
                            cv2.resize(frame, (size, size)))
            i += 1
    finally:
        cap.release()
    log.info("extracted %d frames from %s", saved, feed)
    return load_frame_paths(frames_dir)


def main():
    ap = argparse.ArgumentParser(description="C.R.A.T.E.R. rover")
    ap.add_argument("--config", default=None)
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after N seconds (default: run forever)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["paths"]["root"] + "/logs")
    log.info("rover starting (detector=%s)", cfg.get("detector"))

    detector = make_detector(cfg)
    rcfg = cfg["rover"]
    lcfg = cfg.get("localization", {})
    tracker = Tracker(rcfg["track_iou_threshold"],
                      rcfg["track_confirm_hits"],
                      rcfg["track_max_misses"],
                      max_report_range_m=rcfg.get("max_report_range_m", 8.0),
                      min_merge_radius_m=lcfg.get("min_uncertainty_m", 0.5),
                      pixel_noise_px=lcfg.get("pixel_noise_px", 3.0),
                      camera_height_m=lcfg.get("camera_height_m", 0.4),
                      cam=cfg["camera"])
    sender = Sender(cfg["link"]["host"], cfg["link"]["listen"],
                    repeats=rcfg["hazard_repeats"],
                    repeat_gap_s=rcfg["hazard_repeat_gap_s"],
                    heartbeat_period_s=rcfg["heartbeat_period_s"],
                    max_packets_per_s=rcfg["max_packets_per_s"],
                    queue_size=rcfg["queue_size"])
    cam = cfg["camera"]
    fps = rcfg["fps"]
    frame_paths = ensure_frames(cfg["paths"]["frames_dir"],
                                cfg["paths"].get("feed_video",
                                                 "samples/rover_feed.mp4"))
    log.info("frames available: %d", len(frame_paths))

    cv2 = None
    try:
        import cv2 as _cv2
        cv2 = _cv2
    except ImportError:
        log.warning("opencv not installed - frames not read (ok for fake detector)")

    poses = getattr(detector, "poses", [])
    log_path = Path(cfg["paths"]["earth_log"]).parent / "rover_log.csv"
    log_csv = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_csv)
    log_writer.writerow(["mission_t", "kind", "x", "y", "conf", "err", "seq"])

    start = time.monotonic()
    last_status = 0.0
    n = 0
    running = True
    try:
        while running:
            loop_t0 = time.monotonic()
            if args.duration and loop_t0 - start >= args.duration:
                log.info("duration reached, stopping")
                break

            t_mission = n / fps
            idx = n % len(frame_paths) if frame_paths else n
            frame = None
            if frame_paths and cv2 is not None:
                frame = cv2.imread(str(frame_paths[idx]))

            pose = poses[idx % len(poses)] if poses else {
                "x": 0.0, "y": 0.2 * t_mission, "z": 0.4,
                "yaw_deg": 0.0, "pitch_deg": 15.0}

            try:
                dets = detector.detect(frame, frame_idx=idx)
                positions = [localize_detection(d["box"], pose, cam)
                             for d in dets]
                ranges = [None if xy is None else
                          math.hypot(xy[0] - pose["x"], xy[1] - pose["y"])
                          for xy in positions]
                events = tracker.update(dets, positions, ranges)
                now = time.monotonic() - start
                for ev in events:
                    sender.submit_hazard(ev["cls"], ev["x"], ev["y"],
                                         ev["conf"], now, ev["err"], t_mission)
                    log_writer.writerow([f"{t_mission:.1f}", ev["cls"],
                                         f"{ev['x']:.2f}", f"{ev['y']:.2f}",
                                         f"{ev['conf']:.2f}", ev["err"],
                                         sender.seq])
                sender.tick(now, pose["x"], pose["y"], pose["yaw_deg"],
                            t_mission)
                log_csv.flush()
                if now - last_status >= 10.0:
                    log.info("status t=%.0fs frames=%d tracker=%s sender=%s",
                             t_mission, n, tracker.stats, sender.stats)
                    last_status = now
            except Exception:
                log.exception("frame %d failed, continuing", n)

            n += 1
            nap = 1.0 / fps - (time.monotonic() - loop_t0)
            if nap > 0:
                time.sleep(nap)
    except KeyboardInterrupt:
        log.info("Ctrl+C - shutting down cleanly")
    finally:
        sender.close()
        log_csv.close()
        log.info("rover stopped: frames=%d tracker=%s sender=%s",
                 n, tracker.stats, sender.stats)


if __name__ == "__main__":
    main()
