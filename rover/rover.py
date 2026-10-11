"""Rover side: frames -> detect -> track -> localize -> send.

Runs until Ctrl+C. No silent crashes: per-frame errors are logged and the
loop continues. All tunables come from config.json.
"""
import argparse
import csv
import logging
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
    tracker = Tracker(rcfg["track_iou_threshold"],
                      rcfg["track_confirm_hits"],
                      rcfg["track_max_misses"])
    sender = Sender(cfg["link"]["host"], cfg["link"]["listen"],
                    repeats=rcfg["hazard_repeats"],
                    repeat_gap_s=rcfg["hazard_repeat_gap_s"],
                    heartbeat_period_s=rcfg["heartbeat_period_s"],
                    max_packets_per_s=rcfg["max_packets_per_s"],
                    queue_size=rcfg["queue_size"])
    cam = cfg["camera"]
    fps = rcfg["fps"]
    frame_paths = load_frame_paths(cfg["paths"]["frames_dir"])
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
    log_writer.writerow(["mission_t", "kind", "x", "y", "conf", "seq"])

    start = time.monotonic()
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

            pose = poses[idx] if poses else {
                "x": 0.0, "y": 0.2 * t_mission, "z": 0.4,
                "yaw_deg": 0.0, "pitch_deg": 15.0}

            try:
                dets = detector.detect(frame, frame_idx=idx)
                positions = [localize_detection(d["box"], pose, cam)
                             for d in dets]
                events = tracker.update(dets, positions)
                now = time.monotonic() - start
                for ev in events:
                    sender.submit_hazard(ev["cls"], ev["x"], ev["y"],
                                         ev["conf"], now)
                    log_writer.writerow([f"{t_mission:.1f}", ev["cls"],
                                         f"{ev['x']:.2f}", f"{ev['y']:.2f}",
                                         f"{ev['conf']:.2f}", sender.seq])
                sender.tick(now, pose["x"], pose["y"], pose["yaw_deg"],
                            t_mission)
                log_csv.flush()
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
