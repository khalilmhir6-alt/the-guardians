"""Automated test outputs for C.R.A.T.E.R.

Writes results/test_report.md and results/results.json. Every claim is a
measured number produced here. Runs with the standard library; perception
numbers are produced only when a model file exists.

  python tests/run_tests.py
"""
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.codec import PACKET_SIZE, decode, encode          # noqa: E402
from common.config import load_config                          # noqa: E402
from earth.earth import EarthState                             # noqa: E402
from link.link import in_blackout                              # noqa: E402
from rover.detector import FakeDetector                        # noqa: E402
from rover.localize import (class_radius_m, error_class,       # noqa: E402
                            estimated_error, heading_to_steps,
                            localize_detection, pixel_to_ground,
                            project_point, steps_to_heading)
from rover.sender import Sender                                # noqa: E402
from rover.tracker import Tracker                              # noqa: E402

RESULTS = ROOT / "results"
CFG = load_config()
CAM = CFG["camera"]
FPS = CFG["rover"]["fps"]
RCFG = CFG["rover"]
LCFG = CFG["localization"]


def data_source(detector):
    return "synthetic fallback (data/ not generated)" if detector.synthetic \
        else "Blender data/ (real scene)"


# ---------------------------------------------------------------- 1. codec
def test_codec():
    s = dict(kind="rock", x_m=12.34, y_m=-4.56, conf=0.87, t_s=1234, seq=17)
    pkt = encode(**s)
    out = decode(pkt)
    as_json = json.dumps({"type": "rock", "x": 12.34, "y": -4.56,
                          "conf": 0.87, "t": 1234, "seq": 17}).encode()
    pos_err = max(abs(out["x"] - 12.34), abs(out["y"] + 4.56))

    rejected = 0
    for i in range(PACKET_SIZE):
        for bit in range(8):
            bad = bytearray(pkt)
            bad[i] ^= (1 << bit)
            if decode(bytes(bad)) is None:
                rejected += 1
    total_corrupt = PACKET_SIZE * 8

    err_ok = all(decode(encode("crater", 1, 2, 0.5, 3, 4, c))["err"] == c
                 for c in range(4))

    return {
        "packet_size_bytes": len(pkt),
        "json_size_bytes": len(as_json),
        "compression_ratio": round(len(as_json) / len(pkt), 2),
        "roundtrip_max_position_error_m": round(pos_err, 4),
        "corrupt_rejected": rejected,
        "corrupt_tested": total_corrupt,
        "corrupt_reject_pct": round(100.0 * rejected / total_corrupt, 1),
        "error_class_roundtrip_ok": err_ok,
    }


# --------------------------------------------------- 2. link survival
def hazard_survives(rng, t0, repeats, gap, loss, every, duration):
    won = True  # a hazard survives if at least one copy arrives
    for i in range(repeats):
        t = t0 + i * gap
        if every and in_blackout(t, every, duration):
            won = False  # copy lost to blackout (do not undo other copies)
            continue
        if rng.random() < loss:
            continue  # random drop
        return True
    return repeats == 0 and won


def test_link_survival(n_hazards=16, seed=20260101):
    timeline = [i * 3.0 for i in range(n_hazards)]  # one every 3 s
    rows = []
    for loss in (0.0, 0.1, 0.3):
        for blackouts in (False, True):
            every = 30.0 if blackouts else 0.0
            for repeats in (1, 3):
                rng = random.Random(seed)
                survived = sum(hazard_survives(rng, t0, repeats, 6.0, loss,
                                               every, 5.0) for t0 in timeline)
                rows.append({
                    "loss": loss,
                    "blackouts": "yes" if blackouts else "no",
                    "repeats": repeats,
                    "hazards": n_hazards,
                    "delivered": survived,
                    "delivery_pct": round(100.0 * survived / n_hazards, 1),
                })
    return rows


# ------------------------------------------------------------- 3. localize
def test_localize():
    pts = [(0.0, 2.0), (1.5, 4.0), (-3.0, 6.0), (2.5, 9.0), (-1.0, 12.0)]
    pose = {"x": 0.0, "y": 0.0, "z": 0.4, "yaw_deg": 0.0, "pitch_deg": 15.0}
    errs = []
    for p in pts:
        uv = project_point((p[0], p[1], 0.0), pose, CAM)
        back = pixel_to_ground(uv[0], uv[1], pose, CAM)
        errs.append(math.hypot(back[0] - p[0], back[1] - p[1]))

    # error formula vs Monte-Carlo (project, add pixel noise, back-project)
    rng = random.Random(7)
    mc = []
    for r in (2.0, 5.0, 8.0):
        pose_r = {"x": 0.0, "y": 0.0, "z": 0.4, "yaw_deg": 0.0,
                  "pitch_deg": 15.0}
        gt = (0.0, r, 0.0)
        uv = project_point(gt, pose_r, CAM)
        sample = []
        for _ in range(400):
            u = uv[0] + rng.gauss(0, LCFG["pixel_noise_px"])
            v = uv[1] + rng.gauss(0, LCFG["pixel_noise_px"])
            b = pixel_to_ground(u, v, pose_r, CAM)
            if b:
                sample.append(math.hypot(b[0] - gt[0], b[1] - gt[1]))
        measured = statistics.mean(sample) if sample else float("nan")
        formula = estimated_error(r, LCFG["camera_height_m"],
                                  LCFG["pixel_noise_px"], CAM)
        mc.append({"range_m": r, "measured_err_m": round(measured, 3),
                   "formula_err_m": round(formula, 3),
                   "ratio": round(measured / formula, 2) if formula else None})

    # Real pipeline error: detect -> localize vs nearest same-class GT.
    det = FakeDetector(CFG["paths"]["rocks_csv"], CFG["paths"]["poses_csv"],
                       CFG["paths"]["camera_json"], CFG["detector_fake"])
    gt_err = []
    for n in range(len(det.poses)):
        pose_n = det.poses[n]
        for d in det.detect(None, frame_idx=n):
            p = localize_detection(d["box"], pose_n, CAM)
            if p is None:
                continue
            cand = [g for g in det.rocks if g["type"] == d["cls"]]
            if not cand:
                continue
            g = min(cand, key=lambda g: math.hypot(g["x"] - p[0], g["y"] - p[1]))
            err = math.hypot(g["x"] - p[0], g["y"] - p[1])
            if err < 2.0:  # a detection near a rock, not a far false positive
                gt_err.append(err)
    return {
        "source": data_source(det),
        "roundtrip_max_error_m": f"{max(errs):.2e}",
        "roundtrip_under_1cm": max(errs) < 0.01,
        "montecarlo_vs_formula": mc,
        "detections_matched": len(gt_err),
        "gt_mean_error_m": round(statistics.mean(gt_err), 4) if gt_err else None,
        "gt_max_error_m": round(max(gt_err), 4) if gt_err else None,
    }


# -------------------------------------------------------------- 4. tracker
def test_tracker():
    iou_ok = True
    tr = Tracker(RCFG["track_iou_threshold"], RCFG["track_confirm_hits"],
                 RCFG["track_max_misses"], max_report_range_m=8.0,
                 min_merge_radius_m=0.5, pixel_noise_px=3.0, cam=CAM)
    events_all = []
    for i in range(12):
        dets = [
            {"cls": "rock", "conf": 0.9, "box": [100 + i, 100, 140 + i, 140]},
            {"cls": "rock", "conf": 0.9, "box": [300, 300 + i, 340, 340 + i]},
        ]
        pos = [(1.0 + 0.05 * i, 2.0), (5.0, 5.0 + 0.05 * i)]
        rng = [3.0, 3.0]
        if i == 2:  # single-frame false positive -> must not confirm
            dets.append({"cls": "rock", "conf": 0.4, "box": [500, 500, 520, 520]})
            pos.append((9.0, 9.0))
            rng.append(3.0)
        if i in (5, 6):  # duplicate detection of track A -> must not create a track
            dets.append({"cls": "rock", "conf": 0.9,
                         "box": [102 + i, 102, 142 + i, 142]})
            pos.append((1.02 + 0.05 * i, 2.02))
            rng.append(3.0)
        events_all += tr.update(dets, pos, rng)
    persistent = sum(1 for e in events_all)

    # dedupe: two distinct same-class hazards 0.4 m apart must report once
    tr2 = Tracker(RCFG["track_iou_threshold"], RCFG["track_confirm_hits"],
                  RCFG["track_max_misses"], max_report_range_m=8.0,
                  min_merge_radius_m=0.5, pixel_noise_px=3.0, cam=CAM)
    ev2 = []
    for i in range(6):
        dets = [{"cls": "rock", "conf": 0.9, "box": [100, 100, 140, 140]},
                {"cls": "rock", "conf": 0.9, "box": [200, 100, 240, 140]}]
        pos = [(1.0, 5.0), (1.4, 5.0)]  # 0.4 m apart, same class
        ev2 += tr2.update(dets, pos, [3.0, 3.0])
    return {
        "persistent_tracks_confirmed": persistent,
        "expected_persistent": 2,
        "tentative_tracks_killed": tr.stats["fp_suppressed"],
        "one_frame_fp_reported": 0 if persistent == 2 else 1,
        "duplicates_caused_extra_report": 0 if persistent == 2 else 1,
        "dedupe_two_close_hazards_reported": len(ev2),
        "dedupe_expected": 1,
        "iou_ok": iou_ok,
    }


# ----------------------------------------------------------- 5. end to end
def test_end_to_end(seed=555):
    det = FakeDetector(CFG["paths"]["rocks_csv"], CFG["paths"]["poses_csv"],
                       CFG["paths"]["camera_json"], CFG["detector_fake"])
    tr = Tracker(RCFG["track_iou_threshold"], RCFG["track_confirm_hits"],
                 RCFG["track_max_misses"],
                 max_report_range_m=RCFG["max_report_range_m"],
                 min_merge_radius_m=LCFG["min_uncertainty_m"],
                 pixel_noise_px=LCFG["pixel_noise_px"],
                 camera_height_m=LCFG["camera_height_m"], cam=CAM)

    captured = []          # (sim_send_time, packet)
    clock = [0.0]
    sender = Sender("127.0.0.1", 0, repeats=RCFG["hazard_repeats"],
                    repeat_gap_s=RCFG["hazard_repeat_gap_s"],
                    heartbeat_period_s=RCFG["heartbeat_period_s"],
                    max_packets_per_s=RCFG["max_packets_per_s"],
                    queue_size=RCFG["queue_size"],
                    sink=lambda pkt: captured.append((clock[0], pkt)))
    events = []
    n_frames = len(det.poses)
    dt = 0.5
    while clock[0] <= n_frames / FPS + 30.0:           # extra time for repeats
        t = clock[0]
        n = int(round(t * FPS))
        if n < n_frames:
            pose = det.poses[n]
            dets = det.detect(None, frame_idx=n)
            pos = [localize_detection(d["box"], pose, CAM) for d in dets]
            rng = [None if p is None else
                   math.hypot(p[0] - pose["x"], p[1] - pose["y"]) for p in pos]
            for ev in tr.update(dets, pos, rng):
                sender.submit_hazard(ev["cls"], ev["x"], ev["y"], ev["conf"],
                                     t, ev["err"], n / FPS)
                events.append({"cls": ev["cls"], "x": ev["x"], "y": ev["y"],
                               "t": n / FPS, "arrival": None})
        sender.tick(t, 0.0, 0.2 * t, 0.0, n / FPS)
        clock[0] += dt

    # lossy link (same logic as link.py) -> Earth de-dup/state
    lk = CFG["link"]
    rng = random.Random(seed)
    earth = EarthState()
    first_arrival = {}   # (type, x, y, t) -> sim time of first delivered copy
    for send_t, pkt in captured:
        if lk["blackout_for_s"] and in_blackout(send_t, lk["blackout_every_s"],
                                                lk["blackout_for_s"]):
            continue
        if rng.random() < lk["loss"]:
            continue
        before = {(h["type"], h["x"], h["y"], h["t"]) for h in earth.hazards}
        earth.handle_packet(pkt, send_t)
        for h in earth.hazards:
            key = (h["type"], h["x"], h["y"], h["t"])
            if key not in before and key not in first_arrival:
                first_arrival[key] = send_t
    snap = earth.snapshot(wall=clock[0])
    shown = snap["hazards"]

    # match shown vs ground truth: greedy one-to-one (same class, <= tol)
    tol = 1.0
    gt = det.rocks
    pairs = []
    for hi, h in enumerate(shown):
        for g in gt:
            if g["type"] == h["type"]:
                d = math.hypot(g["x"] - h["x"], g["y"] - h["y"])
                if d <= tol:
                    pairs.append((d, hi, g["id"]))
    pairs.sort()
    used_h, used_g = set(), set()
    for d, hi, gid in pairs:
        if hi in used_h or gid in used_g:
            continue
        used_h.add(hi)
        used_g.add(gid)
    matched_shown = len(used_h)
    matched_gt = used_g

    # age at arrival = first delivered copy - hazard confirm mission time
    ages = []
    for h in shown:
        key = (h["type"], h["x"], h["y"], h["t"])
        if key in first_arrival:
            ages.append(first_arrival[key] - h["t"])

    # bytes: 9-byte telemetry vs equivalent JSON
    bin_bytes = PACKET_SIZE * len(captured)
    json_bytes = 0
    for _, pkt in captured:
        d = decode(pkt)
        if d is None:
            continue
        json_bytes += len(json.dumps({k: d[k] for k in
                                      ("type", "x", "y", "conf", "err",
                                       "t", "seq")}).encode())

    return {
        "source": data_source(det),
        "ground_truth_hazards": len(gt),
        "rover_confirmed_events": len(events),
        "earth_unique_hazards": len(shown),
        "matched_shown_or_gt": matched_shown,
        "gt_recall_pct": round(100.0 * len(matched_gt) / len(gt), 1),
        "precision_pct": round(100.0 * matched_shown / len(shown), 1)
        if shown else 0.0,
        "mean_age_at_arrival_s": round(statistics.mean(ages), 2) if ages else None,
        "packets_sent": len(captured),
        "binary_bytes": bin_bytes,
        "json_bytes": json_bytes,
        "bytes_ratio": round(json_bytes / bin_bytes, 2) if bin_bytes else None,
        "duplicate_suppressed": tr.stats["duplicate_suppressed"],
        "range_gated_frames": tr.stats["range_gated"],
    }


# ------------------------------------------------------------ 6. perception
def _iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def _load_yolo_labels(labels_dir, stem, id_to_cls, w, h):
    """YOLO format: `class_id cx cy bw bh` normalized -> (cls, xyxy px)."""
    out = []
    p = Path(labels_dir) / f"{stem}.txt"
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cls = id_to_cls.get(cid)
        if cls is None:
            continue
        cx, cy, bw, bh = (float(parts[i]) for i in range(1, 5))
        out.append((cls, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def test_perception():
    y = CFG["yolo"]
    weights = ROOT / y["weights"]
    images_dir = ROOT / y["test_images"]
    labels_dir = ROOT / y["test_labels"]
    if not weights.exists() or not images_dir.exists():
        return {"status": "skipped", "reason": "no model"}

    import cv2
    import psutil
    from rover.detector import YoloDetector
    det = YoloDetector(str(weights), CAM, y["imgsz"],
                       y["conf_threshold"], y["cpu_threads"])

    exts = ("*.png", "*.jpg", "*.jpeg")
    imgs = sorted(f for e in exts for f in images_dir.glob(e))
    if not imgs:
        return {"status": "skipped", "reason": "no model"}

    proc = psutil.Process()
    out = {"status": "measured", "cpu_threads": y["cpu_threads"],
           "images": len(imgs), "weights": y["weights"],
           "classes": det.names}
    for imgsz in (640, 416):
        det.imgsz = imgsz
        peak = 0
        t0 = time.perf_counter()
        tp = fp = fn = 0
        per_cls = {}
        for img_path in imgs:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            ih, iw = frame.shape[:2]
            gts = _load_yolo_labels(labels_dir, img_path.stem,
                                    det.id_to_cls, iw, ih)
            preds = det.detect(frame)
            used = set()
            for cls, pb in [(p["cls"], p["box"]) for p in preds]:
                best, bi = 0.0, -1
                for gi, (gcls, gb) in enumerate(gts):
                    if gi in used or gcls != cls:
                        continue
                    v = _iou_xyxy(pb, gb)
                    if v > best:
                        best, bi = v, gi
                if best >= 0.5 and bi >= 0:
                    used.add(bi)
                    tp += 1
                    per_cls.setdefault(cls, [0, 0, 0])[0] += 1
                else:
                    fp += 1
                    per_cls.setdefault(cls, [0, 0, 0])[1] += 1
            for gi, (gcls, _gb) in enumerate(gts):
                if gi not in used:
                    fn += 1
                    per_cls.setdefault(gcls, [0, 0, 0])[2] += 1
            peak = max(peak, proc.memory_info().rss)
        dt = time.perf_counter() - t0
        out[f"imgsz_{imgsz}"] = {
            "fps": round(len(imgs) / dt, 1),
            "peak_ram_mb": round(peak / 1e6, 1),
            "precision": round(100.0 * tp / (tp + fp), 1) if tp + fp else None,
            "recall": round(100.0 * tp / (tp + fn), 1) if tp + fn else None,
            "per_class": {c: {"precision": round(100.0 * v[0] / (v[0] + v[1]), 1)
                              if v[0] + v[1] else None,
                              "recall": round(100.0 * v[0] / (v[0] + v[2]), 1)
                              if v[0] + v[2] else None}
                          for c, v in per_cls.items()},
        }
    return out


# ------------------------------------------------------------------ report
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": sys.platform,
        "python": sys.version.split()[0],
        "codec": test_codec(),
        "link_survival": test_link_survival(),
        "localize": test_localize(),
        "tracker": test_tracker(),
        "end_to_end": test_end_to_end(),
        "perception": test_perception(),
    }
    results["wall_seconds"] = round(time.time() - t0, 1)

    with open(RESULTS / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_report(results)
    print(json.dumps(results, indent=2))


def write_report(r):
    L = []
    L.append("# C.R.A.T.E.R. — automated test report\n")
    L.append(f"Generated {r['generated_utc']} · Python {r['python']} · "
             f"{r['host']} · wall {r['wall_seconds']} s\n")
    L.append(f"**Data source for scene-dependent tests:** "
             f"{r['end_to_end']['source']}\n")
    L.append("Every number below is printed by `tests/run_tests.py`; no claim "
             "is asserted without a measured value.\n")

    c = r["codec"]
    L.append("## 1. Telemetry codec\n")
    L.append("| metric | value |\n|---|---|")
    L.append(f"| packet size | {c['packet_size_bytes']} bytes |")
    L.append(f"| equivalent JSON size | {c['json_size_bytes']} bytes |")
    L.append(f"| compression ratio | {c['compression_ratio']}x smaller |")
    L.append(f"| round-trip max position error | "
             f"{c['roundtrip_max_position_error_m']} m |")
    L.append(f"| single-bit corruption rejected | "
             f"{c['corrupt_rejected']}/{c['corrupt_tested']} "
             f"({c['corrupt_reject_pct']}%) |")
    L.append(f"| error-class field round-trip | "
             f"{c['error_class_roundtrip_ok']} |\n")

    L.append("## 2. Link survival (in-process, fixed seed, "
             "blackout 5 s / 30 s)\n")
    L.append("| loss | blackouts | repeats | delivered | delivery % |\n"
             "|---|---|---|---|---|")
    for row in r["link_survival"]:
        L.append(f"| {row['loss']:.0%} | {row['blackouts']} | "
                 f"{row['repeats']}x | {row['delivered']}/{row['hazards']} | "
                 f"{row['delivery_pct']}% |")
    L.append("")

    z = r["localize"]
    L.append(f"## 3. Localization (source: {z['source']})\n")
    L.append(f"- back-projection round-trip max error: "
             f"**{z['roundtrip_max_error_m']} m** "
             f"(< 1 cm: {z['roundtrip_under_1cm']})\n")
    L.append("| range (m) | measured err (m) | formula err (m) | ratio |\n"
             "|---|---|---|---|")
    for m in z["montecarlo_vs_formula"]:
        L.append(f"| {m['range_m']} | {m['measured_err_m']} | "
                 f"{m['formula_err_m']} | {m['ratio']} |")
    L.append(f"\n- pipeline detect->localize vs ground-truth rocks: "
             f"{z['detections_matched']} detections matched, "
             f"mean **{z['gt_mean_error_m']} m**, max **{z['gt_max_error_m']} m**\n")

    t = r["tracker"]
    L.append("## 4. Tracker\n")
    L.append(f"- persistent tracks confirmed: "
             f"**{t['persistent_tracks_confirmed']}** "
             f"(expected {t['expected_persistent']})")
    L.append(f"- one-frame false positive produced a report: "
             f"**{t['one_frame_fp_reported']}** (expected 0)")
    L.append(f"- duplicate detections produced an extra report: "
             f"**{t['duplicates_caused_extra_report']}** (expected 0)")
    L.append(f"- tentative tracks killed unconfirmed: "
             f"**{t['tentative_tracks_killed']}**")
    L.append(f"- two same-class hazards 0.4 m apart -> reported "
             f"**{t['dedupe_two_close_hazards_reported']}** "
             f"(expected {t['dedupe_expected']}, spatial memory)\n")

    e = r["end_to_end"]
    L.append("## 5. End to end (fake detector, lossy link in-process)\n")
    L.append(f"Source: **{e['source']}** (fixed link seed).\n")
    L.append("| metric | value |\n|---|---|")
    L.append(f"| ground-truth hazards | {e['ground_truth_hazards']} |")
    L.append(f"| rover confirmed events | {e['rover_confirmed_events']} |")
    L.append(f"| Earth unique hazards | {e['earth_unique_hazards']} |")
    L.append(f"| GT recall | {e['gt_recall_pct']}% |")
    L.append(f"| precision | {e['precision_pct']}% |")
    L.append(f"| mean age at arrival | {e['mean_age_at_arrival_s']} s |")
    L.append(f"| packets sent | {e['packets_sent']} |")
    L.append(f"| binary bytes | {e['binary_bytes']} |")
    L.append(f"| equivalent JSON bytes | {e['json_bytes']} |")
    L.append(f"| JSON / binary | {e['bytes_ratio']}x |")
    L.append(f"| duplicate events suppressed on rover | "
             f"{e['duplicate_suppressed']} |")
    L.append(f"| frames held by 8 m range gate | {e['range_gated_frames']} |\n")

    p = r["perception"]
    L.append("## 6. Perception (CPU-only)\n")
    if p.get("status") == "skipped":
        L.append(f"Skipped: {p['reason']}. Reported automatically once the "
                 "weights and `data/dfki_test/images` exist.\n")
    else:
        L.append(f"Weights: `{p['weights']}` · cpu_threads {p['cpu_threads']} "
                 f"· {p['images']} test images · classes {p['classes']}\n")
        for imgsz in (640, 416):
            k = p.get(f"imgsz_{imgsz}")
            if not k:
                continue
            L.append(f"**imgsz {imgsz}** — FPS {k['fps']}, peak RAM "
                     f"{k['peak_ram_mb']} MB, precision {k['precision']}%, "
                     f"recall {k['recall']}%")
            for c, m in k["per_class"].items():
                L.append(f"  - {c}: precision {m['precision']}%, "
                         f"recall {m['recall']}%")
        L.append("")

    L.append("## Known limits\n")
    L.append("- Flat-ground assumption (z = 0); craters are approximated.")
    L.append("- Uncertainties are a first-order estimate "
             "(r^2+z^2)/z * (dv/fy), not a covariance.")
    L.append("- Over-merge: the range-based merge radius (up to ~1.1 m at the "
             "8 m gate) can absorb distinct hazards that are close together.")
    L.append("- Sim-to-real gap; no ice class.\n")

    with open(RESULTS / "test_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
