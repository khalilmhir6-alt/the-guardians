"""Perception: Detector interface + FakeDetector + YoloDetector.

The real ML model plugs in later by dropping models/best.onnx (or .pt)
and setting config.json "detector": "yolo". Nothing else changes.

Detection output contract (both detectors):
    list of {"cls": "rock"|"crater", "conf": float, "box": [x1,y1,x2,y2]}
    box in pixels, image 640x640.
"""
import csv
import logging
import math
import random
from pathlib import Path

from rover.localize import project_point

log = logging.getLogger("crater.detector")


class Detector:
    """Interface: frame in, detections out."""

    def detect(self, frame, frame_idx=None):
        raise NotImplementedError


class FakeDetector(Detector):
    """Projects ground-truth hazards (rocks.csv) into the image with the
    pinhole model, so the full pipeline runs before the model exists.

    Adds +/- pixel noise, occasional misses and false positives.
    If data/ has not been generated yet, falls back to a seeded synthetic
    scene (straight path, random rocks) with the same contracts.
    """

    def __init__(self, rocks_csv: str, poses_csv: str, camera_json: str,
                 cfg: dict = None):
        cfg = cfg or {}
        self.base_conf = cfg.get("base_conf", 0.85)
        self.noise_px = cfg.get("pixel_noise_px", 3.0)
        self.miss_prob = cfg.get("miss_prob", 0.05)
        self.fp_prob = cfg.get("false_positive_prob", 0.05)
        self.fp_conf = cfg.get("false_positive_conf", 0.45)
        self.rng = random.Random(cfg.get("seed", 7))

        self.camera = self._load_camera(camera_json)
        self.rocks = self._load_rocks(rocks_csv)
        self.poses = self._load_poses(poses_csv)
        self.synthetic = not (Path(rocks_csv).exists() and Path(poses_csv).exists())
        if self.synthetic:
            log.warning("data/ not generated yet - FakeDetector using a synthetic scene")
            self.rocks, self.poses = self._synthetic_scene()
        log.info("FakeDetector ready: %d hazards, %d frames, synthetic=%s",
                 len(self.rocks), len(self.poses), self.synthetic)

    # ---------- loading ----------

    @staticmethod
    def _load_camera(path):
        import json
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"focal_mm": 24.0, "sensor_width_mm": 36.0,
                "width_px": 640, "height_px": 640}

    @staticmethod
    def _load_rocks(path):
        rocks = []
        p = Path(path)
        if not p.exists():
            return rocks
        with open(p, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rocks.append({"id": int(row["id"]), "type": row["type"],
                              "x": float(row["x"]), "y": float(row["y"]),
                              "size_m": float(row["size_m"])})
        return rocks

    @staticmethod
    def _load_poses(path):
        poses = []
        p = Path(path)
        if not p.exists():
            return poses
        with open(p, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                poses.append({"frame": int(row["frame"]),
                              "x": float(row["x"]), "y": float(row["y"]),
                              "z": float(row["z"]),
                              "yaw_deg": float(row["yaw_deg"]),
                              "pitch_deg": float(row["pitch_deg"])})
        return poses

    @staticmethod
    def _synthetic_scene():
        """Same shape as the Blender scene: straight 0.2 m/s path, 60 s,
        2 fps, camera 0.4 m high pitched 15 deg down, ~20 seeded rocks."""
        rng = random.Random(1234)
        rocks = []
        for i in range(20):
            rocks.append({"id": i + 1, "type": "rock",
                          "x": rng.uniform(-4.0, 4.0),
                          "y": rng.uniform(2.0, 14.0),
                          "size_m": rng.uniform(0.15, 0.5)})
        for i in range(3):
            rocks.append({"id": 21 + i, "type": "crater",
                          "x": rng.uniform(-3.0, 3.0),
                          "y": rng.uniform(3.0, 12.0),
                          "size_m": rng.uniform(0.6, 1.2)})
        poses = []
        for n in range(120):
            t = n / 2.0
            poses.append({"frame": n + 1, "x": 0.0, "y": 0.2 * t, "z": 0.4,
                          "yaw_deg": 0.0, "pitch_deg": 15.0})
        return rocks, poses

    # ---------- detection ----------

    def detect(self, frame, frame_idx=None):
        if frame_idx is None:
            frame_idx = getattr(self, "_counter", 0)
            self._counter = frame_idx + 1
        if not self.poses:
            return []
        pose = self.poses[min(frame_idx, len(self.poses) - 1)]
        cam = self.camera
        w, h = cam["width_px"], cam["height_px"]
        fx = cam["focal_mm"] / cam["sensor_width_mm"] * cam["width_px"]

        out = []
        for rock in self.rocks:
            if self.rng.random() < self.miss_prob:
                continue  # dropout: mimics a missed detection
            uv = project_point((rock["x"], rock["y"], 0.0), pose, cam)
            if uv is None:
                continue
            u, v = uv
            dist = math.sqrt((rock["x"] - pose["x"]) ** 2 +
                             (rock["y"] - pose["y"]) ** 2 + pose["z"] ** 2)
            side = max(6.0, fx * rock["size_m"] / max(dist, 0.1))
            n = self.rng.uniform(-self.noise_px, self.noise_px)
            x1 = u - side / 2 + n
            x2 = u + side / 2 + n
            y1 = v - side + n
            y2 = v + n
            if x2 < 0 or x1 >= w or y2 < 0 or y1 >= h:
                continue
            conf = max(0.0, min(1.0, self.base_conf + self.rng.uniform(-0.05, 0.05)))
            out.append({"cls": rock["type"], "conf": conf,
                        "box": [x1, y1, x2, y2]})

        if self.rng.random() < self.fp_prob:
            fx0 = self.rng.uniform(0, w - 40)
            fy0 = self.rng.uniform(0, h - 40)
            s = self.rng.uniform(10, 40)
            out.append({"cls": "rock", "conf": self.fp_conf,
                        "box": [fx0, fy0, fx0 + s, fy0 + s]})
        return out


class YoloDetector(Detector):
    """YOLO on CPU only, 2 threads. Loads models/best.onnx (preferred, via
    onnxruntime) or models/best.pt (via ultralytics). Same output format
    as FakeDetector. Class 0 = rock, class 1 = crater.
    """

    CLASS_NAMES = {0: "rock", 1: "crater"}

    def __init__(self, model_path: str, cam: dict, conf_threshold: float = 0.25,
                 threads: int = 2):
        self.conf_threshold = conf_threshold
        self.camera = cam
        self.size = (cam["width_px"], cam["height_px"])  # (w, h)
        self.model_path = model_path
        self.kind = None
        self.session = None
        self.model = None

        p = Path(model_path)
        if p.suffix == ".onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(p), sess_options=so,
                providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.kind = "onnx"
        elif p.suffix == ".pt":
            from ultralytics import YOLO
            import torch
            torch.set_num_threads(threads)
            self.model = YOLO(str(p))
            self.model.to("cpu")
            self.kind = "pt"
        else:
            raise FileNotFoundError(
                f"no model at {model_path} (need .onnx or .pt); "
                'set config "detector": "fake" until the model arrives')
        log.info("YoloDetector loaded %s (%s, CPU, %d threads)",
                 model_path, self.kind, threads)

    def detect(self, frame, frame_idx=None):
        if self.kind == "pt":
            return self._detect_pt(frame)
        return self._detect_onnx(frame)

    def _detect_pt(self, frame):
        import numpy as np
        results = self.model.predict(frame, imgsz=self.size, conf=self.conf_threshold,
                                     device="cpu", verbose=False)
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf, c in zip(xyxy, confs, clss):
                out.append({"cls": self.CLASS_NAMES.get(int(c), "rock"),
                            "conf": float(conf),
                            "box": [float(box[0]), float(box[1]),
                                    float(box[2]), float(box[3])]})
        return out

    def _detect_onnx(self, frame):
        """YOLOv8-style single output [1, 4+nc, N] after letterbox resize."""
        import numpy as np
        import cv2
        w, h = self.size
        img = frame
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        pred = self.session.run(None, {self.input_name: blob})[0][0]  # [4+nc, N]
        num_cls = pred.shape[0] - 4
        boxes_xywh = pred[:4].T          # cx, cy, w, h in pixels
        scores_all = pred[4:].T          # [N, nc]
        out = []
        for (cx, cy, bw, bh), scores in zip(boxes_xywh, scores_all):
            c = int(scores.argmax())
            conf = float(scores[c])
            if conf < self.conf_threshold:
                continue
            out.append({"cls": self.CLASS_NAMES.get(c, "rock"), "conf": conf,
                        "box": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]})
        return out


def make_detector(cfg: dict) -> Detector:
    """Build the detector selected by config.json 'detector'."""
    paths = cfg["paths"]
    kind = cfg.get("detector", "fake")
    if kind == "fake":
        return FakeDetector(paths["rocks_csv"], paths["poses_csv"],
                            paths["camera_json"], cfg.get("detector_fake", {}))
    if kind == "yolo":
        return YoloDetector(cfg["yolo"]["model_path"], cfg["camera"],
                            cfg["yolo"].get("conf_threshold", 0.25),
                            cfg["yolo"].get("threads", 2))
    raise ValueError(f'unknown detector "{kind}" (use "fake" or "yolo")')
