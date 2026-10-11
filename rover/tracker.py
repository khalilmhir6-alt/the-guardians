"""Tracking: keep stable IDs across frames, kill transient false positives.

- Greedy IoU matching (threshold 0.3), class-aware, one-to-one.
- A track is CONFIRMED after `confirm_hits` consecutive hits
  (removes one-frame false positives).
- Dropped after `max_misses` consecutive misses.
- Each confirmed track is reported ONCE as a new-hazard event, with the
  average of its localized positions and average confidence.
"""
import logging

from rover.localize import error_class, estimated_error

log = logging.getLogger("crater.tracker")


def iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class _Track:
    __slots__ = ("tid", "cls", "box", "hits", "misses", "xs", "ys",
                 "ranges", "confs", "reported")

    def __init__(self, tid, cls, box, conf):
        self.tid = tid
        self.cls = cls
        self.box = box
        self.hits = 1
        self.misses = 0
        self.xs = []
        self.ys = []
        self.ranges = []
        self.confs = [conf]
        self.reported = False

    def add_hit(self, box, conf, xy, rng=None):
        self.box = box
        self.hits += 1
        self.misses = 0
        self.confs.append(conf)
        if xy is not None:
            self.xs.append(xy[0])
            self.ys.append(xy[1])
        if rng is not None:
            self.ranges.append(rng)

    def mean_pos(self):
        if not self.xs:
            return None
        return sum(self.xs) / len(self.xs), sum(self.ys) / len(self.ys)

    def mean_range(self):
        return sum(self.ranges) / len(self.ranges) if self.ranges else None


class Tracker:
    def __init__(self, iou_threshold=0.3, confirm_hits=3, max_misses=5,
                 max_report_range_m=8.0, min_merge_radius_m=0.5,
                 pixel_noise_px=3.0, camera_height_m=0.4, cam=None):
        self.iou_threshold = iou_threshold
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.max_report_range_m = max_report_range_m
        self.min_merge_radius_m = min_merge_radius_m
        self.pixel_noise_px = pixel_noise_px
        self.camera_height_m = camera_height_m
        self.cam = cam
        self.tracks = []
        self._next_id = 1
        # Spatial memory: (cls, x, y, merge_radius_m) of everything reported.
        self.reported_points = []
        self.stats = {"confirmed": 0, "dropped": 0, "fp_suppressed": 0,
                      "duplicate_suppressed": 0, "range_gated": 0}

    def _error(self, range_m):
        return estimated_error(range_m, self.camera_height_m,
                               self.pixel_noise_px, self.cam)

    def _is_duplicate(self, cls, x, y, radius_m):
        for c2, x2, y2, r2 in self.reported_points:
            if c2 != cls:
                continue
            thr = max(radius_m, r2)
            if (x - x2) ** 2 + (y - y2) ** 2 < thr * thr:
                return True
        return False

    def update(self, detections, positions, ranges=None):
        """One frame. detections: list of det dicts; positions: list of
        (x, y) or None; ranges: list of ground range (m) from the camera,
        aligned with detections.

        Returns list of new confirmed hazard events:
            {"tid", "cls", "conf", "x", "y", "err"}
        """
        if ranges is None:
            ranges = [None] * len(detections)

        # Greedy best-IoU matching, class-aware, one-to-one.
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                if det["cls"] != tr.cls:
                    continue
                score = iou(tr.box, det["box"])
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            det = detections[di]
            self.tracks[ti].add_hit(det["box"], det["conf"], positions[di],
                                    ranges[di])

        # Unmatched tracks: miss; drop when stale too long.
        kept = []
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
                if tr.misses >= self.max_misses:
                    if not tr.reported:
                        self.stats["fp_suppressed"] += 1
                    self.stats["dropped"] += 1
                    continue
            kept.append(tr)
        self.tracks = kept

        # Unmatched detections: new tentative tracks.
        for di, det in enumerate(detections):
            if di in used_d:
                continue
            tr = _Track(self._next_id, det["cls"], det["box"], det["conf"])
            self._next_id += 1
            if positions[di] is not None:
                tr.xs.append(positions[di][0])
                tr.ys.append(positions[di][1])
            if ranges[di] is not None:
                tr.ranges.append(ranges[di])
            self.tracks.append(tr)

        # Confirm and report each track once (spatial memory + range gate).
        events = []
        for tr in self.tracks:
            if tr.reported or tr.hits < self.confirm_hits:
                continue
            pos = tr.mean_pos()
            if pos is None:
                continue  # never localized -> wait for a good frame
            rng = tr.mean_range()
            if rng is None:
                continue
            if self.max_report_range_m and rng > self.max_report_range_m:
                # Too far to trust: report later, when the rover is closer.
                self.stats["range_gated"] += 1
                continue
            err = self._error(rng)
            radius = max(self.min_merge_radius_m, err)
            if self._is_duplicate(tr.cls, pos[0], pos[1], radius):
                tr.reported = True
                self.stats["duplicate_suppressed"] += 1
                continue
            tr.reported = True
            self.reported_points.append((tr.cls, pos[0], pos[1], radius))
            self.stats["confirmed"] += 1
            events.append({"tid": tr.tid, "cls": tr.cls,
                           "conf": sum(tr.confs) / len(tr.confs),
                           "x": pos[0], "y": pos[1],
                           "err": error_class(err)})
            log.info("track %d confirmed as %s at (%.2f, %.2f) r=%.1fm "
                     "err=%.2fm class=%d", tr.tid, tr.cls, pos[0], pos[1],
                     rng, err, events[-1]["err"])
        return events
