"""Tracking: keep stable IDs across frames, kill transient false positives.

- Greedy IoU matching (threshold 0.3), class-aware, one-to-one.
- A track is CONFIRMED after `confirm_hits` consecutive hits
  (removes one-frame false positives).
- Dropped after `max_misses` consecutive misses.
- Each confirmed track is reported ONCE as a new-hazard event, with the
  average of its localized positions and average confidence.
"""
import logging

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
                 "confs", "reported")

    def __init__(self, tid, cls, box, conf):
        self.tid = tid
        self.cls = cls
        self.box = box
        self.hits = 1
        self.misses = 0
        self.xs = []
        self.ys = []
        self.confs = [conf]
        self.reported = False

    def add_hit(self, box, conf, xy):
        self.box = box
        self.hits += 1
        self.misses = 0
        self.confs.append(conf)
        if xy is not None:
            self.xs.append(xy[0])
            self.ys.append(xy[1])

    def mean_pos(self):
        if not self.xs:
            return None
        return sum(self.xs) / len(self.xs), sum(self.ys) / len(self.ys)


class Tracker:
    def __init__(self, iou_threshold=0.3, confirm_hits=3, max_misses=5):
        self.iou_threshold = iou_threshold
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.tracks = []
        self._next_id = 1
        self.stats = {"confirmed": 0, "dropped": 0, "fp_suppressed": 0}

    def update(self, detections, positions):
        """One frame. detections: list of det dicts; positions: list of
        (x, y) or None aligned with detections.

        Returns list of new confirmed hazard events:
            {"tid", "cls", "conf", "x", "y"}
        """
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
            self.tracks[ti].add_hit(det["box"], det["conf"], positions[di])

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
            self.tracks.append(tr)

        # Confirm and report each track once.
        events = []
        for tr in self.tracks:
            if tr.reported or tr.hits < self.confirm_hits:
                continue
            pos = tr.mean_pos()
            if pos is None:
                continue  # never localized -> wait for a good frame
            tr.reported = True
            self.stats["confirmed"] += 1
            events.append({"tid": tr.tid, "cls": tr.cls,
                           "conf": sum(tr.confs) / len(tr.confs),
                           "x": pos[0], "y": pos[1]})
            log.info("track %d confirmed as %s at (%.2f, %.2f) conf=%.2f",
                     tr.tid, tr.cls, pos[0], pos[1], events[-1]["conf"])
        return events
