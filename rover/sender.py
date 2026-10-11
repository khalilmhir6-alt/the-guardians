"""Telemetry sender: how hazards survive the lossy link.

- Each new confirmed hazard is sent N times (default 3), copies spaced
  6 s apart (gap > blackout length so one blackout can't eat all copies).
- Heartbeat every 2 s: rover position, heading in the confidence field
  (22.5-degree steps, team choice).
- Bandwidth budget: max max_packets_per_s packets per sliding second.
  Bounded queue; when full, heartbeats are dropped first, never new
  hazards. Sequence number increments on every packet actually sent,
  including repeats.
"""
import collections
import logging
import socket

from common.codec import encode
from rover.localize import heading_to_steps

log = logging.getLogger("crater.sender")


class Sender:
    def __init__(self, host, port, repeats=3, repeat_gap_s=6.0,
                 heartbeat_period_s=2.0, max_packets_per_s=5,
                 queue_size=64):
        self.addr = (host, port)
        self.repeats = repeats
        self.repeat_gap_s = repeat_gap_s
        self.heartbeat_period_s = heartbeat_period_s
        self.max_packets_per_s = max_packets_per_s
        self.queue = collections.deque(maxlen=queue_size)
        self.queue_size = queue_size
        self.seq = 0
        self.sent_times = collections.deque()  # sliding 1 s window
        self.last_heartbeat = None
        self.stats = {"sent": 0, "hazards_sent": 0, "heartbeats_sent": 0,
                      "heartbeats_dropped": 0, "hazards_dropped": 0}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)

    # ---------- public API ----------

    def submit_hazard(self, kind, x, y, conf, t_now):
        """Schedule `repeats` copies of a new hazard, spaced repeat_gap_s."""
        for i in range(self.repeats):
            self._enqueue((t_now + i * self.repeat_gap_s, kind, x, y, conf))
        log.info("hazard %s (%.2f, %.2f) conf=%.2f -> %d copies",
                 kind, x, y, conf, self.repeats)

    def tick(self, t_now, rover_x, rover_y, heading_deg, t_mission):
        """Send everything due, within the bandwidth budget."""
        self._prune_window(t_now)
        sent = []
        # 1) due hazard copies first (never dropped for bandwidth)
        remaining = self._budget()
        while remaining > 0 and self.queue and self.queue[0][0] <= t_now:
            due = self.queue.popleft()
            self._send(due[1], due[2], due[3], due[4], t_mission, t_now)
            sent.append(due)
            remaining -= 1
        # 2) heartbeat last: dropped when the budget is exhausted
        if self.last_heartbeat is None:
            self.last_heartbeat = t_now
        if t_now - self.last_heartbeat >= self.heartbeat_period_s:
            if remaining > 0:
                hconf = heading_to_steps(heading_deg) / 15.0
                self._send("heartbeat", rover_x, rover_y, hconf, t_mission, t_now)
            else:
                self.stats["heartbeats_dropped"] += 1
            self.last_heartbeat = t_now  # keep cadence even when skipped
        return sent

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    # ---------- internals ----------

    def _enqueue(self, item):
        if len(self.queue) >= self.queue_size:
            # Queue full: make room by dropping the OLDEST hazard copy,
            # log loudly (flight software: never silent).
            self.queue.popleft()
            self.stats["hazards_dropped"] += 1
            log.error("sender queue full - dropped oldest hazard copy")
        self.queue.append(item)

    def _budget(self):
        return max(0, self.max_packets_per_s - len(self.sent_times))

    def _prune_window(self, t_now):
        while self.sent_times and t_now - self.sent_times[0] >= 1.0:
            self.sent_times.popleft()

    def _send(self, kind, x, y, conf, t_mission, t_now):
        pkt = encode(kind, x, y, conf, t_mission, self.seq)
        try:
            self.sock.sendto(pkt, self.addr)
        except OSError as e:
            log.error("send failed: %s", e)
            return
        self.seq = (self.seq + 1) % 256
        self.sent_times.append(t_now)
        self.stats["sent"] += 1
        if kind == "heartbeat":
            self.stats["heartbeats_sent"] += 1
        else:
            self.stats["hazards_sent"] += 1
