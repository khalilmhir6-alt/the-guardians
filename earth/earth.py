"""Earth side: UDP receiver + tiny HTTP server for the dashboard.

Standard library only. De-duplicates the rover's repeated hazard packets,
counts corruption and estimated losses, logs every accepted packet to
logs/earth_log.csv, and serves the dashboard on :8080.

  /        -> static/index.html
  /state   -> JSON state (hazards, heartbeat, link stats)

Stop: Ctrl+C (prints final statistics).
"""
import argparse
import csv
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.codec import decode          # noqa: E402
from common.config import load_config    # noqa: E402
from rover.localize import steps_to_heading  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"


class EarthState:
    """Thread-safe packet state. Importable by tests without sockets."""

    def __init__(self, max_hazards=500, uncertainty_radius_m=0.5,
                 stale_amber_s=3.0, stale_red_s=10.0):
        self.lock = threading.Lock()
        self.hazards = []          # newest last, bounded
        self.max_hazards = max_hazards
        self.uncertainty_radius_m = uncertainty_radius_m
        self.stale_amber_s = stale_amber_s
        self.stale_red_s = stale_red_s
        self.heartbeat = None
        self.seen = set()          # de-dup key = (type, x, y, rover_time)
        self.stats = {"received": 0, "accepted": 0, "corrupted": 0,
                      "duplicates": 0, "hazards": 0}
        self.last_seq = None
        self.lost_est = 0
        self.last_packet_wall = None

    def handle_packet(self, data: bytes, wall: float):
        """Returns the decoded dict if newly accepted, else None."""
        pkt = decode(data)
        with self.lock:
            self.stats["received"] += 1
            self.last_packet_wall = wall
            if pkt is None:
                self.stats["corrupted"] += 1
                return None
            if self.last_seq is not None:
                gap = (pkt["seq"] - (self.last_seq + 1)) % 256
                if gap <= 64:          # assume loss, not a restart
                    self.lost_est += gap
            self.last_seq = pkt["seq"]

            if pkt["type"] == "heartbeat":
                self.heartbeat = {"x": pkt["x"], "y": pkt["y"],
                                  "heading_deg": steps_to_heading(
                                      round(pkt["conf"] * 15)),
                                  "conf": pkt["conf"], "t": pkt["t"],
                                  "seq": pkt["seq"], "arrival": wall}
                self.stats["accepted"] += 1
                return pkt

            key = (pkt["type"], pkt["x"], pkt["y"], pkt["t"])
            if key in self.seen:
                self.stats["duplicates"] += 1
                return None
            self.seen.add(key)
            self.stats["accepted"] += 1
            self.stats["hazards"] += 1
            self.hazards.append({"type": pkt["type"], "x": pkt["x"],
                                 "y": pkt["y"], "conf": pkt["conf"],
                                 "t": pkt["t"], "seq": pkt["seq"],
                                 "arrival": wall})
            if len(self.hazards) > self.max_hazards:
                self.hazards.pop(0)
            return pkt

    def snapshot(self, wall=None):
        wall = wall if wall is not None else time.time()
        with self.lock:
            hazards = []
            for h in self.hazards:
                hazards.append({**{k: h[k] for k in
                                   ("type", "x", "y", "conf", "t", "seq")},
                                "age_s": round(wall - h["arrival"], 3)})
            hb = None
            if self.heartbeat:
                hb = {**{k: self.heartbeat[k] for k in
                         ("x", "y", "heading_deg", "conf", "t", "seq")},
                      "age_s": round(wall - self.heartbeat["arrival"], 3)}
            received = self.stats["received"]
            lost = self.lost_est + self.stats["corrupted"]
            denom = received + lost
            return {
                "server_time": wall,
                "heartbeat": hb,
                "hazards": hazards,
                "view": {"uncertainty_radius_m": self.uncertainty_radius_m,
                         "stale_amber_s": self.stale_amber_s,
                         "stale_red_s": self.stale_red_s},
                "stats": {
                    **self.stats,
                    "lost_est": self.lost_est,
                    "delivery_pct": round(100.0 * received / denom, 1)
                    if denom else 100.0,
                    "last_packet_age_s": round(
                        wall - self.last_packet_wall, 3)
                    if self.last_packet_wall else None,
                },
            }


def make_http_server(state: EarthState, host: str, port: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):   # quiet
            pass

        def do_GET(self):
            if self.path.split("?")[0] == "/state":
                body = json.dumps(state.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path in ("/", "/index.html"):
                page = (STATIC_DIR / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            else:
                self.send_error(404)

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


def main():
    ap = argparse.ArgumentParser(description="C.R.A.T.E.R. Earth receiver")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ecfg = cfg["earth"]

    state = EarthState(uncertainty_radius_m=ecfg.get("uncertainty_radius_m", 0.5),
                       stale_amber_s=ecfg.get("stale_amber_s", 3.0),
                       stale_red_s=ecfg.get("stale_red_s", 10.0))
    log_path = Path(cfg["paths"]["earth_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_csv = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_csv)
    log_writer.writerow(["arrival_wall", "type", "x", "y", "conf",
                         "rover_t", "seq"])

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind((cfg["link"]["host"], ecfg["listen"]))
    udp.settimeout(0.5)
    http = make_http_server(state, ecfg["http_host"], ecfg["http_port"])
    http_thread = threading.Thread(target=http.serve_forever,
                                   kwargs={"poll_interval": 0.5},
                                   daemon=True)
    http_thread.start()
    print(f"[earth] UDP :{ecfg['listen']}  dashboard http://{ecfg['http_host']}"
          f":{ecfg['http_port']}/")

    running = True
    try:
        while running:
            try:
                data, _ = udp.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            wall = time.time()
            pkt = state.handle_packet(data, wall)
            if pkt is not None:
                log_writer.writerow([f"{wall:.3f}", pkt["type"],
                                     f"{pkt['x']:.1f}", f"{pkt['y']:.1f}",
                                     f"{pkt['conf']:.2f}", pkt["t"],
                                     pkt["seq"]])
                log_csv.flush()
    except KeyboardInterrupt:
        print("\n[earth] stopped.")
    finally:
        http.shutdown()
        http.server_close()
        udp.close()
        log_csv.close()
        snap = state.snapshot()
        print(f"[earth] received={snap['stats']['received']} "
              f"accepted={snap['stats']['accepted']} "
              f"corrupted={snap['stats']['corrupted']} "
              f"duplicates={snap['stats']['duplicates']} "
              f"hazards={snap['stats']['hazards']} "
              f"delivery={snap['stats']['delivery_pct']}%")


if __name__ == "__main__":
    main()
