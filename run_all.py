"""One command: link + earth + rover as separate processes, then dashboard.

  python run_all.py [--duration 45] [--no-browser]

- Restarts a crashed child (logged, bounded retries).
- Stops everything cleanly on Ctrl+C or when --duration elapses.
- Opens the dashboard in the browser.
"""
import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent
MAX_RESTARTS = 5


def build_children(cfg, duration):
    link = cfg["link"]
    children = {
        "link": [sys.executable, str(ROOT / "link" / "link.py"),
                 "--listen", str(link["listen"]),
                 "--forward", str(link["forward"]),
                 "--loss", str(link["loss"]),
                 "--blackout-every", str(link["blackout_every_s"]),
                 "--blackout-for", str(link["blackout_for_s"]),
                 "--seed", str(link.get("seed", 42))],
        "earth": [sys.executable, str(ROOT / "earth" / "earth.py")],
        "rover": [sys.executable, str(ROOT / "rover" / "rover.py")],
    }
    if duration:
        children["rover"] += ["--duration", str(duration)]
    return children


def main():
    ap = argparse.ArgumentParser(description="C.R.A.T.E.R. full pipeline")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after N seconds")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    procs = {}
    restarts = {k: 0 for k in ("link", "earth", "rover")}
    stop_at = time.monotonic() + a.duration if a.duration else None
    stopping = False

    def start(name, argv):
        print(f"[run_all] starting {name}: {' '.join(argv[1:])}")
        procs[name] = subprocess.Popen(argv, cwd=str(ROOT))

    try:
        for name, argv in build_children(cfg, a.duration).items():
            start(name, argv)
        if not a.no_browser:
            time.sleep(1.5)
            webbrowser.open(
                f"http://{cfg['earth']['http_host']}:{cfg['earth']['http_port']}/")

        while not stopping:
            if stop_at and time.monotonic() >= stop_at:
                print("[run_all] duration reached, stopping")
                break
            for name, p in list(procs.items()):
                rc = p.poll()
                if rc is None:
                    continue
                if stopping:
                    continue
                print(f"[run_all] {name} exited rc={rc}")
                if restarts[name] < MAX_RESTARTS:
                    restarts[name] += 1
                    print(f"[run_all] restarting {name} "
                          f"({restarts[name]}/{MAX_RESTARTS})")
                    time.sleep(1.0)
                    start(name, build_children(cfg, a.duration)[name])
                else:
                    print(f"[run_all] {name} crashed {MAX_RESTARTS} times, "
                          "not restarting")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[run_all] Ctrl+C - stopping all children")
    finally:
        stopping = True
        # Give children a moment to stop on their own (rover logs final stats).
        grace = time.monotonic() + 4.0
        while time.monotonic() < grace and any(
                p.poll() is None for p in procs.values()):
            time.sleep(0.2)
        for name, p in procs.items():
            if p.poll() is None:
                p.terminate()
        t0 = time.monotonic()
        for name, p in procs.items():
            try:
                p.wait(timeout=max(0.1, 5 - (time.monotonic() - t0)))
            except subprocess.TimeoutExpired:
                print(f"[run_all] {name} did not stop, killing")
                p.kill()
        print("[run_all] all children stopped")


if __name__ == "__main__":
    main()
