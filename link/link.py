"""
C.R.A.T.E.R. lossy-link emulator.
Sits between the rover and Earth, like the weak radio inside a crater.
Pure Python standard library: no install needed.

  rover --UDP--> [ link.py : port 9000 ] --UDP--> earth (port 9001)

Two kinds of loss, both configurable:
  - random loss  : each packet is dropped with probability --loss
  - blackouts    : every --blackout-every seconds, the link is dead for
                   --blackout-for seconds (rover behind a crater rim)

Run:  python link.py --loss 0.1 --blackout-every 30 --blackout-for 5
Stop: Ctrl+C (prints final statistics)
"""
import argparse
import random
import select
import socket
import time

HOST = "127.0.0.1"


def in_blackout(t, every, duration):
    """True during the last `duration` seconds of every `every`-second cycle."""
    return every > 0 and duration > 0 and (t % every) >= (every - duration)


def main():
    p = argparse.ArgumentParser(description="Lossy lunar link emulator")
    p.add_argument("--listen", type=int, default=9000, help="port the rover sends to")
    p.add_argument("--forward", type=int, default=9001, help="port Earth listens on")
    p.add_argument("--loss", type=float, default=0.10, help="random loss probability 0..1")
    p.add_argument("--blackout-every", type=float, default=30, help="seconds between blackouts (0 = none)")
    p.add_argument("--blackout-for", type=float, default=5, help="blackout duration, seconds")
    p.add_argument("--seed", type=int, default=42, help="random seed (repeatable tests)")
    p.add_argument("--report", type=float, default=5, help="seconds between status lines")
    a = p.parse_args()

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind((HOST, a.listen))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rng = random.Random(a.seed)

    stats = {"in": 0, "out": 0, "lost_random": 0, "lost_blackout": 0}
    start = time.monotonic()
    last_report = start
    was_dark = False

    print(f"[link] listening :{a.listen} -> forwarding :{a.forward} | "
          f"loss={a.loss:.0%} blackout {a.blackout_for}s every {a.blackout_every}s")

    try:
        while True:
            # Wait at most 0.5 s for a packet, so status lines keep coming even if the rover is silent.
            ready, _, _ = select.select([rx], [], [], 0.5)
            t = time.monotonic() - start
            dark = in_blackout(t, a.blackout_every, a.blackout_for)

            if dark != was_dark:
                print(f"[link] t={t:6.1f}s  {'*** BLACKOUT START ***' if dark else '--- link restored ---'}")
                was_dark = dark

            if ready:
                data, _ = rx.recvfrom(1024)
                stats["in"] += 1
                if dark:
                    stats["lost_blackout"] += 1
                elif rng.random() < a.loss:
                    stats["lost_random"] += 1
                else:
                    tx.sendto(data, (HOST, a.forward))
                    stats["out"] += 1

            now = time.monotonic()
            if now - last_report >= a.report:
                print_stats(t, stats)
                last_report = now
    except KeyboardInterrupt:
        print("\n[link] stopped.")
        print_stats(time.monotonic() - start, stats)
    finally:
        rx.close()
        tx.close()


def print_stats(t, s):
    rate = s["out"] / s["in"] if s["in"] else 0
    print(f"[link] t={t:6.1f}s  in={s['in']}  delivered={s['out']}  "
          f"lost_random={s['lost_random']}  lost_blackout={s['lost_blackout']}  "
          f"delivery={rate:.1%}")


if __name__ == "__main__":
    main()
