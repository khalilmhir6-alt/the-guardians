"""
C.R.A.T.E.R. telemetry codec - 9-byte hazard packet.
Pure Python standard library: no install needed.

Layout (big-endian, 9 bytes = 72 bits):
  byte 0     : type (2 bits) | confidence (4 bits) | error class (2 bits)
               error class (team choice): 0 = <0.5 m, 1 = <1 m,
               2 = <2 m, 3 = >=2 m range-based localization uncertainty
  bytes 1-2  : x, signed, decimeters (10 cm steps) from landing site
  bytes 3-4  : y, signed, decimeters
  bytes 5-6  : rover time, seconds since mission start (wraps at 65536)
  byte 7     : sequence number (wraps at 256)
  byte 8     : CRC-8 checksum of bytes 0-7
"""
import json
import struct

TYPES = {"rock": 0, "crater": 1, "heartbeat": 2, "spare": 3}
TYPE_NAMES = {v: k for k, v in TYPES.items()}

BODY = struct.Struct(">BhhHB")   # 8 bytes; checksum appended after
PACKET_SIZE = BODY.size + 1      # 9


def crc8(data: bytes) -> int:
    """CRC-8 (polynomial 0x07). Detects corrupted bytes."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def _clamp_int16(v: int) -> int:
    return max(-32768, min(32767, v))


def encode(kind: str, x_m: float, y_m: float, conf: float, t_s: float,
           seq: int, err_class: int = 0) -> bytes:
    """Hazard -> 9 bytes. err_class fills the 2 spare bits (0..3)."""
    conf_q = max(0, min(15, round(conf * 15)))            # 0..15
    head = (TYPES[kind] << 6) | (conf_q << 2) | (err_class & 0x03)
    x_dm = _clamp_int16(round(x_m * 10))                  # meters -> decimeters
    y_dm = _clamp_int16(round(y_m * 10))
    body = BODY.pack(head, x_dm, y_dm, int(t_s) % 65536, seq % 256)
    return body + bytes([crc8(body)])


def decode(packet: bytes):
    """9 bytes -> dict, or None if wrong size or checksum mismatch."""
    if len(packet) != PACKET_SIZE:
        return None
    body, check = packet[:-1], packet[-1]
    if crc8(body) != check:
        return None
    head, x_dm, y_dm, t_s, seq = BODY.unpack(body)
    return {
        "type": TYPE_NAMES[head >> 6],
        "conf": ((head >> 2) & 0x0F) / 15,
        "err": head & 0x03,
        "x": x_dm / 10,
        "y": y_dm / 10,
        "t": t_s,
        "seq": seq,
    }


if __name__ == "__main__":
    # Self-test: prints the numbers that go into the test outputs.
    sample = dict(kind="rock", x_m=12.34, y_m=-4.56, conf=0.87, t_s=1234, seq=17)
    pkt = encode(**sample)
    out = decode(pkt)

    as_json = json.dumps({"type": "rock", "x": 12.34, "y": -4.56,
                          "conf": 0.87, "t": 1234, "seq": 17}).encode()

    print("binary packet :", pkt.hex(" "), f"({len(pkt)} bytes)")
    print("decoded       :", out)
    print("JSON size     :", len(as_json), "bytes")
    print("ratio         :", f"{len(as_json) / len(pkt):.1f}x smaller")
    print("position error:", f"{abs(out['x'] - 12.34):.3f} m, {abs(out['y'] + 4.56):.3f} m (max 0.05 m)")

    corrupted = bytearray(pkt)
    corrupted[2] ^= 0x01                      # flip one bit
    print("corrupted ->  :", decode(bytes(corrupted)), "(None = rejected, good)")
