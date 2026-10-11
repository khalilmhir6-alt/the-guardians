# Project C.R.A.T.E.R.

A lunar scout rover explores a permanently shadowed crater with no GPS, no cloud and
only a weak, lossy radio. The rover must find hazards (rocks, craters), work out where
they are in metres from a single camera using geometry alone, and push that knowledge
to Earth in a handful of bytes that still arrive when the link drops packets and goes
black. Earth turns whatever survives into a live map with honest data-age and
position-uncertainty. Everything is software-emulated: two separate programs (rover and
Earth) joined only by a lossy UDP link, plus the required
Perception -> Telemetry -> Dashboard flow.

```
 frames      ┌ detector ┐  ┌ tracker ┐  ┌ localize ┐  ┌ sender ┐            ┌ earth ┐   ┌ dashboard ┐
 640×640 ──► │ rock /   │─►│ IoU +   │─►│ pixel →  │─►│ 9 B ×3 │═══ lossy ═►│ dedup │──►│ /state →  │
             │ crater   │  │ confirm │  │ ground   │  │ budget │    UDP     │ + log │   │ canvas    │
             └──────────┘  └─────────┘  └──────────┘  └────────┘  (UDP 9000→9001)└───────┘   └───────────┘
                (numpy/opencv)                                       12 bytes/…  stdlib only   plain HTML/JS
```

Rover and Earth are separate processes; `rover/` never imports `earth/` and vice-versa.
The only contract between them is the 9-byte packet from `common/codec.py`.

## How to run

```powershell
# from C:\guardians\crater
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt      # perception deps only
.venv\Scripts\python run_all.py                              # link + earth + rover, opens dashboard
.venv\Scripts\python tests\run_tests.py                      # writes results\test_report.md + results.json
```

`run_all.py` starts link + Earth + rover as separate processes, restarts a crashed child
(logged), streams frames -> dashboard, and stops everything on Ctrl+C or `--duration N`.
Dashboard: <http://127.0.0.1:8080/>. With no model present the pipeline uses `FakeDetector`
(synthetic scene) so it runs today. To use the real detector, put the trained weights at
`config.json: yolo.weights` (`models/best.pt` by default; `.pt` loads via ultralytics,
`.onnx` via onnxruntime, CPU only with `yolo.cpu_threads`) and set `"detector": "yolo"`.
Classes are mapped **by name** from the model's own names list (`rock`/`crater`), never by
index. If `"detector": "yolo"` but the weights are missing or fail to load, the rover logs
a clear warning and falls back to `FakeDetector` - it never crashes.

## Decisions — real constraint → our choice → measured number

| Constraint | Our choice | Measured number |
|---|---|---|
| Venue proxy blocks installs | Stdlib-only transport/Earth/dashboard; perception deps isolated in `requirements.txt`; all ports/rates in `config.json` | runs offline |
| Radio is lossy and goes black | 3 copies spaced 6 s (> blackout 5 s) + 2 s heartbeat, sliding 5 pkt/s budget | hazard delivery 43.8% → **87.5%** at 30% loss; **100%** at 10% loss with blackouts (3× vs 1×) |
| Payload is precious | Compress each detection to the 9-byte packet instead of JSON | **9 B vs 76 B** (8.44×); end-to-end **837 B vs 8182 B** (9.78×) |
| Bits flip over the air | CRC-8 rejects corrupt packets at Earth | **72/72** single-bit flips rejected (100%) |
| Detector emits false positives | Confirm a track after 3 consecutive hits | 1-frame FP produced **0** reports |
| Pixel noise → position error | First-order error `(r²+z²)/z · δv/fy`, sent as a class so Earth can size the ring | vs ground truth mean **0.34 m**, max **1.93 m**; back-projection unit test **< 1e-14 m** |
| Resources are bounded | Queue 64, drop oldest copy on overflow, log loudly, timeouts everywhere | **0** hazard copies dropped in a 60 s run |
| Model order is unknown | Map classes by name from the model's own names list, never by index | DFKI model (0=crater, 1=rock) handled without hard-coding an index |
| Model may be absent at code freeze | `make_detector` catches a missing/failing load and falls back to `FakeDetector` (logged, never crashes) | rover ran 10 s with `"detector": "yolo"` and no weights, exit 0, fallback warning in the log |
| Many hazards appear at once | Queue ordered by **due time**, not enqueue time | fixed a head-of-line stall that inflated mean arrival age **11.5 s → 1.12 s** |
| Operator must read the map fast | Uncertainty ring, age fade (min 40%), auto-zoom, live hazard list | see dashboard section |

## Spec requirements vs team choices

**From the spec (not ours to change):** 9-byte packet format + CRC-8 (`codec.py`, used
as-is); lossy UDP link with random loss and blackouts (`link.py`, used as-is); the
Perception→Telemetry→Dashboard flow; rover and Earth as separate programs; stdlib-only
Earth/dashboard, no database/web framework; tracker confirms after 3 hits / drops after 5
misses; localization via bottom-centre pinhole ray cast to ground; heartbeat every 2 s;
the repetition + bandwidth-budget design; dedup key `(type, x, y, rover_time)`; dashboard
polls `/state` every 500 ms.

**Team choices (ours):** heading packed into the 4-bit heartbeat confidence field in
22.5° steps; hazard reported only within an 8 m range gate, with the localization error
class carried in two spare bits of byte 0; rover-side spatial-memory de-duplication so the
same hazard is never re-sent; on queue overflow drop the *oldest* copy first; the
uncertainty-ring / fade / list dashboard presentation; all thresholds and rates in
`config.json`. The `codec.py` change is additive (an optional `err_class` argument and an
`err` decode key) and preserves the original 9-byte layout.

## Automated test outputs

`tests/run_tests.py` writes `results/test_report.md` and `results/results.json`. Every
number in this README comes from that run; nothing is asserted without a measured value.
Scene-dependent tests report their **data source** at the top of the report (currently the
synthetic fallback, because `data/` is produced later by `blender/make_scene.py`; when
Blender data exists the same tests run against it with no code change).

Perception FPS/RAM and per-class precision/recall are reported automatically only when
`yolo.weights` exists **and** `data/dfki_test/images` is present; the group then measures
on CPU with `yolo.cpu_threads` at imgsz **640 and 416**, scoring predictions against the
YOLO-format ground-truth labels in `data/dfki_test/labels` (IoU >= 0.5, class by name).
Otherwise the group is written as `skipped: no model`.

## Known limits

- **Flat-ground assumption:** the ray/plane intersection assumes local ground is `z = 0`;
  real slopes and the depth of craters are not modelled, so crater centres are approximate.
- **Uncertainty is first-order,** not a covariance; it grows with range and only a coarse
  4-level class crosses the link.
- **Over-merge:** the range-based merge radius (up to ~1.1 m at the 8 m gate) can absorb
  two distinct hazards that are genuinely closer than that; measured recall is 16/23
  unique hazards on the synthetic scene.
- **Sim-to-real gap:** synthetic (or Blender-rendered) frames do not match the real
  lighting/texture distribution; the detector is plug-and-play but untested on real data.
- **No ice class:** the hazard taxonomy is rock / crater only.

## Dataset credit

Model training data: Shete, S. (2025), *Lunar Rocks and Craters Dataset for Instance
Segmentation (YOLO and COCO Format)*, DFKI, Zenodo, doi:10.5281/zenodo.14673237, CC BY 4.0.
