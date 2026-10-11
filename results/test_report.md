# C.R.A.T.E.R. — automated test report

Generated 2026-10-11T05:15:14Z · Python 3.12.7 · win32 · wall 0.1 s

**Data source for scene-dependent tests:** synthetic fallback (data/ not generated)

Every number below is printed by `tests/run_tests.py`; no claim is asserted without a measured value.

## 1. Telemetry codec

| metric | value |
|---|---|
| packet size | 9 bytes |
| equivalent JSON size | 76 bytes |
| compression ratio | 8.44x smaller |
| round-trip max position error | 0.04 m |
| single-bit corruption rejected | 72/72 (100.0%) |
| error-class field round-trip | True |

## 2. Link survival (in-process, fixed seed, blackout 5 s / 30 s)

| loss | blackouts | repeats | delivered | delivery % |
|---|---|---|---|---|
| 0% | no | 1x | 16/16 | 100.0% |
| 0% | no | 3x | 16/16 | 100.0% |
| 0% | yes | 1x | 15/16 | 93.8% |
| 0% | yes | 3x | 16/16 | 100.0% |
| 10% | no | 1x | 13/16 | 81.2% |
| 10% | no | 3x | 16/16 | 100.0% |
| 10% | yes | 1x | 12/16 | 75.0% |
| 10% | yes | 3x | 16/16 | 100.0% |
| 30% | no | 1x | 7/16 | 43.8% |
| 30% | no | 3x | 14/16 | 87.5% |
| 30% | yes | 1x | 7/16 | 43.8% |
| 30% | yes | 3x | 14/16 | 87.5% |

## 3. Localization (source: synthetic fallback (data/ not generated))

- back-projection round-trip max error: **1.11e-14 m** (< 1 cm: True)

| range (m) | measured err (m) | formula err (m) | ratio |
|---|---|---|---|
| 2.0 | 0.06 | 0.073 | 0.82 |
| 5.0 | 0.368 | 0.442 | 0.83 |
| 8.0 | 0.884 | 1.128 | 0.78 |

- pipeline detect->localize vs ground-truth rocks: 866 detections matched, mean **0.3401 m**, max **1.9341 m**

## 4. Tracker

- persistent tracks confirmed: **2** (expected 2)
- one-frame false positive produced a report: **0** (expected 0)
- duplicate detections produced an extra report: **0** (expected 0)
- tentative tracks killed unconfirmed: **2**
- two same-class hazards 0.4 m apart -> reported **1** (expected 1, spatial memory)

## 5. End to end (fake detector, lossy link in-process)

Source: **synthetic fallback (data/ not generated)** (fixed link seed).

| metric | value |
|---|---|
| ground-truth hazards | 23 |
| rover confirmed events | 16 |
| Earth unique hazards | 16 |
| GT recall | 69.6% |
| precision | 100.0% |
| mean age at arrival | 1.12 s |
| packets sent | 93 |
| binary bytes | 837 |
| equivalent JSON bytes | 8182 |
| JSON / binary | 9.78x |
| duplicate events suppressed on rover | 23 |
| frames held by 8 m range gate | 355 |

## 6. Perception (CPU-only)

Skipped: no model. Reported automatically once the weights and `data/dfki_test/images` exist.

## Known limits

- Flat-ground assumption (z = 0); craters are approximated.
- Uncertainties are a first-order estimate (r^2+z^2)/z * (dv/fy), not a covariance.
- Over-merge: the range-based merge radius (up to ~1.1 m at the 8 m gate) can absorb distinct hazards that are close together.
- Sim-to-real gap; no ice class.
