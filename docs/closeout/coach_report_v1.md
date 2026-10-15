# Coach Report V1 — draft

Run: `pipeline_v4_fresh`

Branch: `identity_aware`

Observed possession frames: 747

Tactical coverage: 405/747 tracking frames

This report describes only the analyzed clip.

## Observed possession

Known A/B possession: 407/747 frames (54.5%).

- Team A: 274 frames; 67.3% of known A/B possession
- Team B: 133 frames; 32.7% of known A/B possession
- Contested: 2 frames
- Unknown: 338 frames

## Possession phases

28 detected phases; all episode ranges are in coach_report_v1.json.
- Longest Team A phase: frames 401–611 (211 frames).
- Longest Team B phase: frames 32–98 (67 frames).

## Team shape

Complete tactical frames by possession: A 219, B 0, Contested 0, Unknown 186.

- All complete tactical frames, Team A (405 frames): width 38.413 m, length 32.367 m, compactness 20.930 m. Evidence: 18 frame ranges; full ranges in coach_report_v1.json.
- All complete tactical frames, Team B (405 frames): width 31.614 m, length 22.227 m, compactness 14.649 m. Evidence: 18 frame ranges; full ranges in coach_report_v1.json.
- During Team A possession, Team A (219 frames): width 39.078 m, length 35.849 m, compactness 22.118 m. Evidence: 10 frame ranges; full ranges in coach_report_v1.json.
- During Team A possession, Team B (219 frames): width 32.020 m, length 20.932 m, compactness 14.642 m. Evidence: 10 frame ranges; full ranges in coach_report_v1.json.

### In-possession versus out-of-possession

Unavailable: the selected tactical frames do not contain enough observations in both possession states for either team.

## Detected pass-to-reception sequences

- Pipeline detections 27 (Pass) and 28 (Reception), Team A, Pass frames [407, 442], Reception frame 443: both-team nearest-opponent metric 4.970 m in the pre-pass window (frames [402, 406]) versus 6.676 m in the post-reception window (frames [444, 448]); window difference +1.706 m.
- Pipeline detections 29 (Pass) and 30 (Reception), Team A, Pass frames [492, 539], Reception frame 540: both-team nearest-opponent metric 7.723 m in the pre-pass window (frames [487, 491]) versus 9.409 m in the post-reception window (frames [541, 545]); window difference +1.686 m.
- Detection 33: unavailable (missing_spacing_frames).
- Detection 25: unavailable (no_paired_pass).

Metric: the median nearest-opponent distance across both teams at each frame, summarized across five frames on each side. These are observed associations, not receiver pressure or causal effects.
