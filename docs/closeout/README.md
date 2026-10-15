# Coach Report V1 — Sample Analysis

[Project overview](../../README.md) · [Generated report](coach_report_v1.md) · [JSON evidence](coach_report_v1.json)

This case study presents the completed reporting prototype on a 30-second football clip. The pipeline combines computer-vision outputs, geometric measurements, and temporal rules to produce descriptive observations linked to video frames.

The snapshot uses baseline commit `a8b2889`, run `pipeline_v4_fresh`, and the `identity_aware` tactical dataset. The clip runs at 25 frames per second; the analysis covers 747 frames, numbered 3–749. Results describe this clip only.

## Method and coverage

Whole-team tactical metrics use frames containing ten selected player tracks per team. This rule checks data availability; it does not verify physical identities or guarantee correct role assignments. Possession states and tactical coverage are reported separately so that missing observations remain visible.

| Measure | Result |
| --- | ---: |
| Analyzed frames | 747 |
| Frames assigned to Team A or B | 407 / 747, 54.5% |
| Team A / Team B frames | 274 / 133 |
| Unknown / contested frames | 338 / 2 |
| Complete tactical frames | 405 / 747, 54.2% |
| Complete frames during A / B possession | 219 / 0 |

Team A accounts for 67.3% of the 407 known A/B assignments. This percentage excludes unknown and contested states and does not represent full-match possession. Consecutive video frames are correlated, so the frame count is not a count of independent tactical situations.

## Team shape during Team A possession

| Median metric | Team A | Team B |
| --- | ---: | ---: |
| Width | 39.078 m | 32.020 m |
| Length | 35.849 m | 20.932 m |
| Compactness distance | 22.118 m | 14.642 m |

Both teams are measured over the same 219 complete frames. Width and length are the 90th–10th percentile spans of the corresponding pitch coordinates, calculated per frame and summarized by the median. Compactness is the mean player distance from the team centroid per frame, also summarized by the median; smaller values indicate a tighter tracked group.

Team A's tracked players were more spread out than Team B's during these selected A-possession frames. This provides a basis for examining whether the wider shape offered usable passing options. The measurements alone do not establish a formation, tactical intent, or attacking effectiveness.

The baseline has no complete tactical frames during Team B possession, so neither team's in-possession versus out-of-possession comparison is available. This coverage gap does not imply that Team B never had possession.

## Spacing around detected passing sequences

| Detected sequence | Pre-pass median | Post-reception median | Difference |
| --- | ---: | ---: | ---: |
| Pass 27 → reception 28; reception frame 443 | 4.970 m | 6.676 m | +1.706 m |
| Pass 29 → reception 30; reception frame 540 | 7.723 m | 9.409 m | +1.686 m |

The first comparison uses frames 402–406 and 444–448; the second uses frames 487–491 and 541–545. Each window contains five frame observations, equivalent to 0.20 seconds at 25 fps.

The metric is the per-frame median of nearest-opponent distances across both teams, summarized across each window. It describes aggregate spacing rather than the receiver's immediate pressure. Both sequences coincided with increased spacing, but the comparison does not isolate the effect of the pass from other player movement or elapsed time.

The report omits detection 33 because spacing frames are missing and detection 25 because no paired pass is available. These omissions and the supporting frame ranges are preserved in the JSON evidence.

## Limitations

| Limitation | Effect on interpretation |
| --- | --- |
| Short-track filtering | A 20-second track-span rule excludes a short Team A fragment and contributes to the absence of complete B-possession tactical frames. |
| Identity and role continuity | Tracking IDs are not verified player identities. Review identified a goalkeeper-class inconsistency and a likely same-player track fragment. |
| Missing observations | Off-camera players, occlusion, missed detections, classification errors, and filtering can all reduce coverage. The cause cannot always be determined from the output alone. |
| Conservative possession rules | Unknown states include uncontrolled transit and missing-ball observations. Unknown is an explicit pipeline state, not automatically a detection error. |
| Limited evaluation | Results come from one short clip. Report tests check software behavior; detection and event accuracy have not been established on an independent labelled benchmark. |

The prototype supports descriptive clip review and evidence inspection. Recruitment decisions, full-match tactical conclusions, and causal claims require additional data and validation.

## Reproduction

The adjacent Markdown and JSON files are fixed output snapshots and can be inspected without running inference. Reproducing the sample requires the completed `pipeline_v4_fresh` run CSVs, which are not bundled with these report snapshots.

From the repository root, using the configured Python environment:

```bash
python -m unittest discover -s reports/tests -v
python reports/coach_report_v1.py --run-dir runs/pipeline_v4_fresh --branch identity_aware --include-event-windows --output-dir runs/pipeline_v4_fresh/reports_closeout
```

The report command reads existing CSVs; it does not rerun video inference. Choose a new output directory or supply `--overwrite` to replace an existing generated report. The `--branch` option selects the tactical dataset variant, independently of the Git branch.

To analyze another video, follow the [pipeline setup and execution instructions](../../README.md#quick-start), run tactical analytics, then generate the report for that run.

Inputs used for this snapshot, relative to `runs/pipeline_v4_fresh/`:

- `possession/data/possession_v6_frames.csv`
- `possession/data/possession_v6_episodes.csv`
- `analytics/outputs/team_tactical_metrics_frame_identity_aware.csv`
- `analytics/outputs/tactical_coverage_comparison.csv`
- `analytics/outputs/inter_team_spatial_metrics_frame_identity_aware.csv`
- `data/football_events_v1.csv`

Report implementation: [coach_report_v1.py](../../reports/coach_report_v1.py), [event_windows.py](../../reports/event_windows.py), and the [report specification](../../reports/specs/coach_report_v1.md).
