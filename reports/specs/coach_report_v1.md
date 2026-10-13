# Coach Report V1 specification

Status: draft implementation. The generator reports possession, phases,
coverage, and descriptive team shape for one analyzed clip. Optional
pass-to-reception spacing observations are available with
`--include-event-windows`.

## Inputs

All paths are relative to `--run-dir`.

| Source | Required fields | Purpose |
| --- | --- | --- |
| `possession/data/possession_v6_frames.csv` | `frame`, `possession_v6` | Observed possession states and frame denominator |
| `possession/data/possession_v6_episodes.csv` | `episode_id`, `start_frame`, `end_frame`, `duration_frames`, `possession` | Inclusive phase boundaries |
| `analytics/outputs/team_tactical_metrics_frame_identity_aware.csv` or `_v2.csv` | `frame`, `team`, `team_width_m`, `team_length_m`, `compactness_m` | Complete-frame team geometry |
| `analytics/outputs/tactical_coverage_comparison.csv` | `mode`, `complete_frames`, `total_frames` | Coverage against tracking frames |

Exactly one tactical branch, `strict` or `identity_aware`, is selected per report.
Identity-aware corrects high-confidence team assignments; it does not establish
physical player identity.

## Calculations

- Possession counts use distinct observed frames. Team share divides A or B
  frames by `A + B` frames; Unknown and Contested are displayed separately.
  This is clip-level observed possession, not full-match possession.
- Episodes must agree with every covered frame and have inclusive durations.
  Missing frames cannot be filled or joined across a gap.
- Tactical coverage uses the tracking-frame denominator from the coverage CSV.
  The per-possession tactical counts use only frames present in both input tables.
  These denominators must not be substituted for one another.
- Descriptive medians use only complete tactical frames with both teams present.
  "All complete frames" may include Unknown possession. A- or B-possession
  medians are reported only when at least 10 valid frames exist.
- An in/out comparison requires at least 10 valid tactical frames in each
  possession state for that team. Otherwise its status is `insufficient_data`.
  This threshold is a reporting rule, not a statistical significance test.

## Output and validation

By default, reports are written to `<run-dir>/reports/`. Use `--output-dir` to keep strict and identity-aware outputs in separate directories.

The CLI writes `reports/coach_report_v1.json` and `.md` within the chosen run.
The JSON retains full frame ranges; Markdown summarizes them. Neither output
claims that a measured association was caused by an event.

Reject duplicate possession frames, duplicate tactical `(frame, team)` rows,
invalid episode boundaries, non-finite reported metrics, incomplete A/B tactical
pairs, and coverage counts inconsistent with the selected tactical CSV.
Existing report files are retained unless `--overwrite` is specified.

For the validated `pipeline_v4_fresh` sample, identity-aware tactical coverage
is 405/747. Of those complete frames, 219 have Team A possession and zero have
Team B possession. An in/out shape comparison is therefore unavailable.

## Optional pass-to-reception windows

`--include-event-windows` reads `data/football_events_v1.csv` and the selected
branch's `inter_team_spatial_metrics_frame*.csv`. Without the option, the
original report works without those inputs.

A Pass pairs with exactly one same-team Reception beginning on the next frame.
If the event files supply target and receiving track IDs, they must agree.
Those IDs check pipeline consistency; they are not verified physical identities.

For each pair, take five consecutive frames immediately before `Pass.start_frame`
and five immediately after `Reception.end_frame`. Both windows and the detected
sequence must fit inside one possession episode for the event team. All ten
spacing frames must exist and have finite metric values. Missing frames, a
phase boundary, ambiguous pairing, or mismatched tracks suppress the
observation with a machine-readable reason.

`median_nearest_opponent_m` is calculated across players from both teams at
each frame. The report then compares the median of that frame metric in the
two five-frame windows. It does not measure receiver pressure. A difference
across a detected pass sequence is an association, not an effect of the pass.

For the validated `pipeline_v4_fresh` sample, event pairs 27/28 and 29/30
have complete windows. Pair 33/34 lacks spacing frames; Reception 25 has
no paired Pass. The output must preserve these omissions.

## Next implementation slice

Validate event-window behavior on another run and inspect detected events
against video before using these observations in a coaching context. Keep
causal explanations and recommendations outside the deterministic report.
