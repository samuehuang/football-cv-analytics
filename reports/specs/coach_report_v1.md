# Coach Report V1 specification

Status: draft implementation. The current generator reports possession, phases,
coverage, and descriptive team shape for one analyzed clip. Event-linked
observations are not implemented yet.

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

## Next implementation slice

Join detected Pass and Reception event IDs to complete, consecutive pre/post
frame windows and available inter-team spatial metrics. Suppress observations
when the window crosses a frame gap or possession boundary. Keep causal and
coaching recommendations out of the deterministic report.
