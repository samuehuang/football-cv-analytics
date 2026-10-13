#!/usr/bin/env python3
"""Generate a deterministic draft coach report from one completed pipeline run."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from event_windows import analyze_event_windows


METRICS = ("team_width_m", "team_length_m", "compactness_m")
TACTICAL_FILES = {
    "strict": "team_tactical_metrics_frame_v2.csv",
    "identity_aware": "team_tactical_metrics_frame_identity_aware.csv",
}


def read_csv(path, required):
    if not path.is_file():
        raise ValueError(f"Missing input: {path}")
    data = pd.read_csv(path)
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return data


def integer_column(data, column):
    values = pd.to_numeric(data[column], errors="raise")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid values in {column}")
    if not (values == values.astype("int64")).all():
        raise ValueError(f"Non-integer values in {column}")
    data[column] = values.astype("int64")


def frame_ranges(frames):
    result = []
    for frame in sorted(int(value) for value in frames):
        if result and frame == result[-1][1] + 1:
            result[-1][1] = frame
        else:
            result.append([frame, frame])
    return result


def build_report(run_dir, branch, min_frames, include_events=False):
    possession_path = run_dir / "possession/data/possession_v6_frames.csv"
    episodes_path = run_dir / "possession/data/possession_v6_episodes.csv"
    tactical_path = run_dir / "analytics/outputs" / TACTICAL_FILES[branch]
    coverage_path = run_dir / "analytics/outputs/tactical_coverage_comparison.csv"

    possession = read_csv(possession_path, ("frame", "possession_v6"))
    if possession.empty:
        raise ValueError("Possession file has no frames")
    integer_column(possession, "frame")
    if possession["frame"].duplicated().any():
        raise ValueError("Duplicate possession frame")
    possession["possession_v6"] = (
        possession["possession_v6"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )
    possession = possession.sort_values("frame")
    states_by_frame = possession.set_index("frame")["possession_v6"].to_dict()

    episodes = read_csv(
        episodes_path,
        ("episode_id", "start_frame", "end_frame", "duration_frames", "possession"),
    )
    for column in ("episode_id", "start_frame", "end_frame", "duration_frames"):
        integer_column(episodes, column)
    if episodes["episode_id"].duplicated().any():
        raise ValueError("Duplicate possession episode ID")
    episodes["possession"] = episodes["possession"].fillna("Unknown").astype(str)

    covered = set()
    phase_rows = []
    for row in episodes.itertuples(index=False):
        start, end = int(row.start_frame), int(row.end_frame)
        if start > end or int(row.duration_frames) != end - start + 1:
            raise ValueError(f"Invalid possession episode {row.episode_id}")
        for frame in range(start, end + 1):
            if frame in covered or states_by_frame.get(frame) != row.possession:
                raise ValueError(
                    f"Episode {row.episode_id} disagrees at frame {frame}"
                )
            covered.add(frame)
        phase_rows.append({
            "episode_id": int(row.episode_id),
            "start_frame": start,
            "end_frame": end,
            "duration_frames": int(row.duration_frames),
            "possession": row.possession,
        })
    if covered != set(states_by_frame):
        raise ValueError("Episodes do not cover exactly the possession frames")
    phase_rows.sort(key=lambda row: row["start_frame"])

    tactical = read_csv(tactical_path, ("frame", "team", *METRICS))
    integer_column(tactical, "frame")
    if not tactical["team"].isin(("A", "B")).all():
        raise ValueError("Unexpected team in tactical data")
    if tactical.duplicated(("frame", "team")).any():
        raise ValueError("Duplicate tactical frame/team")
    if not set(tactical["frame"]).issubset(states_by_frame):
        raise ValueError("Tactical frame outside the possession data")
    for metric in METRICS:
        tactical[metric] = pd.to_numeric(tactical[metric], errors="raise")
        if not np.isfinite(tactical[metric].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite tactical metric: {metric}")
    team_counts = tactical.groupby("frame")["team"].nunique()
    if not team_counts.eq(2).all():
        raise ValueError("Tactical frame without both A and B")

    coverage = read_csv(
        coverage_path, ("mode", "complete_frames", "total_frames")
    )
    selected = coverage.loc[coverage["mode"] == branch]
    if len(selected) != 1:
        raise ValueError(f"Expected one coverage row for {branch}")
    complete = int(selected.iloc[0]["complete_frames"])
    total = int(selected.iloc[0]["total_frames"])
    if complete != len(team_counts) or total <= 0 or complete > total:
        raise ValueError("Coverage disagrees with tactical frame data")

    counts = Counter(possession["possession_v6"])
    known = counts["A"] + counts["B"]
    tactical = tactical.merge(
        possession[["frame", "possession_v6"]],
        on="frame",
        validate="many_to_one",
    )

    valid_frames_by_possession = {
        state: int(tactical.loc[
            tactical["possession_v6"] == state, "frame"
        ].nunique())
        for state in sorted(counts)
    }

    descriptive_medians = []
    contexts = (
        ("all_complete_frames", tactical),
        ("A_possession", tactical.loc[tactical["possession_v6"] == "A"]),
        ("B_possession", tactical.loc[tactical["possession_v6"] == "B"]),
    )
    for context, context_rows in contexts:
        frames_in_context = sorted(int(x) for x in context_rows["frame"].unique())
        if len(frames_in_context) < min_frames:
            continue
        for team in ("A", "B"):
            team_rows = context_rows.loc[context_rows["team"] == team]
            descriptive_medians.append({
                "context": context,
                "team": team,
                "status": "supported",
                "valid_frames": len(frames_in_context),
                "frame_ranges": frame_ranges(frames_in_context),
                "median_m": {
                    metric: round(float(team_rows[metric].median()), 3)
                    for metric in METRICS
                },
            })

    comparisons = []
    for team in ("A", "B"):
        other = "B" if team == "A" else "A"
        team_rows = tactical.loc[tactical["team"] == team]
        in_rows = team_rows.loc[team_rows["possession_v6"] == team]
        out_rows = team_rows.loc[team_rows["possession_v6"] == other]
        enough = min(len(in_rows), len(out_rows)) >= min_frames

        for metric in METRICS:
            inside = round(float(in_rows[metric].median()), 3) if enough else None
            outside = round(float(out_rows[metric].median()), 3) if enough else None
            comparisons.append({
                "team": team,
                "metric": metric,
                "unit": "m",
                "status": "supported" if enough else "insufficient_data",
                "reason": None if enough else "too_few_frames",
                "in_possession_median": inside,
                "out_of_possession_median": outside,
                "difference_in_minus_out": (
                    round(inside - outside, 3) if enough else None
                ),
                "valid_frames": {
                    "in": len(in_rows),
                    "out": len(out_rows),
                },
                "eligible_frames": {
                    "in": counts[team],
                    "out": counts[other],
                },
                "frame_ranges": {
                    "in": frame_ranges(in_rows["frame"]),
                    "out": frame_ranges(out_rows["frame"]),
                },
            })

    windows = (
        analyze_event_windows(
            run_dir, branch, phase_rows, set(team_counts.index)
        )
        if include_events else None
    )
    frames = sorted(states_by_frame)
    missing_frames = set(range(frames[0], frames[-1] + 1)) - set(frames)
    return {
        "schema_version": "1.0-draft",
        "run_id": run_dir.name,
        "clip_scope": {
            "first_frame": frames[0],
            "last_frame": frames[-1],
            "missing_frame_ranges": frame_ranges(missing_frames),
        },
        "provenance": {
            "tactical_branch": branch,
            "inputs": {
                "possession_frames": str(possession_path),
                "possession_episodes": str(episodes_path),
                "tactical_frames": str(tactical_path),
                "tactical_coverage": str(coverage_path),
            },
        },
        "coverage": {
            "possession_frames": len(possession),
            "tactical_valid_frames": complete,
            "tactical_denominator_tracking_frames": total,
        },
        "possession": {
            "state_frames": dict(sorted(counts.items())),
            "known_team_frames": known,
            "team_shares_of_known": {
                team: counts[team] / known if known else None
                for team in ("A", "B")
            },
            "episodes": phase_rows,
        },
        "team_shape": {
            "min_state_frames": min_frames,
            "valid_frames_by_possession": valid_frames_by_possession,
            "descriptive_medians": descriptive_medians,
            "comparisons": comparisons,
        },
        "opponent_spacing": (
            {
                "status": (
                    "supported" if windows["supported"]
                    else "insufficient_data"
                ),
                "metric": windows["metric"],
                "unit": windows["unit"],
                "source_files": windows["source_files"],
            }
            if windows is not None
            else {
                "status": "not_requested",
                "reason": "include_event_windows_disabled",
            }
        ),
        "event_sequences": (
            windows if windows is not None
            else {
                "status": "not_requested",
                "reason": "include_event_windows_disabled",
            }
        ),
    }


def render_markdown(report):
    coverage = report["coverage"]
    possession = report["possession"]
    lines = [
        "# Coach Report V1 — draft",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Branch: `{report['provenance']['tactical_branch']}`  ",
        f"Observed possession frames: {coverage['possession_frames']}  ",
        (
            "Tactical coverage: "
            f"{coverage['tactical_valid_frames']}/"
            f"{coverage['tactical_denominator_tracking_frames']} tracking frames"
        ),
        "",
        "This report describes only the analyzed clip.",
        "",
        "## Observed possession",
        "",
    ]
    known = possession["known_team_frames"]
    observed = coverage["possession_frames"]
    lines.append(
        f"Known A/B possession: {known}/{observed} frames "
        f"({known / observed:.1%})."
    )
    lines.append("")
    for team in ("A", "B"):
        share = possession["team_shares_of_known"][team]
        label = f"{share:.1%}" if share is not None else "unavailable"
        lines.append(
            f"- Team {team}: {possession['state_frames'].get(team, 0)} frames; "
            f"{label} of known A/B possession"
        )
    for state, count in possession["state_frames"].items():
        if state not in ("A", "B"):
            lines.append(f"- {state}: {count} frames")

    phases = possession["episodes"]
    lines += ["", "## Possession phases", "",
              f"{len(phases)} detected phases; all episode ranges are in coach_report_v1.json."]
    for team in ("A", "B"):
        team_phases = [phase for phase in phases if phase["possession"] == team]
        if team_phases:
            longest = max(team_phases, key=lambda phase: phase["duration_frames"])
            lines.append(
                f"- Longest Team {team} phase: frames "
                f"{longest['start_frame']}–{longest['end_frame']} "
                f"({longest['duration_frames']} frames)."
            )

    lines += ["", "## Team shape", ""]
    available = report["team_shape"]["valid_frames_by_possession"]
    lines.append(
        "Complete tactical frames by possession: "
        + ", ".join(
            f"{state} {count}" for state, count in available.items()
        )
        + "."
    )
    lines.append("")

    labels = {
        "all_complete_frames": "All complete tactical frames",
        "A_possession": "During Team A possession",
        "B_possession": "During Team B possession",
    }
    for item in report["team_shape"]["descriptive_medians"]:
        metrics = item["median_m"]
        lines.append(
            f"- {labels[item['context']]}, Team {item['team']} "
            f"({item['valid_frames']} frames): "
            f"width {metrics['team_width_m']:.3f} m, "
            f"length {metrics['team_length_m']:.3f} m, "
            f"compactness {metrics['compactness_m']:.3f} m. "
            f"Evidence: {len(item['frame_ranges'])} frame ranges; full ranges in coach_report_v1.json."
        )

    lines += ["", "### In-possession versus out-of-possession", ""]
    supported = [
        item for item in report["team_shape"]["comparisons"]
        if item["status"] == "supported"
    ]
    if not supported:
        lines.append(
            "Unavailable: the selected tactical frames do not contain "
            "enough observations in both possession states for either team."
        )
    for item in supported:
        lines.append(
            f"- Team {item['team']} {item['metric']}: "
            f"in {item['in_possession_median']:.3f} m "
            f"({item['valid_frames']['in']} frames), "
            f"out {item['out_of_possession_median']:.3f} m "
            f"({item['valid_frames']['out']} frames); "
            f"difference {item['difference_in_minus_out']:+.3f} m. "
            f"Evidence frames: in {item['frame_ranges']['in']}, "
            f"out {item['frame_ranges']['out']}."
        )

    lines += ["", "## Detected pass-to-reception sequences", ""]
    windows = report["event_sequences"]
    if windows.get("status") == "not_requested":
        lines.append("Not included in this report.")
    else:
        for item in windows["supported"]:
            lines.append(
                f"- Pipeline detections {item['pass_event_id']} (Pass) "
                f"and {item['reception_event_id']} (Reception), "
                f"Team {item['team']}, Pass frames {item['pass_frames']}, "
                f"Reception frame {item['reception_frame']}: "
                f"both-team nearest-opponent metric "
                f"{item['before_median']:.3f} m in the pre-pass window "
                f"(frames {item['before_frames']}) versus "
                f"{item['after_median']:.3f} m in the post-reception window "
                f"(frames {item['after_frames']}); window difference "
                f"{item['difference_after_minus_before']:+.3f} m."
            )
        for item in windows["unavailable"]:
            event_id = item.get(
                "pass_event_id", item.get("reception_event_id")
            )
            lines.append(
                f"- Detection {event_id}: unavailable "
                f"({item['reason']})."
            )
        lines += [
            "",
            "Metric: the median nearest-opponent distance across "
            "both teams at each frame, summarized across five frames "
            "on each side. These are observed associations, not "
            "receiver pressure or causal effects.",
        ]
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--branch", required=True, choices=TACTICAL_FILES)
    parser.add_argument("--min-state-frames", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-event-windows", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.min_state_frames < 1:
        parser.error("--min-state-frames must be at least 1")

    run_dir = args.run_dir.resolve()
    report = build_report(
        run_dir, args.branch, args.min_state_frames,
        args.include_event_windows,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir else run_dir / "reports"
    )
    json_path = output_dir / "coach_report_v1.json"
    markdown_path = output_dir / "coach_report_v1.md"

    if not args.overwrite and (json_path.exists() or markdown_path.exists()):
        parser.error("Report already exists; use --overwrite to replace it")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
