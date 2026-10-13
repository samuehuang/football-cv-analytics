"""Evidence windows for detected pass-to-reception sequences."""

from pathlib import Path

import numpy as np
import pandas as pd


SPACING_FILES = {
    "strict": "inter_team_spatial_metrics_frame.csv",
    "identity_aware": "inter_team_spatial_metrics_frame_identity_aware.csv",
}


def read_input(path, required):
    if not path.is_file():
        raise ValueError(f"Missing input: {path}")
    data = pd.read_csv(path)
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return data


def integer_column(data, column):
    values = pd.to_numeric(data[column], errors="raise")
    if values.isna().any() or not np.isfinite(
        values.to_numpy(dtype=float)
    ).all():
        raise ValueError(f"Invalid {column}")
    if not (values == values.astype("int64")).all():
        raise ValueError(f"Non-integer {column}")
    data[column] = values.astype("int64")


def analyze_event_windows(run_dir, branch, phases, tactical_frames, window=5):
    run_dir = Path(run_dir)
    events_path = run_dir / "data/football_events_v1.csv"
    spacing_path = (
        run_dir / "analytics/outputs" / SPACING_FILES[branch]
    )

    events = read_input(
        events_path,
        ("event_id", "event_type", "start_frame", "end_frame", "team"),
    )
    spacing = read_input(
        spacing_path, ("frame", "median_nearest_opponent_m")
    )

    for column in ("event_id", "start_frame", "end_frame"):
        integer_column(events, column)
    integer_column(spacing, "frame")

    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate event ID")
    if spacing["frame"].duplicated().any():
        raise ValueError("Duplicate spacing frame")
    if (events["start_frame"] > events["end_frame"]).any():
        raise ValueError("Event ends before it starts")
    if not set(spacing["frame"]).issubset(set(tactical_frames)):
        raise ValueError(
            "Spacing frame absent from selected tactical branch"
        )

    metric = spacing.set_index("frame")["median_nearest_opponent_m"]
    selected = events.loc[
        events["event_type"].isin(("Pass", "Reception"))
    ]
    passes = selected.loc[
        selected["event_type"] == "Pass"
    ].sort_values("event_id")
    receptions = selected.loc[
        selected["event_type"] == "Reception"
    ]

    used_receptions = set()
    supported = []
    unavailable = []

    for passing in passes.itertuples(index=False):
        matches = receptions.loc[
            (receptions["start_frame"] == passing.end_frame + 1)
            & (receptions["team"] == passing.team)
        ]
        if "target_team" in events and pd.notna(passing.target_team):
            matches = matches.loc[
                matches["team"] == passing.target_team
            ]

        if len(matches) != 1 or passing.team not in ("A", "B"):
            unavailable.append({
                "pass_event_id": int(passing.event_id),
                "reason": "unpaired_or_ambiguous_reception",
            })
            continue

        reception = matches.iloc[0]
        reception_id = int(reception["event_id"])

        competing_passes = passes.loc[
            (passes["end_frame"] == int(reception["start_frame"]) - 1)
            & (passes["team"] == passing.team)
        ]
        if len(competing_passes) > 1:
            used_receptions.add(reception_id)
            unavailable.append({
                "pass_event_id": int(passing.event_id),
                "reception_event_id": reception_id,
                "reason": "ambiguous_reception",
            })
            continue

        # Track IDs are used only to check consistency between detections.
        # They are not treated as verified physical player identities.
        if (
            "target_player_id" in events
            and "player_id" in events
            and pd.notna(passing.target_player_id)
            and pd.notna(reception["player_id"])
        ):
            try:
                same_track = (
                    float(passing.target_player_id)
                    == float(reception["player_id"])
                )
            except (TypeError, ValueError):
                same_track = (
                    str(passing.target_player_id)
                    == str(reception["player_id"])
                )
            if not same_track:
                unavailable.append({
                    "pass_event_id": int(passing.event_id),
                    "reason": "target_track_mismatch",
                })
                continue

        used_receptions.add(reception_id)

        before_frames = list(
            range(int(passing.start_frame) - window,
                  int(passing.start_frame))
        )
        after_frames = list(
            range(int(reception["end_frame"]) + 1,
                  int(reception["end_frame"]) + 1 + window)
        )
        evidence = {
            "pass_event_id": int(passing.event_id),
            "reception_event_id": reception_id,
            "team": passing.team,
            "pass_frames": [
                int(passing.start_frame),
                int(passing.end_frame),
            ],
            "reception_frame": int(reception["start_frame"]),
            "before_frames": [
                before_frames[0], before_frames[-1],
            ],
            "after_frames": [
                after_frames[0], after_frames[-1],
            ],
        }

        same_phase = any(
            phase["possession"] == passing.team
            and phase["start_frame"] <= before_frames[0]
            and phase["end_frame"] >= after_frames[-1]
            for phase in phases
        )
        missing = [
            frame for frame in before_frames + after_frames
            if frame not in metric.index
        ]

        if not same_phase:
            reason = "phase_boundary_or_unknown_possession"
        elif missing:
            reason = "missing_spacing_frames"
        else:
            before = pd.to_numeric(
                metric.loc[before_frames], errors="coerce"
            ).to_numpy(dtype=float)
            after = pd.to_numeric(
                metric.loc[after_frames], errors="coerce"
            ).to_numpy(dtype=float)
            reason = (
                None
                if np.isfinite(before).all()
                and np.isfinite(after).all()
                else "invalid_metric"
            )

        if reason:
            unavailable.append({**evidence, "reason": reason})
            continue

        before_median = round(float(np.median(before)), 3)
        after_median = round(float(np.median(after)), 3)
        supported.append({
            **evidence,
            "status": "supported",
            "metric": "median_nearest_opponent_m",
            "unit": "m",
            "valid_frames": {"before": window, "after": window},
            "before_median": before_median,
            "after_median": after_median,
            "difference_after_minus_before": round(
                after_median - before_median, 3
            ),
        })

    for reception in receptions.itertuples(index=False):
        if int(reception.event_id) not in used_receptions:
            unavailable.append({
                "reception_event_id": int(reception.event_id),
                "reason": "no_paired_pass",
            })

    return {
        "window_frames_per_side": window,
        "metric": "median_nearest_opponent_m",
        "unit": "m",
        "source_files": [str(events_path), str(spacing_path)],
        "supported": supported,
        "unavailable": unavailable,
    }
