#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "track_id",
    "from_team",
    "to_team",
    "before_start_frame",
    "before_end_frame",
    "after_start_frame",
    "after_end_frame",
    "before_duration_sec",
    "after_duration_sec",
    "transition_time_sec",
    "transition_gap_sec",
    "transition_distance_m",
    "transition_speed_mps",
]


PAIR_COLUMNS = [
    "track_id_a",
    "track_id_b",
    "transition_a",
    "transition_b",
    "transition_time_a_sec",
    "transition_time_b_sec",
    "transition_time_delta_sec",
    "closest_frame",
    "closest_time_sec",
    "minimum_distance_m",
    "confidence",
]


REQUIRED_COLUMNS = {
    "frame",
    "time_sec",
    "track_id",
    "team",
    "pitch_x_final_m",
    "pitch_y_final_m",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect high-confidence reciprocal team-label transitions "
            "that may indicate player-tracking ID switches."
        )
    )

    parser.add_argument(
        "--input",
        default="outputs/tracking_data_clean_v2.csv",
        help=(
            "Clean player tracking CSV. "
            "Default: outputs/tracking_data_clean_v2.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help=(
            "Directory for QA CSV outputs. "
            "Default: outputs"
        ),
    )

    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=2.0,
        help=(
            "Minimum duration before and after a team transition. "
            "Short label flickers are ignored."
        ),
    )

    parser.add_argument(
        "--max-transition-gap-sec",
        type=float,
        default=0.20,
        help=(
            "Maximum time gap between the last observation of the old "
            "team segment and the first observation of the new segment."
        ),
    )

    parser.add_argument(
        "--pair-time-window-sec",
        type=float,
        default=0.30,
        help=(
            "Maximum difference between two reciprocal transition times."
        ),
    )

    parser.add_argument(
        "--proximity-window-sec",
        type=float,
        default=0.40,
        help=(
            "Time window around a reciprocal transition used to measure "
            "the minimum distance between both tracks."
        ),
    )

    parser.add_argument(
        "--max-pair-distance-m",
        type=float,
        default=3.0,
        help=(
            "Maximum nearby distance for a reciprocal switch pair "
            "to be considered high-confidence."
        ),
    )

    parser.add_argument(
        "--max-transition-speed-mps",
        type=float,
        default=15.0,
        help=(
            "Maximum implied motion speed across the team transition."
        ),
    )

    return parser.parse_args()


def build_team_segments(track: pd.DataFrame):
    track = (
        track
        .sort_values(["frame", "time_sec"])
        .reset_index(drop=True)
        .copy()
    )

    group_id = (
        track["team"]
        .ne(track["team"].shift())
        .cumsum()
    )

    segments = []

    for _, seg in track.groupby(group_id):
        seg = seg.sort_values("frame")

        first = seg.iloc[0]
        last = seg.iloc[-1]

        segments.append(
            {
                "team": str(first["team"]),
                "start_frame": int(first["frame"]),
                "end_frame": int(last["frame"]),
                "start_time": float(first["time_sec"]),
                "end_time": float(last["time_sec"]),
                "duration_sec": float(
                    last["time_sec"] - first["time_sec"]
                ),
                "start_x": float(
                    first["pitch_x_final_m"]
                ),
                "start_y": float(
                    first["pitch_y_final_m"]
                ),
                "end_x": float(
                    last["pitch_x_final_m"]
                ),
                "end_y": float(
                    last["pitch_y_final_m"]
                ),
                "rows": int(len(seg)),
            }
        )

    return segments


def detect_switch_events(
    df: pd.DataFrame,
    args,
):
    events = []

    for track_id, track in df.groupby("track_id"):
        segments = build_team_segments(track)

        if len(segments) < 2:
            continue

        for i in range(len(segments) - 1):
            before = segments[i]
            after = segments[i + 1]

            if before["team"] == after["team"]:
                continue

            if (
                before["duration_sec"]
                < args.min_segment_sec
            ):
                continue

            if (
                after["duration_sec"]
                < args.min_segment_sec
            ):
                continue

            transition_gap = (
                after["start_time"]
                - before["end_time"]
            )

            if transition_gap <= 0:
                continue

            if (
                transition_gap
                > args.max_transition_gap_sec
            ):
                continue

            dx = (
                after["start_x"]
                - before["end_x"]
            )

            dy = (
                after["start_y"]
                - before["end_y"]
            )

            transition_distance = float(
                np.hypot(dx, dy)
            )

            transition_speed = (
                transition_distance
                / transition_gap
            )

            if (
                transition_speed
                > args.max_transition_speed_mps
            ):
                continue

            events.append(
                {
                    "track_id":
                        int(track_id),

                    "from_team":
                        before["team"],

                    "to_team":
                        after["team"],

                    "before_start_frame":
                        before["start_frame"],

                    "before_end_frame":
                        before["end_frame"],

                    "after_start_frame":
                        after["start_frame"],

                    "after_end_frame":
                        after["end_frame"],

                    "before_duration_sec":
                        before["duration_sec"],

                    "after_duration_sec":
                        after["duration_sec"],

                    "transition_time_sec":
                        after["start_time"],

                    "transition_gap_sec":
                        transition_gap,

                    "transition_distance_m":
                        transition_distance,

                    "transition_speed_mps":
                        transition_speed,
                }
            )

    return pd.DataFrame(
        events,
        columns=EVENT_COLUMNS,
    )


def closest_pair_distance(
    df: pd.DataFrame,
    track_a: int,
    track_b: int,
    transition_time_a: float,
    transition_time_b: float,
    window_sec: float,
):
    center_start = (
        min(
            transition_time_a,
            transition_time_b,
        )
        - window_sec
    )

    center_end = (
        max(
            transition_time_a,
            transition_time_b,
        )
        + window_sec
    )

    a = df[
        (df["track_id"] == track_a)
        &
        (
            df["time_sec"]
            .between(
                center_start,
                center_end,
            )
        )
    ][
        [
            "frame",
            "time_sec",
            "pitch_x_final_m",
            "pitch_y_final_m",
        ]
    ].copy()

    b = df[
        (df["track_id"] == track_b)
        &
        (
            df["time_sec"]
            .between(
                center_start,
                center_end,
            )
        )
    ][
        [
            "frame",
            "time_sec",
            "pitch_x_final_m",
            "pitch_y_final_m",
        ]
    ].copy()

    merged = a.merge(
        b,
        on="frame",
        suffixes=("_a", "_b"),
    )

    if merged.empty:
        return None

    merged["distance_m"] = np.hypot(
        (
            merged["pitch_x_final_m_a"]
            - merged["pitch_x_final_m_b"]
        ),
        (
            merged["pitch_y_final_m_a"]
            - merged["pitch_y_final_m_b"]
        ),
    )

    row = merged.loc[
        merged["distance_m"].idxmin()
    ]

    return {
        "closest_frame":
            int(row["frame"]),

        "closest_time_sec":
            float(row["time_sec_a"]),

        "minimum_distance_m":
            float(row["distance_m"]),
    }


def pair_switch_events(
    df: pd.DataFrame,
    events: pd.DataFrame,
    args,
):
    if events.empty:
        return pd.DataFrame(
            columns=PAIR_COLUMNS,
        )

    pairs = []

    for i in range(len(events)):
        a = events.iloc[i]

        for j in range(i + 1, len(events)):
            b = events.iloc[j]

            if (
                a["track_id"]
                == b["track_id"]
            ):
                continue

            reciprocal = (
                a["from_team"]
                == b["to_team"]
                and
                a["to_team"]
                == b["from_team"]
            )

            if not reciprocal:
                continue

            time_delta = abs(
                a["transition_time_sec"]
                - b["transition_time_sec"]
            )

            if (
                time_delta
                > args.pair_time_window_sec
            ):
                continue

            proximity = closest_pair_distance(
                df=df,
                track_a=int(
                    a["track_id"]
                ),
                track_b=int(
                    b["track_id"]
                ),
                transition_time_a=float(
                    a["transition_time_sec"]
                ),
                transition_time_b=float(
                    b["transition_time_sec"]
                ),
                window_sec=(
                    args.proximity_window_sec
                ),
            )

            if proximity is None:
                continue

            if (
                proximity["minimum_distance_m"]
                > args.max_pair_distance_m
            ):
                continue

            pairs.append(
                {
                    "track_id_a":
                        int(a["track_id"]),

                    "track_id_b":
                        int(b["track_id"]),

                    "transition_a":
                        (
                            f"{a['from_team']}"
                            f"->{a['to_team']}"
                        ),

                    "transition_b":
                        (
                            f"{b['from_team']}"
                            f"->{b['to_team']}"
                        ),

                    "transition_time_a_sec":
                        float(
                            a[
                                "transition_time_sec"
                            ]
                        ),

                    "transition_time_b_sec":
                        float(
                            b[
                                "transition_time_sec"
                            ]
                        ),

                    "transition_time_delta_sec":
                        float(time_delta),

                    "closest_frame":
                        proximity[
                            "closest_frame"
                        ],

                    "closest_time_sec":
                        proximity[
                            "closest_time_sec"
                        ],

                    "minimum_distance_m":
                        proximity[
                            "minimum_distance_m"
                        ],

                    "confidence":
                        "HIGH",
                }
            )

    return pd.DataFrame(
        pairs,
        columns=PAIR_COLUMNS,
    )


def main():
    args = parse_args()

    input_path = (
        Path(args.input)
        .expanduser()
        .resolve()
    )

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Tracking input not found: "
            f"{input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        input_path
    )

    missing = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    df = df[
        df["team"].isin(["A", "B"])
    ].copy()

    df = df[
        np.isfinite(
            df["pitch_x_final_m"]
        )
        &
        np.isfinite(
            df["pitch_y_final_m"]
        )
    ].copy()

    events = detect_switch_events(
        df,
        args,
    )

    pairs = pair_switch_events(
        df,
        events,
        args,
    )

    events_path = (
        output_dir
        / "identity_switch_events.csv"
    )

    pairs_path = (
        output_dir
        / "identity_switch_pairs.csv"
    )

    events.to_csv(
        events_path,
        index=False,
    )

    pairs.to_csv(
        pairs_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("IDENTITY QA")
    print("=" * 70)

    print(
        "Input:",
        input_path,
    )

    print(
        "Persistent team transitions:",
        len(events),
    )

    print(
        "High-confidence reciprocal pairs:",
        len(pairs),
    )

    if not events.empty:
        print()
        print("TRANSITIONS")

        print(
            events.to_string(
                index=False
            )
        )

    if not pairs.empty:
        print()
        print("HIGH-CONFIDENCE PAIRS")

        print(
            pairs.to_string(
                index=False
            )
        )

    print()
    print(
        "Events:",
        events_path,
    )

    print(
        "Pairs: ",
        pairs_path,
    )


if __name__ == "__main__":
    main()