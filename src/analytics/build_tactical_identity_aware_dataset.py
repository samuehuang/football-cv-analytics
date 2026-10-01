#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRACKING_CSV = Path(
    "outputs/tracking_data_clean_v2.csv"
)

METRICS_CSV = Path(
    "outputs/player_metrics.csv"
)

SWITCH_EVENTS_CSV = Path(
    "outputs/identity_switch_events.csv"
)

SWITCH_PAIRS_CSV = Path(
    "outputs/identity_switch_pairs.csv"
)

OUTPUT_CSV = Path(
    "outputs/tracking_data_tactical_identity_aware.csv"
)

FRAME_QUALITY_CSV = Path(
    "outputs/tactical_frame_quality_identity_aware.csv"
)

COVERAGE_CSV = Path(
    "outputs/tactical_coverage_comparison.csv"
)


MIN_TRACKED_TIME = 20.0


def load_high_confidence_switches():
    if not SWITCH_PAIRS_CSV.exists():
        return []

    pairs = pd.read_csv(
        SWITCH_PAIRS_CSV
    )

    if pairs.empty:
        return []

    if "confidence" not in pairs.columns:
        return []

    pairs = pairs[
        pairs["confidence"] == "HIGH"
    ].copy()

    switches = []

    for _, row in pairs.iterrows():
        switches.append(
            {
                "track_id":
                    int(row["track_id_a"]),

                "from_team":
                    str(row["transition_a"]).split("->")[0],

                "to_team":
                    str(row["transition_a"]).split("->")[1],

                "transition_time_sec":
                    float(
                        row[
                            "transition_time_a_sec"
                        ]
                    ),
            }
        )

        switches.append(
            {
                "track_id":
                    int(row["track_id_b"]),

                "from_team":
                    str(row["transition_b"]).split("->")[0],

                "to_team":
                    str(row["transition_b"]).split("->")[1],

                "transition_time_sec":
                    float(
                        row[
                            "transition_time_b_sec"
                        ]
                    ),
            }
        )

    switches = sorted(
        switches,
        key=lambda x: (
            x["track_id"],
            x["transition_time_sec"],
        ),
    )

    return switches


def assign_identity_aware_team(
    row,
    stable_team,
    switch_map,
):
    track_id = int(
        row["track_id"]
    )

    team = stable_team.get(
        track_id
    )

    if team is None:
        return None

    events = switch_map.get(
        track_id,
        [],
    )

    effective_team = team

    for event in events:
        if (
            row["time_sec"]
            >= event["transition_time_sec"]
        ):
            effective_team = (
                event["to_team"]
            )

    return effective_team


def assign_identity_segment(
    row,
    switch_map,
):
    track_id = int(
        row["track_id"]
    )

    events = switch_map.get(
        track_id,
        [],
    )

    segment = 0

    for event in events:
        if (
            row["time_sec"]
            >= event["transition_time_sec"]
        ):
            segment += 1

    return segment


def build_quality_table(
    df: pd.DataFrame,
):
    quality_rows = []

    for frame_id, frame_df in df.groupby(
        "frame"
    ):
        time_sec = float(
            frame_df[
                "time_sec"
            ].iloc[0]
        )

        a = frame_df[
            frame_df["team"] == "A"
        ]

        b = frame_df[
            frame_df["team"] == "B"
        ]

        a_ids = sorted(
            a["track_id"]
            .astype(int)
            .unique()
            .tolist()
        )

        b_ids = sorted(
            b["track_id"]
            .astype(int)
            .unique()
            .tolist()
        )

        a_count = len(
            a_ids
        )

        b_count = len(
            b_ids
        )

        complete = (
            a_count == 10
            and
            b_count == 10
        )

        quality_rows.append(
            {
                "frame":
                    int(frame_id),

                "time_sec":
                    time_sec,

                "team_A_count":
                    a_count,

                "team_B_count":
                    b_count,

                "team_A_ids":
                    ",".join(
                        map(
                            str,
                            a_ids,
                        )
                    ),

                "team_B_ids":
                    ",".join(
                        map(
                            str,
                            b_ids,
                        )
                    ),

                "complete_20_players":
                    complete,
            }
        )

    return pd.DataFrame(
        quality_rows
    )


def main():
    for path in [
        TRACKING_CSV,
        METRICS_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: "
                f"{path}"
            )

    tracking = pd.read_csv(
        TRACKING_CSV
    )

    metrics = pd.read_csv(
        METRICS_CSV
    )

    tracking = tracking[
        tracking["team"].isin(
            ["A", "B"]
        )
    ].copy()

    tracking = tracking[
        np.isfinite(
            tracking[
                "pitch_x_final_m"
            ]
        )
        &
        np.isfinite(
            tracking[
                "pitch_y_final_m"
            ]
        )
    ].copy()

    core = metrics[
        metrics[
            "tracked_time_sec"
        ]
        >= MIN_TRACKED_TIME
    ][
        [
            "track_id",
            "team",
            "tracked_time_sec",
        ]
    ].copy()

    stable_team = dict(
        zip(
            core["track_id"]
            .astype(int),

            core["team"],
        )
    )

    core_ids = set(
        stable_team.keys()
    )

    df = tracking[
        tracking[
            "track_id"
        ].isin(core_ids)
    ].copy()

    df["stable_team"] = (
        df["track_id"]
        .astype(int)
        .map(
            stable_team
        )
    )

    switches = (
        load_high_confidence_switches()
    )

    switch_map = {}

    for event in switches:
        switch_map.setdefault(
            event["track_id"],
            [],
        ).append(
            event
        )

    print()
    print("=" * 70)
    print(
        "IDENTITY-AWARE TACTICAL DATASET"
    )
    print("=" * 70)

    print(
        "High-confidence switch events:",
        len(switches),
    )

    if switches:
        print()

        for event in switches:
            print(
                f"track {event['track_id']}: "
                f"{event['from_team']}"
                f"->{event['to_team']} "
                f"@ "
                f"{event['transition_time_sec']:.2f}s"
            )

    # ========================================================
    # STRICT BASELINE
    # ========================================================

    strict = df[
        df["team"]
        ==
        df["stable_team"]
    ].copy()

    strict_quality = (
        build_quality_table(
            strict
        )
    )

    strict_valid_frames = set(
        strict_quality[
            strict_quality[
                "complete_20_players"
            ]
        ]["frame"]
    )

    # ========================================================
    # IDENTITY-AWARE TEAM ASSIGNMENT
    # ========================================================

    df[
        "identity_aware_team"
    ] = df.apply(
        lambda row:
            assign_identity_aware_team(
                row,
                stable_team,
                switch_map,
            ),
        axis=1,
    )

    df[
        "identity_segment"
    ] = df.apply(
        lambda row:
            assign_identity_segment(
                row,
                switch_map,
            ),
        axis=1,
    )

    # Keep only observations whose observed team agrees
    # with the identity-aware expected team.
    #
    # This remains conservative:
    # random classifier disagreement is still discarded.
    identity_aware = df[
        df["team"]
        ==
        df[
            "identity_aware_team"
        ]
    ].copy()

    identity_aware[
        "team"
    ] = identity_aware[
        "identity_aware_team"
    ]

    identity_aware[
        "analytics_identity"
    ] = (
        identity_aware[
            "track_id"
        ]
        .astype(int)
        .astype(str)
        +
        "_s"
        +
        identity_aware[
            "identity_segment"
        ]
        .astype(int)
        .astype(str)
    )

    quality = (
        build_quality_table(
            identity_aware
        )
    )

    valid_frames = set(
        quality[
            quality[
                "complete_20_players"
            ]
        ]["frame"]
    )

    tactical = identity_aware[
        identity_aware[
            "frame"
        ].isin(
            valid_frames
        )
    ].copy()

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tactical.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    quality.to_csv(
        FRAME_QUALITY_CSV,
        index=False,
    )

    total_frames = (
        tracking[
            "frame"
        ]
        .nunique()
    )

    strict_frames = len(
        strict_valid_frames
    )

    identity_frames = len(
        valid_frames
    )

    recovered_frames = len(
        valid_frames
        -
        strict_valid_frames
    )

    strict_coverage = (
        strict_frames
        /
        total_frames
        *
        100.0
    )

    identity_coverage = (
        identity_frames
        /
        total_frames
        *
        100.0
    )

    comparison = pd.DataFrame(
        [
            {
                "mode":
                    "strict",

                "complete_frames":
                    strict_frames,

                "total_frames":
                    total_frames,

                "coverage_pct":
                    strict_coverage,
            },
            {
                "mode":
                    "identity_aware",

                "complete_frames":
                    identity_frames,

                "total_frames":
                    total_frames,

                "coverage_pct":
                    identity_coverage,
            },
        ]
    )

    comparison.to_csv(
        COVERAGE_CSV,
        index=False,
    )

    print()
    print("=" * 70)
    print(
        "TACTICAL COVERAGE COMPARISON"
    )
    print("=" * 70)

    print(
        f"Total frames:             "
        f"{total_frames}"
    )

    print(
        f"Strict complete frames:   "
        f"{strict_frames} "
        f"({strict_coverage:.1f}%)"
    )

    print(
        f"Identity-aware frames:    "
        f"{identity_frames} "
        f"({identity_coverage:.1f}%)"
    )

    print(
        f"Recovered frames:         "
        f"{recovered_frames}"
    )

    print()
    print(
        f"Rows:                     "
        f"{len(tactical)}"
    )

    print(
        f"Expected rows:            "
        f"{identity_frames * 20}"
    )

    if identity_frames > 0:
        print(
            f"First valid time:         "
            f"{tactical['time_sec'].min():.2f}s"
        )

        print(
            f"Last valid time:          "
            f"{tactical['time_sec'].max():.2f}s"
        )

    print()
    print(
        "Dataset:",
        OUTPUT_CSV,
    )

    print(
        "Frame quality:",
        FRAME_QUALITY_CSV,
    )

    print(
        "Coverage:",
        COVERAGE_CSV,
    )


if __name__ == "__main__":
    main()
