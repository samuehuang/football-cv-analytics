import argparse
import json
import os

import numpy as np
import pandas as pd


def clean_text(v, default=None):
    if pd.isna(v):
        return default
    s = str(v).strip()
    return default if (not s or s.lower() in {"nan", "none"}) else s


def clean_team(v):
    s = clean_text(v)
    return s if s in {"A", "B"} else None


def clean_id(v):
    if pd.isna(v):
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except Exception:
        return None


def estimate_fps(df):
    if "time_sec" not in df.columns:
        return 25.0

    x = df[["frame", "time_sec"]].dropna().sort_values("frame")

    if len(x) < 2:
        return 25.0

    fd = x["frame"].diff()
    td = x["time_sec"].diff()

    valid = (
        fd.notna()
        & td.notna()
        & (fd > 0)
        & (td > 0)
    )

    vals = (
        fd[valid] / td[valid]
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    return float(vals.median()) if len(vals) else 25.0


def json_safe(v):
    if isinstance(v, np.integer):
        return int(v)

    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)

    if pd.isna(v):
        return None

    return v


def main():

    parser = argparse.ArgumentParser(
        description="Football Event Engine V1"
    )

    parser.add_argument(
        "--frames",
        default="outputs/possession_v6_frames.csv",
    )

    parser.add_argument(
        "--possession-episodes",
        default="outputs/possession_v6_episodes.csv",
    )

    parser.add_argument(
        "--motion-episodes",
        default="outputs/ball_motion_state_v1_episodes.csv",
    )

    parser.add_argument(
        "--air-events",
        default="outputs/air_touch_v4_events.csv",
    )

    parser.add_argument(
        "--dribble-gaps",
        default="outputs/ball_motion_state_v1_dribble_gaps.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )


    print("Loading Possession V6...")
    frames = pd.read_csv(
        args.frames
    )

    print("Loading possession episodes...")
    possession_episodes = pd.read_csv(
        args.possession_episodes
    )

    print("Loading motion episodes...")
    motion_episodes = pd.read_csv(
        args.motion_episodes
    )

    print("Loading physical air events...")
    air_events = pd.read_csv(
        args.air_events
    )

    print("Loading dribble gaps...")
    dribble_gaps = pd.read_csv(
        args.dribble_gaps
    )


    frames["frame"] = pd.to_numeric(
        frames["frame"],
        errors="coerce",
    )

    frames = (
        frames
        .dropna(subset=["frame"])
        .copy()
        .sort_values("frame")
        .reset_index(drop=True)
    )

    frames["frame"] = (
        frames["frame"]
        .astype(int)
    )


    fps = estimate_fps(
        frames
    )

    print(
        f"Estimated FPS: {fps:.2f}"
    )


    event_rows = []


    def add_event(
        event_type,
        subtype,
        start_frame,
        end_frame,
        team=None,
        player_id=None,
        target_team=None,
        target_player_id=None,
        success=None,
        confidence="high",
        source=None,
        details=None,
    ):

        start_frame = int(
            start_frame
        )

        end_frame = int(
            end_frame
        )

        event_rows.append(
            {
                "event_type":
                    event_type,

                "subtype":
                    subtype,

                "start_frame":
                    start_frame,

                "end_frame":
                    end_frame,

                "start_time_sec":
                    start_frame / fps,

                "end_time_sec":
                    end_frame / fps,

                "duration_sec":
                    (
                        end_frame
                        -
                        start_frame
                        +
                        1
                    )
                    /
                    fps,

                "team":
                    clean_team(team),

                "player_id":
                    clean_id(player_id),

                "target_team":
                    clean_team(target_team),

                "target_player_id":
                    clean_id(
                        target_player_id
                    ),

                "success":
                    success,

                "confidence":
                    confidence,

                "source":
                    source,

                "details":
                    details,
            }
        )


    # ========================================================
    # POSSESSION EPISODES
    # ========================================================

    for _, row in (
        possession_episodes
        .iterrows()
    ):

        possession = clean_text(
            row.get(
                "possession"
            ),
            "Unknown",
        )

        add_event(
            event_type="PossessionEpisode",

            subtype=possession,

            start_frame=row[
                "start_frame"
            ],

            end_frame=row[
                "end_frame"
            ],

            team=(
                possession
                if possession in {"A", "B"}
                else None
            ),

            confidence=(
                "high"
                if possession in {"A", "B"}
                else "conservative"
            ),

            source="possession_v6",

            details=(
                f"possession={possession}"
            ),
        )


    # ========================================================
    # PASS / RECEPTION / TURNOVER
    # ========================================================

    for _, row in (
        motion_episodes
        .iterrows()
    ):

        event_class = clean_text(
            row.get(
                "event_class"
            ),
            "",
        )

        launch_team = clean_team(
            row.get(
                "launch_team"
            )
        )

        launch_id = clean_id(
            row.get(
                "launch_track_id"
            )
        )

        receiver_team = clean_team(
            row.get(
                "receiver_team"
            )
        )

        receiver_id = clean_id(
            row.get(
                "receiver_track_id"
            )
        )

        start_frame = int(
            row[
                "start_frame"
            ]
        )

        end_frame = int(
            row[
                "end_frame"
            ]
        )

        resolution = clean_text(
            row.get(
                "resolution_type"
            ),
            "",
        )


        if (
            event_class
            ==
            "ConfirmedPass"
        ):

            add_event(
                event_type="Pass",

                subtype="ConfirmedPass",

                start_frame=start_frame,

                end_frame=end_frame,

                team=launch_team,

                player_id=launch_id,

                target_team=receiver_team,

                target_player_id=receiver_id,

                success=True,

                confidence="high",

                source="ball_motion_state_v1",

                details=(
                    "same-team trusted "
                    "launch-to-reception "
                    "without AirTouch/"
                    "AerialContest"
                ),
            )


        if (
            resolution
            ==
            "trusted_reception"

            and

            receiver_team
            is not None

            and

            receiver_id
            is not None
        ):

            reception_frame = (
                end_frame + 1
            )

            add_event(
                event_type="Reception",

                subtype="TrustedReception",

                start_frame=reception_frame,

                end_frame=reception_frame,

                team=receiver_team,

                player_id=receiver_id,

                success=True,

                confidence="high",

                source=(
                    "controller_gate_v2+"
                    "ball_motion_state_v1"
                ),

                details=(
                    "resolved motion episode "
                    f"{int(row['episode_id'])}"
                ),
            )


        if (
            event_class
            ==
            "TurnoverCandidate"
        ):

            reception_frame = (
                end_frame + 1
            )

            add_event(
                event_type="Turnover",

                subtype=(
                    "ControlledTurnoverCandidate"
                ),

                start_frame=reception_frame,

                end_frame=reception_frame,

                team=receiver_team,

                player_id=receiver_id,

                target_team=launch_team,

                target_player_id=launch_id,

                success=True,

                confidence="medium",

                source="ball_motion_state_v1",

                details=(
                    "trusted reception by "
                    "opposite team after "
                    "known launch"
                ),
            )


    # ========================================================
    # AERIAL EVENTS
    # ========================================================

    for _, row in (
        air_events
        .iterrows()
    ):

        final_class = clean_text(
            row.get(
                "final_class"
            ),
            "",
        )

        start_frame = int(
            row[
                "start_frame"
            ]
        )

        end_frame = int(
            row[
                "end_frame"
            ]
        )

        participants = clean_text(
            row.get(
                "participants"
            ),
            "",
        )


        if (
            final_class
            ==
            "AerialContest"
        ):

            add_event(
                event_type="AerialDuel",

                subtype="AerialContest",

                start_frame=start_frame,

                end_frame=end_frame,

                confidence="high",

                source="air_touch_v4",

                details=(
                    f"participants="
                    f"{participants}"
                ),
            )


        elif (
            final_class
            ==
            "AirTouch"
        ):

            add_event(
                event_type="AirTouch",

                subtype="ConfirmedAirTouch",

                start_frame=start_frame,

                end_frame=end_frame,

                team=row.get(
                    "last_touch_team"
                ),

                player_id=row.get(
                    "last_touch_track_id"
                ),

                success=True,

                confidence="high",

                source="air_touch_v4",

                details=(
                    f"participants="
                    f"{participants}"
                ),
            )


    # ========================================================
    # DRIBBLE GAP
    # ========================================================

    for _, row in (
        dribble_gaps
        .iterrows()
    ):

        add_event(
            event_type="Dribble",

            subtype="DribbleGap",

            start_frame=row[
                "start_frame"
            ],

            end_frame=row[
                "end_frame"
            ],

            team=row.get(
                "team"
            ),

            player_id=row.get(
                "track_id"
            ),

            success=True,

            confidence="high",

            source="ball_motion_state_v1",

            details=(
                "same-player return "
                f"at frame "
                f"{int(row['reception_frame'])}"
            ),
        )


    # ========================================================
    # SORT EVENTS
    # ========================================================

    events = pd.DataFrame(
        event_rows
    )


    priority = {
        "Pass": 0,
        "Reception": 1,
        "Turnover": 2,
        "AerialDuel": 3,
        "AirTouch": 4,
        "Dribble": 5,
        "PossessionEpisode": 6,
    }


    events[
        "_priority"
    ] = (
        events[
            "event_type"
        ]
        .map(priority)
        .fillna(99)
    )


    events = (
        events
        .sort_values(
            [
                "start_frame",
                "_priority",
                "end_frame",
            ]
        )
        .drop(
            columns=[
                "_priority"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    events.insert(
        0,
        "event_id",
        np.arange(
            1,
            len(events) + 1,
        ),
    )


    # ========================================================
    # INVARIANT CHECKS
    # ========================================================

    passes = events[
        events[
            "event_type"
        ]
        ==
        "Pass"
    ]


    checks = {

        "pass_team_mismatch":
            int(
                (
                    passes[
                        "team"
                    ]
                    !=
                    passes[
                        "target_team"
                    ]
                )
                .sum()
            ),

        "pass_same_player":
            int(
                (
                    passes[
                        "player_id"
                    ]
                    ==
                    passes[
                        "target_player_id"
                    ]
                )
                .sum()
            ),

        "reception_missing_player":
            int(
                (
                    (
                        events[
                            "event_type"
                        ]
                        ==
                        "Reception"
                    )

                    &

                    (
                        events[
                            "team"
                        ].isna()

                        |

                        events[
                            "player_id"
                        ].isna()
                    )
                )
                .sum()
            ),

        "turnover_same_team":
            int(
                (
                    (
                        events[
                            "event_type"
                        ]
                        ==
                        "Turnover"
                    )

                    &

                    (
                        events[
                            "team"
                        ]
                        ==
                        events[
                            "target_team"
                        ]
                    )
                )
                .sum()
            ),
    }


    # ========================================================
    # MATCH SUMMARY
    # ========================================================

    total_frames = len(
        frames
    )


    duration_sec = (
        total_frames
        /
        fps
    )


    counts = (
        frames[
            "possession_v6"
        ]
        .value_counts()
        .to_dict()
    )


    seconds = {
        key:
            value / fps

        for key, value
        in counts.items()
    }


    percentages = {
        key:
            (
                100.0
                *
                value
                /
                total_frames
            )

        for key, value
        in counts.items()
    }


    known_frames = (
        counts.get(
            "A",
            0,
        )
        +
        counts.get(
            "B",
            0,
        )
    )


    known_share = {

        "A":
            (
                100.0
                *
                counts.get(
                    "A",
                    0,
                )
                /
                known_frames

                if known_frames
                else 0.0
            ),

        "B":
            (
                100.0
                *
                counts.get(
                    "B",
                    0,
                )
                /
                known_frames

                if known_frames
                else 0.0
            ),
    }


    summary = {

        "pipeline_version":
            "football_event_engine_v1",

        "fps":
            fps,

        "total_frames":
            total_frames,

        "duration_sec":
            duration_sec,

        "event_counts":
            {
                str(k):
                    int(v)

                for k, v
                in
                events[
                    "event_type"
                ]
                .value_counts()
                .items()
            },

        "possession_seconds":
            seconds,

        "possession_percent_all_frames":
            percentages,

        "known_team_possession_share_percent":
            known_share,

        "qa_checks":
            checks,
    }


    # ========================================================
    # SAVE
    # ========================================================

    events_csv = os.path.join(
        args.output_dir,
        "football_events_v1.csv",
    )

    events_json = os.path.join(
        args.output_dir,
        "football_events_v1.json",
    )

    summary_json = os.path.join(
        args.output_dir,
        "match_summary_v1.json",
    )


    events.to_csv(
        events_csv,
        index=False,
    )


    with open(
        events_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            [
                {
                    k:
                        json_safe(v)

                    for k, v
                    in row.items()
                }

                for row
                in events.to_dict(
                    "records"
                )
            ],

            f,

            ensure_ascii=False,

            indent=2,
        )


    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # PRINT
    # ========================================================

    print("")

    print(
        "=" * 56
    )

    print(
        "FOOTBALL EVENT ENGINE V1 SUMMARY"
    )

    print(
        "=" * 56
    )


    print(
        f"Frames: "
        f"{total_frames}"
    )

    print(
        f"Duration: "
        f"{duration_sec:.2f}s"
    )

    print(
        f"Events: "
        f"{len(events)}"
    )


    print("")

    print(
        "EVENT COUNTS"
    )


    print(
        events[
            "event_type"
        ]
        .value_counts()
        .to_string()
    )


    print("")

    print(
        "HIGH-CONFIDENCE FOOTBALL EVENTS"
    )


    key_events = events[
        events[
            "event_type"
        ]
        .isin(
            [
                "Pass",
                "Reception",
                "Turnover",
                "AerialDuel",
                "AirTouch",
                "Dribble",
            ]
        )
    ]


    columns = [
        "event_id",
        "event_type",
        "subtype",
        "start_frame",
        "end_frame",
        "team",
        "player_id",
        "target_team",
        "target_player_id",
        "confidence",
    ]


    if len(
        key_events
    ):

        print(
            key_events[
                columns
            ]
            .to_string(
                index=False
            )
        )

    else:

        print(
            "None"
        )


    print("")

    print(
        "POSSESSION"
    )


    for team in [
        "A",
        "B",
        "Unknown",
        "Contested",
    ]:

        print(
            f"{team:10s} "
            f"{counts.get(team, 0):4d} frames | "
            f"{seconds.get(team, 0):6.2f}s | "
            f"{percentages.get(team, 0):6.2f}%"
        )


    print("")

    print(
        "Known-team share: "
        f"A {known_share['A']:.2f}% / "
        f"B {known_share['B']:.2f}%"
    )


    print("")

    print(
        "INVARIANT CHECKS"
    )


    for name, count in (
        checks.items()
    ):

        status = (
            "PASS"
            if count == 0
            else "FAIL"
        )

        print(
            f"{status} | "
            f"{name}: "
            f"{count}"
        )


    print("")

    print(
        "Outputs:"
    )

    print(
        "CSV:",
        events_csv
    )

    print(
        "JSON:",
        events_json
    )

    print(
        "Summary:",
        summary_json
    )


if __name__ == "__main__":
    main()
