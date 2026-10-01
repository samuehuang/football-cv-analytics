import argparse
import os

import numpy as np
import pandas as pd


VALID_TEAMS = {"A", "B"}
VALID_POSSESSION = {"A", "B", "Unknown", "Contested"}


def clean_text(v, default=None):
    if pd.isna(v):
        return default

    s = str(v).strip()

    if not s or s.lower() in {"nan", "none"}:
        return default

    return s


def clean_team(v):
    s = clean_text(v)

    return (
        s
        if s in VALID_TEAMS
        else None
    )


def clean_id(v):
    if pd.isna(v):
        return np.nan

    try:
        return float(v)

    except Exception:
        return np.nan


def clean_possession(v):
    s = clean_text(
        v,
        "Unknown",
    )

    return (
        s
        if s in VALID_POSSESSION
        else "Unknown"
    )


def valid_controller(
    team,
    track_id,
):
    return (
        clean_team(team)
        is not None
        and
        pd.notna(
            clean_id(track_id)
        )
    )


def estimate_fps(df):
    if (
        "time_sec"
        not in
        df.columns
    ):
        return 25.0

    timing = (
        df[
            [
                "frame",
                "time_sec",
            ]
        ]
        .dropna()
        .sort_values(
            "frame"
        )
    )

    if len(timing) < 2:
        return 25.0

    frame_diff = (
        timing[
            "frame"
        ]
        .diff()
    )

    time_diff = (
        timing[
            "time_sec"
        ]
        .diff()
    )

    valid = (
        frame_diff.notna()
        &
        time_diff.notna()
        &
        (
            frame_diff > 0
        )
        &
        (
            time_diff > 0
        )
    )

    values = (
        frame_diff[valid]
        /
        time_diff[valid]
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).dropna()

    if len(values) == 0:
        return 25.0

    return float(
        values.median()
    )


def build_possession_episodes(
    df,
    fps,
):
    rows = []

    start = None
    prev = None
    current = None
    episode_id = 0


    def close_episode(
        end_frame,
    ):
        nonlocal episode_id

        if (
            start is None
            or
            end_frame is None
        ):
            return

        episode_id += 1

        rows.append(
            {
                "episode_id":
                    episode_id,

                "start_frame":
                    int(start),

                "end_frame":
                    int(end_frame),

                "duration_frames":
                    int(
                        end_frame
                        -
                        start
                        +
                        1
                    ),

                "duration_sec":
                    float(
                        end_frame
                        -
                        start
                        +
                        1
                    )
                    /
                    fps,

                "possession":
                    current,
            }
        )


    for _, row in (
        df
        .sort_values(
            "frame"
        )
        .iterrows()
    ):

        frame = int(
            row[
                "frame"
            ]
        )

        possession = (
            clean_possession(
                row[
                    "possession_v6"
                ]
            )
        )


        if start is None:

            start = frame

            prev = frame

            current = (
                possession
            )

            continue


        if (
            frame
            ==
            prev + 1

            and

            possession
            ==
            current
        ):

            prev = frame

            continue


        close_episode(
            prev
        )


        start = frame

        prev = frame

        current = (
            possession
        )


    close_episode(
        prev
    )


    return pd.DataFrame(
        rows
    )


def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Possession V6 built on "
                "Ball Motion State V1."
            )
        )
    )


    parser.add_argument(
        "--motion",
        default=(
            "outputs/"
            "ball_motion_state_v1_frames.csv"
        ),
    )


    parser.add_argument(
        "--output-dir",
        default="outputs",
    )


    args = (
        parser.parse_args()
    )


    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )


    frames_out = (
        os.path.join(
            args.output_dir,
            "possession_v6_frames.csv",
        )
    )


    episodes_out = (
        os.path.join(
            args.output_dir,
            "possession_v6_episodes.csv",
        )
    )


    print(
        "Loading Ball Motion State V1..."
    )


    df = pd.read_csv(
        args.motion
    )


    required = [
        "frame",

        "motion_state_v1",

        "possession_motion_v1",

        "controller_team_motion_v1",

        "controller_id_motion_v1",

        "last_touch_team_motion_v1",

        "last_touch_id_motion_v1",

        "possession_v5_3",

        "ball_state_v5_3",

        "controller_team_v5_3",

        "controller_id_v5_3",

        "last_touch_team_v5_3",

        "last_touch_id_v5_3",
    ]


    missing = [
        column
        for column
        in required
        if column
        not in
        df.columns
    ]


    if missing:

        raise ValueError(
            "Missing required columns: "
            +
            ", ".join(
                missing
            )
        )


    for column in [
        "frame",

        "time_sec",

        "controller_id_motion_v1",

        "last_touch_id_motion_v1",

        "controller_id_v5_3",

        "last_touch_id_v5_3",
    ]:

        if (
            column
            in
            df.columns
        ):

            df[
                column
            ] = pd.to_numeric(
                df[
                    column
                ],
                errors="coerce",
            )


    df = (
        df
        .dropna(
            subset=[
                "frame"
            ]
        )
        .copy()
        .sort_values(
            "frame"
        )
        .reset_index(
            drop=True
        )
    )


    df[
        "frame"
    ] = (
        df[
            "frame"
        ]
        .astype(int)
    )


    fps = estimate_fps(
        df
    )


    print(
        f"Estimated FPS: "
        f"{fps:.2f}"
    )


    print(
        "Applying Possession V6 rules..."
    )


    output_rows = []


    for _, row in (
        df.iterrows()
    ):

        out = (
            row.to_dict()
        )


        motion_state = (
            clean_text(
                row.get(
                    "motion_state_v1"
                ),
                "GroundOrControlled",
            )
        )


        motion_possession = (
            clean_possession(
                row.get(
                    "possession_motion_v1"
                )
            )
        )


        motion_controller_team = (
            clean_team(
                row.get(
                    "controller_team_motion_v1"
                )
            )
        )


        motion_controller_id = (
            clean_id(
                row.get(
                    "controller_id_motion_v1"
                )
            )
        )


        motion_last_touch_team = (
            clean_team(
                row.get(
                    "last_touch_team_motion_v1"
                )
            )
        )


        motion_last_touch_id = (
            clean_id(
                row.get(
                    "last_touch_id_motion_v1"
                )
            )
        )


        baseline_possession = (
            clean_possession(
                row.get(
                    "possession_v5_3"
                )
            )
        )


        baseline_state = (
            clean_text(
                row.get(
                    "ball_state_v5_3"
                ),
                "Unknown",
            )
        )


        baseline_last_touch_team = (
            clean_team(
                row.get(
                    "last_touch_team_v5_3"
                )
            )
        )


        baseline_last_touch_id = (
            clean_id(
                row.get(
                    "last_touch_id_v5_3"
                )
            )
        )


        possession = (
            "Unknown"
        )


        ball_state = (
            "Unknown"
        )


        controller_team = (
            None
        )


        controller_id = (
            np.nan
        )


        last_touch_team = (
            motion_last_touch_team
        )


        last_touch_id = (
            motion_last_touch_id
        )


        decision_reason = (
            "fallback_unknown"
        )


        confidence = (
            "conservative"
        )


        # ====================================================
        # DRIBBLE GAP
        # ====================================================

        if (
            motion_state
            ==
            "DribbleGap"
        ):

            controller_team = (
                motion_controller_team
            )


            controller_id = (
                motion_controller_id
            )


            if valid_controller(
                controller_team,
                controller_id,
            ):

                possession = (
                    controller_team
                )


                last_touch_team = (
                    controller_team
                )


                last_touch_id = (
                    controller_id
                )


                confidence = (
                    "high"
                )


            else:

                possession = (

                    motion_possession

                    if
                    motion_possession
                    in
                    VALID_TEAMS

                    else

                    "Unknown"
                )


                confidence = (
                    "medium"
                )


            ball_state = (
                "DribbleGap"
            )


            decision_reason = (
                "temporal_dribble_continuity"
            )


        # ====================================================
        # CONFIRMED PASS IN TRANSIT
        # ====================================================

        elif (
            motion_state
            ==
            "ConfirmedPassInTransit"
        ):

            possession = (

                motion_possession

                if
                motion_possession
                in
                VALID_TEAMS

                else

                "Unknown"
            )


            ball_state = (
                "InTransit"
            )


            decision_reason = (
                "retrospective_confirmed_pass"
            )


            confidence = (
                "high"
            )


        # ====================================================
        # AERIAL CONTEST
        # ====================================================

        elif (
            motion_state
            ==
            "AerialContest"
        ):

            possession = (
                "Contested"
            )


            ball_state = (
                "AerialContest"
            )


            decision_reason = (
                "physical_aerial_contest"
            )


            confidence = (
                "high"
            )


        # ====================================================
        # AIR TOUCH
        # ====================================================

        elif (
            motion_state
            ==
            "AirTouchInTransit"
        ):

            possession = (
                "Unknown"
            )


            ball_state = (
                "AirTouchInTransit"
            )


            decision_reason = (
                "physical_air_touch_without_control"
            )


            confidence = (
                "high"
            )


        # ====================================================
        # TRANSIT MISSING
        # ====================================================

        elif (
            motion_state
            ==
            "TransitMissing"
        ):

            possession = (
                "Unknown"
            )


            ball_state = (
                "Missing"
            )


            decision_reason = (
                "transit_ball_missing"
            )


        # ====================================================
        # TRANSIT UNCONTROLLED
        # ====================================================

        elif (
            motion_state
            ==
            "TransitUncontrolled"
        ):

            possession = (
                "Unknown"
            )


            ball_state = (
                "InTransit"
            )


            decision_reason = (
                "uncontrolled_transit"
            )


        # ====================================================
        # GROUND / CONTROLLED
        # ====================================================

        else:

            if valid_controller(
                motion_controller_team,
                motion_controller_id,
            ):

                possession = (
                    motion_controller_team
                )


                ball_state = (
                    "Controlled"
                )


                controller_team = (
                    motion_controller_team
                )


                controller_id = (
                    motion_controller_id
                )


                last_touch_team = (
                    motion_controller_team
                )


                last_touch_id = (
                    motion_controller_id
                )


                decision_reason = (
                    "trusted_controller"
                )


                confidence = (
                    "high"
                )


            else:

                controller_team = (
                    None
                )


                controller_id = (
                    np.nan
                )


                if (
                    baseline_state
                    ==
                    "Missing"
                ):

                    possession = (
                        "Unknown"
                    )


                    ball_state = (
                        "Missing"
                    )


                    decision_reason = (
                        "baseline_missing"
                    )


                elif (
                    baseline_state
                    ==
                    "AerialContest"
                ):

                    possession = (
                        "Contested"
                    )


                    ball_state = (
                        "AerialContest"
                    )


                    decision_reason = (
                        "baseline_contest_no_controller"
                    )


                    confidence = (
                        "medium"
                    )


                elif (
                    baseline_state
                    ==
                    "Controlled"
                ):

                    possession = (
                        "Unknown"
                    )


                    ball_state = (
                        "Uncontrolled"
                    )


                    decision_reason = (
                        "controlled_without_trusted_controller"
                    )


                else:

                    possession = (
                        baseline_possession
                    )


                    ball_state = (
                        baseline_state
                    )


                    decision_reason = (
                        "validated_ground_baseline"
                    )


                    confidence = (
                        "medium"
                    )


                    if (
                        last_touch_team
                        is None
                    ):

                        last_touch_team = (
                            baseline_last_touch_team
                        )


                        last_touch_id = (
                            baseline_last_touch_id
                        )


        # ====================================================
        # HARD INVARIANTS
        # ====================================================

        if valid_controller(
            controller_team,
            controller_id,
        ):

            possession = (
                controller_team
            )


        if (
            motion_state
            in
            {
                "ConfirmedPassInTransit",

                "AerialContest",

                "AirTouchInTransit",

                "TransitMissing",

                "TransitUncontrolled",
            }
        ):

            controller_team = (
                None
            )


            controller_id = (
                np.nan
            )


        if (
            motion_state
            ==
            "AerialContest"
        ):

            possession = (
                "Contested"
            )


        if (
            motion_state
            in
            {
                "AirTouchInTransit",

                "TransitMissing",

                "TransitUncontrolled",
            }
        ):

            possession = (
                "Unknown"
            )


        # ====================================================
        # OUTPUT
        # ====================================================

        out[
            "possession_v6"
        ] = (
            possession
        )


        out[
            "ball_state_v6"
        ] = (
            ball_state
        )


        out[
            "controller_team_v6"
        ] = (
            controller_team
        )


        out[
            "controller_id_v6"
        ] = (
            controller_id
        )


        out[
            "last_touch_team_v6"
        ] = (
            last_touch_team
        )


        out[
            "last_touch_id_v6"
        ] = (
            last_touch_id
        )


        out[
            "decision_reason_v6"
        ] = (
            decision_reason
        )


        out[
            "confidence_v6"
        ] = (
            confidence
        )


        output_rows.append(
            out
        )


    result = pd.DataFrame(
        output_rows
    )


    # ========================================================
    # INVARIANT CHECKS
    # ========================================================

    controller_mask = (

        result[
            "controller_team_v6"
        ].isin(
            [
                "A",
                "B",
            ]
        )

        &

        result[
            "controller_id_v6"
        ].notna()
    )


    controller_mismatch = (

        controller_mask

        &

        (
            result[
                "controller_team_v6"
            ]

            !=

            result[
                "possession_v6"
            ]
        )
    )


    illegal_controller = (

        result[
            "motion_state_v1"
        ].isin(
            {
                "ConfirmedPassInTransit",

                "AerialContest",

                "AirTouchInTransit",

                "TransitMissing",

                "TransitUncontrolled",
            }
        )

        &

        controller_mask
    )


    contest_mismatch = (

        (
            result[
                "motion_state_v1"
            ]
            ==
            "AerialContest"
        )

        &

        (
            result[
                "possession_v6"
            ]
            !=
            "Contested"
        )
    )


    generic_transit_mismatch = (

        result[
            "motion_state_v1"
        ].isin(
            {
                "AirTouchInTransit",

                "TransitMissing",

                "TransitUncontrolled",
            }
        )

        &

        (
            result[
                "possession_v6"
            ]
            !=
            "Unknown"
        )
    )


    checks = {

        "controller_possession_mismatch":
            int(
                controller_mismatch.sum()
            ),

        "controller_inside_uncontrolled_transit":
            int(
                illegal_controller.sum()
            ),

        "aerial_contest_not_contested":
            int(
                contest_mismatch.sum()
            ),

        "generic_transit_not_unknown":
            int(
                generic_transit_mismatch.sum()
            ),
    }


    # ========================================================
    # SAVE
    # ========================================================

    result.to_csv(
        frames_out,
        index=False,
    )


    possession_episodes = (
        build_possession_episodes(
            result,
            fps,
        )
    )


    possession_episodes.to_csv(
        episodes_out,
        index=False,
    )


    # ========================================================
    # COMPARE WITH V5.3
    # ========================================================

    changed_possession = (

        result[
            "possession_v6"
        ].astype(str)

        !=

        result[
            "possession_v5_3"
        ].astype(str)
    )


    changed_controller = (

        (
            result[
                "controller_team_v6"
            ]
            .fillna(
                "NONE"
            )
            .astype(str)

            !=

            result[
                "controller_team_v5_3"
            ]
            .fillna(
                "NONE"
            )
            .astype(str)
        )

        |

        (
            result[
                "controller_id_v6"
            ]
            .fillna(
                -1
            )
            .astype(float)

            !=

            result[
                "controller_id_v5_3"
            ]
            .fillna(
                -1
            )
            .astype(float)
        )
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("")

    print(
        "="
        *
        54
    )


    print(
        "POSSESSION V6 SUMMARY"
    )


    print(
        "="
        *
        54
    )


    print(
        f"Frames: "
        f"{len(result)}"
    )


    print(
        "Possession changes vs V5.3:",
        int(
            changed_possession.sum()
        ),
    )


    print(
        "Controller changes vs V5.3:",
        int(
            changed_controller.sum()
        ),
    )


    print("")

    print(
        "POSSESSION"
    )


    print(
        result[
            "possession_v6"
        ]
        .value_counts()
        .to_string()
    )


    print("")

    print(
        "BALL STATES"
    )


    print(
        result[
            "ball_state_v6"
        ]
        .value_counts()
        .to_string()
    )


    print("")

    print(
        "DECISION REASONS"
    )


    print(
        result[
            "decision_reason_v6"
        ]
        .value_counts()
        .to_string()
    )


    print("")

    print(
        "CONFIDENCE"
    )


    print(
        result[
            "confidence_v6"
        ]
        .value_counts()
        .to_string()
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

            if
            count == 0

            else

            "FAIL"
        )


        print(
            f"{status} | "
            f"{name}: "
            f"{count}"
        )


    print("")

    print(
        "Possession episodes:",
        len(
            possession_episodes
        ),
    )


    print("")

    print(
        "Outputs:"
    )


    print(
        "Frames:",
        frames_out
    )


    print(
        "Episodes:",
        episodes_out
    )


if __name__ == "__main__":

    main()
