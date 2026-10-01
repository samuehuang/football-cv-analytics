import argparse
import os

import numpy as np
import pandas as pd


def clean_team(v):
    if pd.isna(v):
        return None

    v = str(v).strip()

    return (
        v
        if v in {"A", "B"}
        else None
    )


def clean_id(v):
    if pd.isna(v):
        return np.nan

    try:
        return float(v)

    except Exception:
        return np.nan


def clean_text(
    v,
    default=None,
):
    if pd.isna(v):
        return default

    s = str(v).strip()

    if (
        s == ""
        or
        s.lower()
        in {
            "none",
            "nan",
        }
    ):
        return default

    return s


def same_player(
    team_a,
    id_a,
    team_b,
    id_b,
):
    team_a = clean_team(
        team_a
    )

    team_b = clean_team(
        team_b
    )

    if (
        team_a is None
        or
        team_b is None
        or
        team_a != team_b
        or
        pd.isna(id_a)
        or
        pd.isna(id_b)
    ):
        return False

    return (
        abs(
            float(id_a)
            -
            float(id_b)
        )
        <
        0.1
    )


def estimate_fps(
    df,
):
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
        .sort_values("frame")
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


def group_keep_runs(
    gate_keep,
):
    runs = []

    current = []

    prev_frame = None
    prev_team = None
    prev_id = np.nan

    for _, row in (
        gate_keep
        .sort_values("frame")
        .iterrows()
    ):
        frame = int(
            row["frame"]
        )

        team = clean_team(
            row[
                "controller_team"
            ]
        )

        track_id = clean_id(
            row[
                "controller_id"
            ]
        )

        if not current:
            current = [
                row.to_dict()
            ]

        else:
            consecutive = (
                frame
                ==
                prev_frame + 1
            )

            same = same_player(
                team,
                track_id,
                prev_team,
                prev_id,
            )

            if (
                consecutive
                and
                same
            ):
                current.append(
                    row.to_dict()
                )

            else:
                runs.append(
                    pd.DataFrame(
                        current
                    )
                )

                current = [
                    row.to_dict()
                ]

        prev_frame = frame
        prev_team = team
        prev_id = track_id

    if current:
        runs.append(
            pd.DataFrame(
                current
            )
        )

    return runs


def main():

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "Airborne State V2: "
            "DribbleGap + temporal reception confirmation"
        )
    )

    parser.add_argument(
        "--possession",
        default=(
            "outputs/"
            "possession_v5_3_frames.csv"
        ),
    )

    parser.add_argument(
        "--gate",
        default=(
            "outputs/"
            "controller_gate_v2.csv"
        ),
    )

    parser.add_argument(
        "--events",
        default=(
            "outputs/"
            "air_touch_v4_events.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
    )

    parser.add_argument(
        "--launch-min-speed",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--reception-confirm-sec",
        type=float,
        default=0.12,
        help=(
            "Minimum KEEP duration "
            "required for a trusted reception."
        ),
    )

    parser.add_argument(
        "--dribble-return-sec",
        type=float,
        default=0.32,
        help=(
            "Maximum same-player return gap "
            "classified as DribbleGap."
        ),
    )

    args = parser.parse_args()


    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    frames_out = os.path.join(
        args.output_dir,
        "airborne_state_v2_frames.csv",
    )

    episodes_out = os.path.join(
        args.output_dir,
        "airborne_state_v2_episodes.csv",
    )

    receptions_out = os.path.join(
        args.output_dir,
        "airborne_state_v2_receptions.csv",
    )

    dribble_out = os.path.join(
        args.output_dir,
        "airborne_state_v2_dribble_gaps.csv",
    )

    launch_out = os.path.join(
        args.output_dir,
        "airborne_state_v2_launch_candidates.csv",
    )


    # ========================================================
    # LOAD
    # ========================================================

    print("")
    print(
        "Loading Possession V5.3..."
    )

    possession = pd.read_csv(
        args.possession
    )


    print(
        "Loading Controller Gate V2..."
    )

    gate = pd.read_csv(
        args.gate
    )


    print(
        "Loading AirTouch V4 events..."
    )

    events = pd.read_csv(
        args.events
    )


    # ========================================================
    # NORMALIZE POSSESSION
    # ========================================================

    for column in [
        "frame",
        "time_sec",
        "ball_speed_mps",
        "controller_id_v5_3",
        "last_touch_id_v5_3",
    ]:

        if (
            column
            in
            possession.columns
        ):

            possession[
                column
            ] = pd.to_numeric(
                possession[
                    column
                ],
                errors="coerce",
            )


    possession = (
        possession
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


    possession[
        "frame"
    ] = (
        possession[
            "frame"
        ]
        .astype(int)
    )


    possession_by_frame = (
        possession
        .set_index(
            "frame"
        )
    )


    frame_max = int(
        possession[
            "frame"
        ]
        .max()
    )


    fps = estimate_fps(
        possession
    )


    reception_confirm_frames = max(
        2,
        int(
            round(
                args.reception_confirm_sec
                *
                fps
            )
        ),
    )


    dribble_return_frames = max(
        2,
        int(
            round(
                args.dribble_return_sec
                *
                fps
            )
        ),
    )


    print(
        f"Estimated FPS: "
        f"{fps:.2f}"
    )

    print(
        "Reception confirm:",
        reception_confirm_frames,
        "frames",
    )

    print(
        "Dribble return window:",
        dribble_return_frames,
        "frames",
    )


    # ========================================================
    # NORMALIZE CONTROLLER GATE
    # ========================================================

    for column in [
        "frame",
        "controller_id",
        "controller_run_id",
        "controller_run_length",
        "controller_run_position",
        "ball_speed_mps",
        "foot_distance_px",
        "foot_distance_norm",
        "ball_rel_y",
    ]:

        if (
            column
            in
            gate.columns
        ):

            gate[
                column
            ] = pd.to_numeric(
                gate[
                    column
                ],
                errors="coerce",
            )


    gate = gate.dropna(
        subset=[
            "frame"
        ]
    ).copy()


    gate[
        "frame"
    ] = (
        gate[
            "frame"
        ]
        .astype(int)
    )


    gate_keep = gate[
        (
            gate[
                "gate_status"
            ]
            ==
            "KEEP"
        )

        &

        gate[
            "controller_team"
        ].isin(
            [
                "A",
                "B",
            ]
        )

        &

        gate[
            "controller_id"
        ].notna()
    ].copy()


    keep_frames = set(
        gate_keep[
            "frame"
        ]
        .astype(int)
        .tolist()
    )


    keep_runs = (
        group_keep_runs(
            gate_keep
        )
    )


    # ========================================================
    # NORMALIZE AIR EVENTS
    # ========================================================

    for column in [
        "physical_event_id",
        "start_frame",
        "end_frame",
        "last_touch_track_id",
    ]:

        if (
            column
            in
            events.columns
        ):

            events[
                column
            ] = pd.to_numeric(
                events[
                    column
                ],
                errors="coerce",
            )


    events = events.dropna(
        subset=[
            "start_frame",
            "end_frame",
            "final_class",
        ]
    ).copy()


    events[
        "start_frame"
    ] = (
        events[
            "start_frame"
        ]
        .astype(int)
    )


    events[
        "end_frame"
    ] = (
        events[
            "end_frame"
        ]
        .astype(int)
    )


    contest_frames = set()
    air_touch_frames = set()

    air_touch_info = {}


    for _, event in (
        events.iterrows()
    ):

        start = int(
            event[
                "start_frame"
            ]
        )

        end = int(
            event[
                "end_frame"
            ]
        )


        event_class = clean_text(
            event[
                "final_class"
            ],
            "",
        )


        if (
            event_class
            ==
            "AerialContest"
        ):

            contest_frames.update(
                range(
                    start,
                    end + 1,
                )
            )


        elif (
            event_class
            ==
            "AirTouch"
        ):

            for frame in range(
                start,
                end + 1,
            ):

                air_touch_frames.add(
                    frame
                )


                air_touch_info[
                    frame
                ] = {

                    "team":
                        clean_team(
                            event.get(
                                "last_touch_team"
                            )
                        ),

                    "track_id":
                        clean_id(
                            event.get(
                                "last_touch_track_id"
                            )
                        ),
                }


    # ========================================================
    # TRUSTED RECEPTION RUNS
    #
    # V1:
    # 5-frame stable reception
    #
    # V2:
    # time-based 0.12 sec
    # = 3 frames at 25 FPS
    # ========================================================

    reception_rows = []

    trusted_receptions = []


    for run_id, run in enumerate(
        keep_runs,
        start=1,
    ):

        run = (
            run
            .sort_values(
                "frame"
            )
            .reset_index(
                drop=True
            )
        )


        start = int(
            run.iloc[0][
                "frame"
            ]
        )


        end = int(
            run.iloc[-1][
                "frame"
            ]
        )


        team = clean_team(
            run.iloc[0][
                "controller_team"
            ]
        )


        track_id = clean_id(
            run.iloc[0][
                "controller_id"
            ]
        )


        length = len(
            run
        )


        trusted = (
            length
            >=
            reception_confirm_frames
        )


        record = {

            "run_id":
                run_id,

            "start_frame":
                start,

            "end_frame":
                end,

            "start_time_sec":
                start / fps,

            "end_time_sec":
                end / fps,

            "team":
                team,

            "track_id":
                track_id,

            "run_length":
                length,

            "close_control_frames":
                int(
                    (
                        run[
                            "control_mode"
                        ]
                        ==
                        "close_control"
                    ).sum()
                ),

            "dribble_frames":
                int(
                    (
                        run[
                            "control_mode"
                        ]
                        ==
                        "dribble"
                    ).sum()
                ),

            "median_foot_distance_norm":
                float(
                    run[
                        "foot_distance_norm"
                    ]
                    .median()
                ),

            "median_ball_rel_y":
                float(
                    run[
                        "ball_rel_y"
                    ]
                    .median()
                ),

            "median_ball_speed_mps":
                float(
                    run[
                        "ball_speed_mps"
                    ]
                    .median()
                ),

            "trusted_reception":
                trusted,
        }


        reception_rows.append(
            record
        )


        if trusted:

            trusted_receptions.append(
                record
            )


    receptions_df = pd.DataFrame(
        reception_rows
    )


    receptions_df.to_csv(
        receptions_out,
        index=False,
    )


    trusted_receptions = sorted(
        trusted_receptions,
        key=lambda r: (
            r[
                "start_frame"
            ]
        ),
    )


    reception_start_map = {

        int(
            r[
                "start_frame"
            ]
        ):
            r

        for r in trusted_receptions
    }


    print("")

    print(
        "Controller KEEP runs:",
        len(
            keep_runs
        ),
    )


    print(
        "Trusted reception runs:",
        len(
            trusted_receptions
        ),
    )


    # ========================================================
    # CONTROL EXIT
    #
    # Build potential launch candidates.
    # ========================================================

    launch_candidates = []


    for run in keep_runs:

        run = (
            run
            .sort_values(
                "frame"
            )
            .reset_index(
                drop=True
            )
        )


        run_end = int(
            run.iloc[-1][
                "frame"
            ]
        )


        candidate_frame = (
            run_end + 1
        )


        if (
            candidate_frame
            >
            frame_max
        ):
            continue


        if (
            candidate_frame
            in
            keep_frames
        ):
            continue


        if (
            candidate_frame
            not in
            possession_by_frame.index
        ):
            continue


        current = (
            possession_by_frame.loc[
                candidate_frame
            ]
        )


        if isinstance(
            current,
            pd.DataFrame,
        ):

            current = (
                current.iloc[0]
            )


        state = clean_text(
            current.get(
                "ball_state_v5_3"
            ),
            "",
        )


        speed = current.get(
            "ball_speed_mps",
            np.nan,
        )


        if (
            state
            not in
            {
                "InTransit",
                "Loose",
                "Uncontrolled",
            }
        ):
            continue


        if (
            pd.isna(
                speed
            )

            or

            float(
                speed
            )
            <
            args.launch_min_speed
        ):
            continue


        launch_candidates.append(
            {

                "candidate_frame":
                    candidate_frame,

                "candidate_time_sec":
                    candidate_frame
                    /
                    fps,

                "launch_team":
                    clean_team(
                        run.iloc[-1][
                            "controller_team"
                        ]
                    ),

                "launch_track_id":
                    clean_id(
                        run.iloc[-1][
                            "controller_id"
                        ]
                    ),

                "previous_keep_run_start":
                    int(
                        run.iloc[0][
                            "frame"
                        ]
                    ),

                "previous_keep_run_end":
                    run_end,

                "ball_state":
                    state,

                "ball_speed_mps":
                    float(
                        speed
                    ),
            }
        )


    # ========================================================
    # DRIBBLE GAP
    #
    # If:
    #
    # same player loses close control
    # +
    # same player quickly gets it back
    # +
    # no AirTouch / AerialContest
    #
    # then:
    #
    # NOT airborne
    # = DribbleGap
    # ========================================================

    dribble_gaps = []

    dribble_frame_map = {}

    committed_launch_map = {}


    for candidate in (
        launch_candidates
    ):

        start = int(
            candidate[
                "candidate_frame"
            ]
        )


        launch_team = (
            candidate[
                "launch_team"
            ]
        )


        launch_id = (
            candidate[
                "launch_track_id"
            ]
        )


        next_reception = next(
            (

                reception

                for reception
                in trusted_receptions

                if int(
                    reception[
                        "start_frame"
                    ]
                )
                >
                start
            ),
            None,
        )


        is_dribble = False


        if (
            next_reception
            is not None
        ):

            reception_start = int(
                next_reception[
                    "start_frame"
                ]
            )


            gap_len = (
                reception_start
                -
                start
            )


            same_return_player = (
                same_player(
                    launch_team,
                    launch_id,
                    next_reception[
                        "team"
                    ],
                    next_reception[
                        "track_id"
                    ],
                )
            )


            has_air_event = any(
                (

                    frame
                    in
                    contest_frames

                    or

                    frame
                    in
                    air_touch_frames

                )

                for frame in range(
                    start,
                    reception_start,
                )
            )


            if (

                same_return_player

                and

                gap_len
                <=
                dribble_return_frames

                and

                not has_air_event

            ):

                is_dribble = True


                gap_record = {

                    "start_frame":
                        start,

                    "end_frame":
                        reception_start - 1,

                    "start_time_sec":
                        start / fps,

                    "end_time_sec":
                        (
                            reception_start - 1
                        )
                        /
                        fps,

                    "duration_frames":
                        max(
                            0,
                            reception_start
                            -
                            start,
                        ),

                    "team":
                        launch_team,

                    "track_id":
                        launch_id,

                    "reception_frame":
                        reception_start,

                    "reason":
                        "same_player_quick_return",
                }


                dribble_gaps.append(
                    gap_record
                )


                for frame in range(
                    start,
                    reception_start,
                ):

                    dribble_frame_map[
                        frame
                    ] = (
                        gap_record
                    )


        candidate[
            "classified_as_dribble_gap"
        ] = (
            is_dribble
        )


        candidate[
            "next_trusted_reception_frame"
        ] = (

            int(
                next_reception[
                    "start_frame"
                ]
            )

            if (
                next_reception
                is not None
            )

            else

            np.nan
        )


        if not is_dribble:

            committed_launch_map[
                start
            ] = candidate


    # ========================================================
    # SAVE LAUNCH / DRIBBLE DEBUG
    # ========================================================

    pd.DataFrame(
        launch_candidates
    ).to_csv(
        launch_out,
        index=False,
    )


    dribble_df = pd.DataFrame(
        dribble_gaps
    )


    dribble_df.to_csv(
        dribble_out,
        index=False,
    )


    # ========================================================
    # AIRBORNE STATE MACHINE
    # ========================================================

    output_rows = []

    episode_rows = []


    airborne = False

    episode_id = 0


    episode_start = None

    episode_launch_reason = None

    episode_launch_team = None

    episode_launch_id = np.nan


    episode_had_contest = False

    episode_had_air_touch = False


    episode_last_touch_team = None

    episode_last_touch_id = np.nan


    def close_episode(
        end_frame,
        resolution_type,
        receiver_team=None,
        receiver_id=np.nan,
    ):

        nonlocal airborne

        nonlocal episode_start

        nonlocal episode_launch_reason

        nonlocal episode_launch_team

        nonlocal episode_launch_id

        nonlocal episode_had_contest

        nonlocal episode_had_air_touch

        nonlocal episode_last_touch_team

        nonlocal episode_last_touch_id


        if (

            episode_start
            is None

            or

            end_frame
            <
            episode_start

        ):

            airborne = False

            return


        confirmed_pass = (

            resolution_type
            ==
            "trusted_reception"

            and

            episode_launch_reason
            ==
            "trusted_control_exit"

            and

            episode_launch_team
            is not None

            and

            receiver_team
            is not None

            and

            episode_launch_team
            ==
            receiver_team

            and

            pd.notna(
                episode_launch_id
            )

            and

            pd.notna(
                receiver_id
            )

            and

            abs(
                float(
                    episode_launch_id
                )
                -
                float(
                    receiver_id
                )
            )
            >
            0.1

            and

            not episode_had_contest

            and

            not episode_had_air_touch
        )


        episode_rows.append(
            {

                "episode_id":
                    episode_id,

                "start_frame":
                    episode_start,

                "end_frame":
                    end_frame,

                "start_time_sec":
                    episode_start
                    /
                    fps,

                "end_time_sec":
                    end_frame
                    /
                    fps,

                "duration_frames":
                    (
                        end_frame
                        -
                        episode_start
                        +
                        1
                    ),

                "duration_sec":
                    (
                        end_frame
                        -
                        episode_start
                        +
                        1
                    )
                    /
                    fps,

                "launch_reason":
                    episode_launch_reason,

                "launch_team":
                    episode_launch_team,

                "launch_track_id":
                    episode_launch_id,

                "had_aerial_contest":
                    episode_had_contest,

                "had_air_touch":
                    episode_had_air_touch,

                "last_touch_team":
                    episode_last_touch_team,

                "last_touch_track_id":
                    episode_last_touch_id,

                "resolution_type":
                    resolution_type,

                "receiver_team":
                    receiver_team,

                "receiver_track_id":
                    receiver_id,

                "confirmed_pass":
                    confirmed_pass,

                "flight_class":
                    (
                        "ConfirmedPassInFlight"

                        if
                        confirmed_pass

                        else

                        "AirborneUncontrolled"
                    ),
            }
        )


        airborne = False

        episode_start = None

        episode_launch_reason = None

        episode_launch_team = None

        episode_launch_id = np.nan

        episode_had_contest = False

        episode_had_air_touch = False

        episode_last_touch_team = None

        episode_last_touch_id = np.nan


    # ========================================================
    # RUN
    # ========================================================

    print("")

    print(
        "========================================"
    )

    print(
        "AIRBORNE STATE V2"
    )

    print(
        "DribbleGap + 3-frame Reception"
    )

    print(
        "========================================"
    )

    print("")


    for _, base in (
        possession.iterrows()
    ):

        frame = int(
            base[
                "frame"
            ]
        )


        baseline_state = clean_text(
            base.get(
                "ball_state_v5_3"
            ),
            "Unknown",
        )


        baseline_possession = (
            clean_text(
                base.get(
                    "possession_v5_3"
                ),
                "Unknown",
            )
        )


        baseline_controller_team = (
            clean_team(
                base.get(
                    "controller_team_v5_3"
                )
            )
        )


        baseline_controller_id = (
            clean_id(
                base.get(
                    "controller_id_v5_3"
                )
            )
        )


        physical_contest = (

            frame
            in
            contest_frames

            or

            baseline_state
            ==
            "AerialContest"
        )


        physical_air_touch = (

            frame
            in
            air_touch_frames
        )


        # ====================================================
        # RECEPTION ENDS AIRBORNE
        # ====================================================

        reception_here = (
            reception_start_map.get(
                frame
            )
        )


        if (

            airborne

            and

            reception_here
            is not None

            and

            not physical_contest

            and

            not physical_air_touch

            and

            frame
            >
            episode_start

        ):

            close_episode(
                end_frame=(
                    frame - 1
                ),

                resolution_type=(
                    "trusted_reception"
                ),

                receiver_team=(
                    reception_here[
                        "team"
                    ]
                ),

                receiver_id=(
                    reception_here[
                        "track_id"
                    ]
                ),
            )


        # ====================================================
        # START AIRBORNE
        # ====================================================

        if not airborne:

            launch_reason = None

            launch_team = None

            launch_id = np.nan


            # ------------------------------------------------
            # Aerial contest
            # ------------------------------------------------

            if physical_contest:

                launch_reason = (
                    "aerial_contest"
                )


            # ------------------------------------------------
            # Confirmed AirTouch
            # ------------------------------------------------

            elif physical_air_touch:

                info = (
                    air_touch_info.get(
                        frame,
                        {},
                    )
                )


                launch_reason = (
                    "confirmed_air_touch"
                )


                launch_team = clean_team(
                    info.get(
                        "team"
                    )
                )


                launch_id = clean_id(
                    info.get(
                        "track_id"
                    )
                )


            # ------------------------------------------------
            # Trusted control exit
            # ------------------------------------------------

            elif (
                frame
                in
                committed_launch_map
            ):

                info = (
                    committed_launch_map[
                        frame
                    ]
                )


                launch_reason = (
                    "trusted_control_exit"
                )


                launch_team = (
                    info[
                        "launch_team"
                    ]
                )


                launch_id = (
                    info[
                        "launch_track_id"
                    ]
                )


            if (
                launch_reason
                is not None
            ):

                airborne = True

                episode_id += 1

                episode_start = frame

                episode_launch_reason = (
                    launch_reason
                )

                episode_launch_team = (
                    launch_team
                )

                episode_launch_id = (
                    launch_id
                )


                episode_last_touch_team = (
                    launch_team
                )

                episode_last_touch_id = (
                    launch_id
                )


        # ====================================================
        # OUTPUT ROW
        # ====================================================

        row = base.to_dict()


        row[
            "airborne_active_v2"
        ] = airborne


        row[
            "airborne_episode_id_v2"
        ] = (

            episode_id

            if airborne

            else

            np.nan
        )


        row[
            "airborne_launch_reason_v2"
        ] = (

            episode_launch_reason

            if airborne

            else

            None
        )


        row[
            "airborne_launch_team_v2"
        ] = (

            episode_launch_team

            if airborne

            else

            None
        )


        row[
            "airborne_launch_track_id_v2"
        ] = (

            episode_launch_id

            if airborne

            else

            np.nan
        )


        row[
            "controller_suppressed_by_airborne_v2"
        ] = False


        row[
            "dribble_gap_v2"
        ] = (

            frame
            in
            dribble_frame_map
        )


        # ====================================================
        # DRIBBLE GAP
        # ====================================================

        if (

            not airborne

            and

            frame
            in
            dribble_frame_map

        ):

            gap = (
                dribble_frame_map[
                    frame
                ]
            )


            row[
                "airborne_state_v2"
            ] = (
                "DribbleGap"
            )


            row[
                "possession_airborne_v2"
            ] = (
                gap[
                    "team"
                ]
            )


            row[
                "controller_team_airborne_v2"
            ] = (
                gap[
                    "team"
                ]
            )


            row[
                "controller_id_airborne_v2"
            ] = (
                gap[
                    "track_id"
                ]
            )


            row[
                "airborne_last_touch_team_v2"
            ] = (
                gap[
                    "team"
                ]
            )


            row[
                "airborne_last_touch_id_v2"
            ] = (
                gap[
                    "track_id"
                ]
            )


        # ====================================================
        # AIRBORNE
        # ====================================================

        elif airborne:

            # ------------------------------------------------
            # Aerial Contest
            # ------------------------------------------------

            if physical_contest:

                episode_had_contest = (
                    True
                )


                episode_last_touch_team = (
                    None
                )

                episode_last_touch_id = (
                    np.nan
                )


                row[
                    "airborne_state_v2"
                ] = (
                    "AerialContest"
                )


                row[
                    "possession_airborne_v2"
                ] = (
                    "Contested"
                )


            # ------------------------------------------------
            # AirTouch
            # ------------------------------------------------

            elif physical_air_touch:

                episode_had_air_touch = (
                    True
                )


                info = (
                    air_touch_info.get(
                        frame,
                        {},
                    )
                )


                episode_last_touch_team = (
                    clean_team(
                        info.get(
                            "team"
                        )
                    )
                )


                episode_last_touch_id = (
                    clean_id(
                        info.get(
                            "track_id"
                        )
                    )
                )


                row[
                    "airborne_state_v2"
                ] = (
                    "AirTouchInFlight"
                )


                row[
                    "possession_airborne_v2"
                ] = (
                    "Unknown"
                )


            # ------------------------------------------------
            # Missing
            # ------------------------------------------------

            elif (
                baseline_state
                ==
                "Missing"
            ):

                row[
                    "airborne_state_v2"
                ] = (
                    "AirborneMissing"
                )


                row[
                    "possession_airborne_v2"
                ] = (
                    "Unknown"
                )


            # ------------------------------------------------
            # Generic Airborne
            # ------------------------------------------------

            else:

                row[
                    "airborne_state_v2"
                ] = (
                    "AirborneUncontrolled"
                )


                row[
                    "possession_airborne_v2"
                ] = (
                    "Unknown"
                )


            # ------------------------------------------------
            # Airborne = no controller
            # ------------------------------------------------

            row[
                "controller_team_airborne_v2"
            ] = None


            row[
                "controller_id_airborne_v2"
            ] = np.nan


            row[
                "airborne_last_touch_team_v2"
            ] = (
                episode_last_touch_team
            )


            row[
                "airborne_last_touch_id_v2"
            ] = (
                episode_last_touch_id
            )


            if (

                baseline_controller_team
                is not None

                and

                pd.notna(
                    baseline_controller_id
                )

            ):

                row[
                    "controller_suppressed_by_airborne_v2"
                ] = True


        # ====================================================
        # GROUND / RESOLVED
        # ====================================================

        else:

            row[
                "airborne_state_v2"
            ] = (
                "GroundOrResolved"
            )


            row[
                "possession_airborne_v2"
            ] = (
                baseline_possession
            )


            row[
                "controller_team_airborne_v2"
            ] = (
                baseline_controller_team
            )


            row[
                "controller_id_airborne_v2"
            ] = (
                baseline_controller_id
            )


            row[
                "airborne_last_touch_team_v2"
            ] = clean_team(
                base.get(
                    "last_touch_team_v5_3"
                )
            )


            row[
                "airborne_last_touch_id_v2"
            ] = clean_id(
                base.get(
                    "last_touch_id_v5_3"
                )
            )


        output_rows.append(
            row
        )


    # ========================================================
    # CLOSE LAST EPISODE
    # ========================================================

    if airborne:

        close_episode(
            end_frame=frame_max,

            resolution_type=(
                "video_end"
            ),
        )


    # ========================================================
    # DATAFRAME
    # ========================================================

    frames_df = pd.DataFrame(
        output_rows
    )


    episodes_df = pd.DataFrame(
        episode_rows
    )


    # ========================================================
    # RETROACTIVE PASS LABEL
    # ========================================================

    if (
        len(
            episodes_df
        )
        >
        0
    ):

        for _, episode in (
            episodes_df.iterrows()
        ):

            if not bool(
                episode[
                    "confirmed_pass"
                ]
            ):
                continue


            eid = int(
                episode[
                    "episode_id"
                ]
            )


            team = clean_team(
                episode[
                    "launch_team"
                ]
            )


            mask = (

                (
                    frames_df[
                        "airborne_episode_id_v2"
                    ]
                    ==
                    eid
                )

                &

                (
                    frames_df[
                        "airborne_state_v2"
                    ]
                    ==
                    "AirborneUncontrolled"
                )
            )


            frames_df.loc[
                mask,
                "airborne_state_v2"
            ] = (
                "ConfirmedPassInFlight"
            )


            frames_df.loc[
                mask,
                "possession_airborne_v2"
            ] = (
                team
            )


    # ========================================================
    # SAVE
    # ========================================================

    frames_df.to_csv(
        frames_out,
        index=False,
    )


    episodes_df.to_csv(
        episodes_out,
        index=False,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "========================================"
    )

    print(
        "AIRBORNE STATE V2 SUMMARY"
    )

    print(
        "========================================"
    )

    print("")


    print(
        "Frames:",
        len(
            frames_df
        ),
    )


    print(
        "Airborne frames:",
        int(
            frames_df[
                "airborne_active_v2"
            ].sum()
        ),
    )


    print(
        "DribbleGap frames:",
        int(
            frames_df[
                "dribble_gap_v2"
            ].sum()
        ),
    )


    print(
        "Airborne episodes:",
        len(
            episodes_df
        ),
    )


    print(
        "Controllers suppressed while airborne:",
        int(
            frames_df[
                "controller_suppressed_by_airborne_v2"
            ]
            .sum()
        ),
    )


    print("")

    print(
        "AIRBORNE STATES"
    )


    print(
        frames_df[
            "airborne_state_v2"
        ]
        .value_counts()
        .to_string()
    )


    # ========================================================
    # DRIBBLE GAPS
    # ========================================================

    print("")

    print(
        "========================================"
    )

    print(
        "DRIBBLE GAPS"
    )

    print(
        "========================================"
    )

    print("")


    if len(
        dribble_df
    ) > 0:

        print(
            dribble_df[
                [
                    "start_frame",
                    "end_frame",
                    "duration_frames",
                    "team",
                    "track_id",
                    "reception_frame",
                ]
            ]
            .to_string(
                index=False
            )
        )

    else:

        print(
            "None"
        )


    # ========================================================
    # AIRBORNE EPISODES
    # ========================================================

    print("")

    print(
        "========================================"
    )

    print(
        "AIRBORNE EPISODES"
    )

    print(
        "========================================"
    )

    print("")


    if len(
        episodes_df
    ) > 0:

        columns = [
            "episode_id",
            "start_frame",
            "end_frame",
            "duration_frames",
            "duration_sec",
            "launch_reason",
            "launch_team",
            "launch_track_id",
            "had_aerial_contest",
            "had_air_touch",
            "resolution_type",
            "receiver_team",
            "receiver_track_id",
            "confirmed_pass",
            "flight_class",
        ]


        print(
            episodes_df[
                columns
            ]
            .round(3)
            .to_string(
                index=False
            )
        )


        # ====================================================
        # LONGEST EPISODES
        # ====================================================

        print("")

        print(
            "Longest episodes:"
        )


        print(
            episodes_df[
                [
                    "episode_id",
                    "start_frame",
                    "end_frame",
                    "duration_sec",
                    "launch_reason",
                    "resolution_type",
                ]
            ]
            .sort_values(
                "duration_sec",
                ascending=False,
            )
            .head(5)
            .round(3)
            .to_string(
                index=False
            )
        )


    # ========================================================
    # OUTPUTS
    # ========================================================

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

    print(
        "Receptions:",
        receptions_out
    )

    print(
        "Dribble gaps:",
        dribble_out
    )

    print(
        "Launch candidates:",
        launch_out
    )


if __name__ == "__main__":

    main()
