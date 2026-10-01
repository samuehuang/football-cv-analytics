import os

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

POSSESSION_CSV = "outputs/possession_v5_3_frames.csv"
CONTROLLER_GATE_CSV = "outputs/controller_gate_v2.csv"
AIR_TOUCH_EVENTS_CSV = "outputs/air_touch_v4_events.csv"

OUTPUT_FRAMES_CSV = "outputs/airborne_state_v1_frames.csv"
OUTPUT_EPISODES_CSV = "outputs/airborne_state_v1_episodes.csv"
OUTPUT_RECEPTIONS_CSV = "outputs/airborne_state_v1_receptions.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

# ------------------------------------------------------------
# LAUNCH
#
# Controlled -> ball leaves player.
# We use speed only for this launch type.
#
# AerialContest / AirTouch do NOT need this threshold.
# ------------------------------------------------------------

LAUNCH_MIN_SPEED_MPS = 6.0
LAUNCH_CONFIRM_NONCONTROL_FRAMES = 2


# ------------------------------------------------------------
# TRUSTED RECEPTION
#
# Important:
#
# One KEEP frame is NOT enough.
#
# Reception must look stable over time.
# ------------------------------------------------------------

MIN_RECEPTION_RUN_FRAMES = 5

MIN_RECEPTION_CLOSE_CONTROL_FRAMES = 2

MAX_RECEPTION_MEDIAN_FOOT_NORM = 1.05
MAX_RECEPTION_MEDIAN_BALL_SPEED = 8.0

RECEPTION_MIN_MEDIAN_REL_Y = 0.55
RECEPTION_MAX_MEDIAN_REL_Y = 1.25


# ------------------------------------------------------------
# Alternative dribble-style reception.
#
# Example:
# A#11 around frame 647-651.
#
# A valid controller can push ball ahead, so we cannot demand
# every frame to be close_control.
# ------------------------------------------------------------

MIN_DRIBBLE_RECEPTION_RUN_FRAMES = 6
MIN_DRIBBLE_FRAMES = 4
MAX_DRIBBLE_RECEPTION_MEDIAN_NORM = 1.10


# ============================================================
# HELPERS
# ============================================================

def clean_team(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {"A", "B"}:
        return value

    return None


def clean_id(value):

    if pd.isna(value):
        return np.nan

    try:
        return float(value)

    except Exception:
        return np.nan


def clean_text(value, default=None):

    if pd.isna(value):
        return default

    text = str(value).strip()

    if (
        text == ""
        or
        text.lower() in {
            "none",
            "nan",
        }
    ):
        return default

    return text


def same_player(team1, id1, team2, id2):

    team1 = clean_team(team1)
    team2 = clean_team(team2)

    if (
        team1 is None
        or
        team2 is None
        or
        team1 != team2
        or
        pd.isna(id1)
        or
        pd.isna(id2)
    ):
        return False

    return abs(
        float(id1) - float(id2)
    ) < 0.1


# ============================================================
# LOAD
# ============================================================

print("")
print("Loading Possession V5.3...")

possession = pd.read_csv(
    POSSESSION_CSV
)


print("Loading Controller Gate V2...")

gate = pd.read_csv(
    CONTROLLER_GATE_CSV
)


print("Loading AirTouch V4 events...")

events = pd.read_csv(
    AIR_TOUCH_EVENTS_CSV
)


# ============================================================
# NORMALIZE POSSESSION
# ============================================================

for column in [
    "frame",
    "time_sec",
    "ball_speed_mps",
    "controller_id_v5_3",
    "last_touch_id_v5_3",
]:

    if column in possession.columns:

        possession[column] = pd.to_numeric(
            possession[column],
            errors="coerce",
        )


possession = possession.dropna(
    subset=["frame"]
).copy()


possession["frame"] = (
    possession["frame"]
    .astype(int)
)


possession = (
    possession
    .sort_values("frame")
    .reset_index(drop=True)
)


possession_by_frame = (
    possession
    .set_index("frame")
)


all_frames = (
    possession["frame"]
    .astype(int)
    .tolist()
)


# ============================================================
# FPS
# ============================================================

fps = 25.0


timing = (
    possession[
        [
            "frame",
            "time_sec",
        ]
    ]
    .dropna()
    .sort_values("frame")
)


if len(timing) >= 2:

    df_frame = (
        timing["frame"]
        .diff()
    )

    df_time = (
        timing["time_sec"]
        .diff()
    )


    valid = (

        df_frame.notna()

        &

        df_time.notna()

        &

        (df_frame > 0)

        &

        (df_time > 0)
    )


    estimates = (

        df_frame[valid]
        /
        df_time[valid]
    )


    estimates = estimates[
        np.isfinite(estimates)
    ]


    if len(estimates) > 0:

        fps = float(
            estimates.median()
        )


print(
    f"Estimated FPS: {fps:.2f}"
)


# ============================================================
# NORMALIZE CONTROLLER GATE
# ============================================================

for column in [
    "frame",
    "time_sec",
    "controller_id",
    "controller_run_id",
    "controller_run_length",
    "controller_run_position",
    "ball_speed_mps",
    "foot_distance_px",
    "foot_distance_norm",
    "ball_rel_y",
]:

    if column in gate.columns:

        gate[column] = pd.to_numeric(
            gate[column],
            errors="coerce",
        )


gate = gate.dropna(
    subset=["frame"]
).copy()


gate["frame"] = (
    gate["frame"]
    .astype(int)
)


gate = (
    gate
    .sort_values("frame")
    .reset_index(drop=True)
)


# ============================================================
# NORMALIZE AIR EVENTS
# ============================================================

for column in [
    "physical_event_id",
    "start_frame",
    "end_frame",
    "last_touch_track_id",
]:

    if column in events.columns:

        events[column] = pd.to_numeric(
            events[column],
            errors="coerce",
        )


events = events.dropna(
    subset=[
        "start_frame",
        "end_frame",
        "final_class",
    ]
).copy()


events["start_frame"] = (
    events["start_frame"]
    .astype(int)
)


events["end_frame"] = (
    events["end_frame"]
    .astype(int)
)


# ============================================================
# PHYSICAL EVENT LOOKUPS
# ============================================================

aerial_contest_frames = set()
air_touch_frames = set()

air_touch_info = {}


for _, event in events.iterrows():

    start = int(
        event["start_frame"]
    )

    end = int(
        event["end_frame"]
    )

    event_class = clean_text(
        event["final_class"],
        "",
    )


    if event_class == "AerialContest":

        for frame in range(
            start,
            end + 1,
        ):

            aerial_contest_frames.add(
                frame
            )


    elif event_class == "AirTouch":

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
                            "last_touch_team",
                            None,
                        )
                    ),

                "track_id":
                    clean_id(
                        event.get(
                            "last_touch_track_id",
                            np.nan,
                        )
                    ),

                "event_id":
                    event.get(
                        "physical_event_id",
                        np.nan,
                    ),
            }


# ============================================================
# BUILD GATE KEEP RUNS
#
# These runs are candidate receptions.
# ============================================================

keep_gate = gate[
    gate["gate_status"]
    ==
    "KEEP"
].copy()


keep_gate = keep_gate[
    keep_gate["controller_team"].isin(
        ["A", "B"]
    )
].copy()


keep_gate = keep_gate.dropna(
    subset=[
        "controller_id",
    ]
).copy()


keep_gate = (
    keep_gate
    .sort_values("frame")
    .reset_index(drop=True)
)


runs = []

current_rows = []
current_team = None
current_id = np.nan
previous_frame = None


def flush_run():

    global current_rows
    global current_team
    global current_id
    global previous_frame

    if len(current_rows) > 0:

        runs.append(
            pd.DataFrame(
                current_rows
            )
        )

    current_rows = []
    current_team = None
    current_id = np.nan
    previous_frame = None


for _, row in keep_gate.iterrows():

    frame = int(
        row["frame"]
    )

    team = clean_team(
        row["controller_team"]
    )

    track_id = clean_id(
        row["controller_id"]
    )


    if len(current_rows) == 0:

        current_rows = [
            row.to_dict()
        ]

        current_team = team
        current_id = track_id
        previous_frame = frame

        continue


    consecutive = (
        frame
        ==
        previous_frame + 1
    )


    same_controller = (
        same_player(
            team,
            track_id,
            current_team,
            current_id,
        )
    )


    if (
        consecutive
        and
        same_controller
    ):

        current_rows.append(
            row.to_dict()
        )

        previous_frame = frame


    else:

        flush_run()

        current_rows = [
            row.to_dict()
        ]

        current_team = team
        current_id = track_id
        previous_frame = frame


flush_run()


# ============================================================
# CLASSIFY TRUSTED RECEPTION RUNS
# ============================================================

reception_rows = []

trusted_reception_start = {}


for run_id, run in enumerate(
    runs,
    start=1,
):

    run = (
        run
        .sort_values("frame")
        .reset_index(drop=True)
    )


    length = len(run)


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


    close_count = int(
        (
            run["control_mode"]
            ==
            "close_control"
        ).sum()
    )


    dribble_count = int(
        (
            run["control_mode"]
            ==
            "dribble"
        ).sum()
    )


    median_norm = float(
        run[
            "foot_distance_norm"
        ]
        .median()
    )


    median_rel_y = float(
        run[
            "ball_rel_y"
        ]
        .median()
    )


    median_speed = float(
        run[
            "ball_speed_mps"
        ]
        .median()
    )


    normal_reception = (

        length
        >=
        MIN_RECEPTION_RUN_FRAMES

        and

        close_count
        >=
        MIN_RECEPTION_CLOSE_CONTROL_FRAMES

        and

        median_norm
        <=
        MAX_RECEPTION_MEDIAN_FOOT_NORM

        and

        median_speed
        <=
        MAX_RECEPTION_MEDIAN_BALL_SPEED

        and

        median_rel_y
        >=
        RECEPTION_MIN_MEDIAN_REL_Y

        and

        median_rel_y
        <=
        RECEPTION_MAX_MEDIAN_REL_Y
    )


    dribble_reception = (

        length
        >=
        MIN_DRIBBLE_RECEPTION_RUN_FRAMES

        and

        dribble_count
        >=
        MIN_DRIBBLE_FRAMES

        and

        median_norm
        <=
        MAX_DRIBBLE_RECEPTION_MEDIAN_NORM

        and

        median_speed
        <=
        MAX_RECEPTION_MEDIAN_BALL_SPEED

        and

        median_rel_y
        >=
        RECEPTION_MIN_MEDIAN_REL_Y

        and

        median_rel_y
        <=
        RECEPTION_MAX_MEDIAN_REL_Y
    )


    trusted = (

        normal_reception
        or
        dribble_reception
    )


    if normal_reception:

        reception_type = (
            "stable_close_control"
        )

    elif dribble_reception:

        reception_type = (
            "stable_dribble_control"
        )

    else:

        reception_type = (
            "not_trusted"
        )


    start_frame = int(
        run.iloc[0][
            "frame"
        ]
    )

    end_frame = int(
        run.iloc[-1][
            "frame"
        ]
    )


    reception_rows.append(
        {
            "run_id":
                run_id,

            "start_frame":
                start_frame,

            "end_frame":
                end_frame,

            "start_time_sec":
                start_frame / fps,

            "end_time_sec":
                end_frame / fps,

            "team":
                team,

            "track_id":
                track_id,

            "run_length":
                length,

            "close_control_frames":
                close_count,

            "dribble_frames":
                dribble_count,

            "median_foot_distance_norm":
                median_norm,

            "median_ball_rel_y":
                median_rel_y,

            "median_ball_speed_mps":
                median_speed,

            "trusted_reception":
                trusted,

            "reception_type":
                reception_type,
        }
    )


    if trusted:

        trusted_reception_start[
            start_frame
        ] = {
            "run_id":
                run_id,

            "team":
                team,

            "track_id":
                track_id,

            "end_frame":
                end_frame,

            "reception_type":
                reception_type,
        }


receptions = pd.DataFrame(
    reception_rows
)


receptions.to_csv(
    OUTPUT_RECEPTIONS_CSV,
    index=False,
)


print("")
print(
    "Controller KEEP runs:",
    len(runs)
)

print(
    "Trusted reception runs:",
    int(
        receptions[
            "trusted_reception"
        ].sum()
    )
)


# ============================================================
# TRUSTED CONTROL FRAME LOOKUP
#
# Used only for launch detection outside airborne state.
# ============================================================

trusted_control_frames = set(
    keep_gate[
        "frame"
    ]
    .astype(int)
    .tolist()
)


trusted_control_info = {}


for _, row in keep_gate.iterrows():

    frame = int(
        row[
            "frame"
        ]
    )


    trusted_control_info[
        frame
    ] = {
        "team":
            clean_team(
                row[
                    "controller_team"
                ]
            ),

        "track_id":
            clean_id(
                row[
                    "controller_id"
                ]
            ),
    }


# ============================================================
# LAUNCH CONFIRMATION
# ============================================================

def confirmed_control_exit(
    frame,
):

    previous_frame = (
        frame - 1
    )


    if (
        previous_frame
        not in
        trusted_control_frames
    ):

        return None


    if (
        frame
        not in
        possession_by_frame.index
    ):

        return None


    current = possession_by_frame.loc[
        frame
    ]


    if isinstance(
        current,
        pd.DataFrame
    ):

        current = current.iloc[0]


    state = clean_text(
        current.get(
            "ball_state_v5_3",
            None,
        ),
        "",
    )


    speed = current.get(
        "ball_speed_mps",
        np.nan,
    )


    if (
        state
        not in {
            "InTransit",
            "Loose",
            "Uncontrolled",
        }
    ):

        return None


    if (
        pd.isna(speed)

        or

        float(speed)
        <
        LAUNCH_MIN_SPEED_MPS
    ):

        return None


    # --------------------------------------------------------
    # Confirm that control stays absent briefly.
    # --------------------------------------------------------

    for offset in range(
        LAUNCH_CONFIRM_NONCONTROL_FRAMES
    ):

        check_frame = (
            frame + offset
        )


        if (
            check_frame
            in
            trusted_control_frames
        ):

            return None


    launch_info = (
        trusted_control_info.get(
            previous_frame,
            None,
        )
    )


    if launch_info is None:
        return None


    return {
        "team":
            launch_info[
                "team"
            ],

        "track_id":
            launch_info[
                "track_id"
            ],

        "reason":
            "trusted_control_exit",
    }


# ============================================================
# FIRST PASS:
# BUILD AIRBORNE EPISODES
# ============================================================

episode_rows = []

airborne = False

episode_id = 0

episode_start = None
episode_launch_reason = None
episode_launch_team = None
episode_launch_id = np.nan

episode_had_contest = False
episode_had_air_touch = False
episode_had_opponent_air_touch = False

episode_last_touch_team = None
episode_last_touch_id = np.nan

episode_frame_records = []


def close_episode(
    end_frame,
    resolution_type,
    receiver_team=None,
    receiver_id=np.nan,
):

    global airborne
    global episode_start
    global episode_launch_reason
    global episode_launch_team
    global episode_launch_id
    global episode_had_contest
    global episode_had_air_touch
    global episode_had_opponent_air_touch
    global episode_last_touch_team
    global episode_last_touch_id
    global episode_frame_records


    if (
        episode_start is None

        or

        len(
            episode_frame_records
        )
        ==
        0
    ):

        airborne = False
        return


    # ========================================================
    # PASS CONFIRMATION
    #
    # Very conservative:
    #
    # launch player known
    # receiver known
    # same team
    # different player
    # no aerial contest
    # no AirTouch interruption
    # ========================================================

    confirmed_pass = (

        resolution_type
        ==
        "trusted_reception"

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


    flight_class = (

        "ConfirmedPassInFlight"

        if confirmed_pass

        else

        "AirborneUncontrolled"
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
                episode_start / fps,

            "end_time_sec":
                end_frame / fps,

            "duration_frames":
                end_frame
                -
                episode_start
                +
                1,

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

            "had_opponent_air_touch":
                episode_had_opponent_air_touch,

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
                flight_class,
        }
    )


    # Write class into frame records.
    for record in episode_frame_records:

        record[
            "episode_flight_class"
        ] = flight_class

        record[
            "confirmed_pass"
        ] = confirmed_pass

        record[
            "confirmed_pass_team"
        ] = (
            episode_launch_team
            if confirmed_pass
            else None
        )


    airborne = False

    episode_start = None
    episode_launch_reason = None
    episode_launch_team = None
    episode_launch_id = np.nan

    episode_had_contest = False
    episode_had_air_touch = False
    episode_had_opponent_air_touch = False

    episode_last_touch_team = None
    episode_last_touch_id = np.nan

    episode_frame_records = []


# ============================================================
# OUTPUT FRAME RECORDS
# ============================================================

frame_output = []


print("")
print(
    "========================================"
)

print(
    "AIRBORNE STATE V1"
)

print(
    "Temporal flight state machine"
)

print(
    "========================================"
)

print("")


for _, base in possession.iterrows():

    frame = int(
        base[
            "frame"
        ]
    )


    baseline_state = clean_text(
        base.get(
            "ball_state_v5_3",
            None,
        ),
        "Unknown",
    )


    baseline_possession = clean_text(
        base.get(
            "possession_v5_3",
            None,
        ),
        "Unknown",
    )


    baseline_controller_team = clean_team(
        base.get(
            "controller_team_v5_3",
            None,
        )
    )


    baseline_controller_id = clean_id(
        base.get(
            "controller_id_v5_3",
            np.nan,
        )
    )


    physical_contest = (

        frame
        in
        aerial_contest_frames

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


    # ========================================================
    # START AIRBORNE
    # ========================================================

    if not airborne:

        launch = None


        # ----------------------------------------------------
        # Highest-confidence launch:
        # aerial contest.
        # ----------------------------------------------------

        if physical_contest:

            launch = {
                "team":
                    None,

                "track_id":
                    np.nan,

                "reason":
                    "aerial_contest",
            }


        # ----------------------------------------------------
        # Confirmed AirTouch.
        # ----------------------------------------------------

        elif physical_air_touch:

            info = air_touch_info.get(
                frame,
                {},
            )


            launch = {
                "team":
                    info.get(
                        "team",
                        None,
                    ),

                "track_id":
                    info.get(
                        "track_id",
                        np.nan,
                    ),

                "reason":
                    "confirmed_air_touch",
            }


        # ----------------------------------------------------
        # Trusted control -> ball leaves.
        # ----------------------------------------------------

        else:

            launch = confirmed_control_exit(
                frame
            )


        if launch is not None:

            airborne = True

            episode_id += 1

            episode_start = frame

            episode_launch_reason = (
                launch[
                    "reason"
                ]
            )

            episode_launch_team = (
                launch[
                    "team"
                ]
            )

            episode_launch_id = (
                launch[
                    "track_id"
                ]
            )

            episode_last_touch_team = (
                episode_launch_team
            )

            episode_last_touch_id = (
                episode_launch_id
            )


    # ========================================================
    # RECEPTION CAN END AIRBORNE
    #
    # Physical AirTouch / contest always wins over reception.
    # ========================================================

    reception_here = (
        trusted_reception_start.get(
            frame,
            None,
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
            end_frame=frame - 1,

            resolution_type="trusted_reception",

            receiver_team=reception_here[
                "team"
            ],

            receiver_id=reception_here[
                "track_id"
            ],
        )


    # ========================================================
    # FRAME OUTPUT DEFAULT
    # ========================================================

    output = base.to_dict()


    output[
        "airborne_active_v1"
    ] = airborne


    output[
        "airborne_episode_id_v1"
    ] = (
        episode_id
        if airborne
        else
        np.nan
    )


    output[
        "airborne_launch_reason_v1"
    ] = (
        episode_launch_reason
        if airborne
        else
        None
    )


    output[
        "airborne_launch_team_v1"
    ] = (
        episode_launch_team
        if airborne
        else
        None
    )


    output[
        "airborne_launch_track_id_v1"
    ] = (
        episode_launch_id
        if airborne
        else
        np.nan
    )


    output[
        "airborne_last_touch_team_v1"
    ] = (
        episode_last_touch_team
        if airborne
        else
        None
    )


    output[
        "airborne_last_touch_id_v1"
    ] = (
        episode_last_touch_id
        if airborne
        else
        np.nan
    )


    output[
        "controller_suppressed_by_airborne_v1"
    ] = False


    output[
        "episode_flight_class"
    ] = None


    output[
        "confirmed_pass"
    ] = False


    output[
        "confirmed_pass_team"
    ] = None


    # ========================================================
    # IF AIRBORNE
    # ========================================================

    if airborne:

        # ----------------------------------------------------
        # Physical contest.
        # ----------------------------------------------------

        if physical_contest:

            episode_had_contest = True


            output[
                "airborne_state_v1"
            ] = (
                "AerialContest"
            )


            output[
                "possession_airborne_v1"
            ] = (
                "Contested"
            )


            episode_last_touch_team = None
            episode_last_touch_id = np.nan


        # ----------------------------------------------------
        # Confirmed AirTouch.
        # ----------------------------------------------------

        elif physical_air_touch:

            episode_had_air_touch = True


            info = air_touch_info.get(
                frame,
                {},
            )


            touch_team = clean_team(
                info.get(
                    "team",
                    None,
                )
            )


            touch_id = clean_id(
                info.get(
                    "track_id",
                    np.nan,
                )
            )


            if (
                episode_launch_team
                is not None

                and

                touch_team
                is not None

                and

                touch_team
                !=
                episode_launch_team
            ):

                episode_had_opponent_air_touch = True


            episode_last_touch_team = (
                touch_team
            )

            episode_last_touch_id = (
                touch_id
            )


            output[
                "airborne_state_v1"
            ] = (
                "AirTouchInFlight"
            )


            output[
                "possession_airborne_v1"
            ] = (
                "Unknown"
            )


        # ----------------------------------------------------
        # Missing ball while airborne.
        # Still do not invent controller.
        # ----------------------------------------------------

        elif (
            baseline_state
            ==
            "Missing"
        ):

            output[
                "airborne_state_v1"
            ] = (
                "AirborneMissing"
            )


            output[
                "possession_airborne_v1"
            ] = (
                "Unknown"
            )


        # ----------------------------------------------------
        # Generic unresolved flight.
        # ----------------------------------------------------

        else:

            output[
                "airborne_state_v1"
            ] = (
                "AirborneUncontrolled"
            )


            output[
                "possession_airborne_v1"
            ] = (
                "Unknown"
            )


        # ====================================================
        # AIRBORNE = NO CONTROLLER
        # ====================================================

        output[
            "controller_team_airborne_v1"
        ] = None


        output[
            "controller_id_airborne_v1"
        ] = np.nan


        if (
            baseline_controller_team
            is not None

            and

            pd.notna(
                baseline_controller_id
            )
        ):

            output[
                "controller_suppressed_by_airborne_v1"
            ] = True


        # ====================================================
        # LAST TOUCH
        # ====================================================

        output[
            "airborne_last_touch_team_v1"
        ] = (
            episode_last_touch_team
        )


        output[
            "airborne_last_touch_id_v1"
        ] = (
            episode_last_touch_id
        )


        episode_frame_records.append(
            output
        )


    # ========================================================
    # NOT AIRBORNE
    # ========================================================

    else:

        output[
            "airborne_state_v1"
        ] = (
            "GroundOrResolved"
        )


        output[
            "possession_airborne_v1"
        ] = (
            baseline_possession
        )


        output[
            "controller_team_airborne_v1"
        ] = (
            baseline_controller_team
        )


        output[
            "controller_id_airborne_v1"
        ] = (
            baseline_controller_id
        )


    frame_output.append(
        output
    )


# ============================================================
# CLOSE OPEN EPISODE AT END OF VIDEO
# ============================================================

if airborne:

    close_episode(
        end_frame=int(
            possession[
                "frame"
            ]
            .max()
        ),

        resolution_type="video_end",

        receiver_team=None,

        receiver_id=np.nan,
    )


# ============================================================
# IMPORTANT:
#
# episode_frame_records contains references to dict objects
# already stored in frame_output, so close_episode() has
# retroactively filled flight class / pass fields.
# ============================================================

frames_df = pd.DataFrame(
    frame_output
)


episodes_df = pd.DataFrame(
    episode_rows
)


# ============================================================
# SECOND PASS:
#
# Confirmed pass episodes can retroactively carry TEAM
# possession during flight.
#
# Controller always remains NONE.
# ============================================================

if len(
    episodes_df
) > 0:

    for _, episode in (
        episodes_df.iterrows()
    ):

        if not bool(
            episode[
                "confirmed_pass"
            ]
        ):
            continue


        start = int(
            episode[
                "start_frame"
            ]
        )

        end = int(
            episode[
                "end_frame"
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
                    "frame"
                ]
                >=
                start
            )

            &

            (
                frames_df[
                    "frame"
                ]
                <=
                end
            )

            &

            (
                frames_df[
                    "airborne_active_v1"
                ]
                ==
                True
            )

            &

            (
                frames_df[
                    "airborne_state_v1"
                ]
                ==
                "AirborneUncontrolled"
            )
        )


        frames_df.loc[
            mask,
            "airborne_state_v1"
        ] = (
            "ConfirmedPassInFlight"
        )


        frames_df.loc[
            mask,
            "possession_airborne_v1"
        ] = (
            team
        )


        frames_df.loc[
            mask,
            "confirmed_pass"
        ] = True


        frames_df.loc[
            mask,
            "confirmed_pass_team"
        ] = team


# ============================================================
# SAVE
# ============================================================

frames_df.to_csv(
    OUTPUT_FRAMES_CSV,
    index=False,
)


episodes_df.to_csv(
    OUTPUT_EPISODES_CSV,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("")
print(
    "========================================"
)

print(
    "AIRBORNE STATE V1 SUMMARY"
)

print(
    "========================================"
)

print("")


print(
    "Frames:",
    len(
        frames_df
    )
)


print(
    "Airborne frames:",
    int(
        frames_df[
            "airborne_active_v1"
        ]
        .sum()
    )
)


print(
    "Airborne episodes:",
    len(
        episodes_df
    )
)


print("")

print(
    "AIRBORNE STATES"
)


print(
    frames_df[
        "airborne_state_v1"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "CONTROLLERS SUPPRESSED WHILE AIRBORNE:",
    int(
        frames_df[
            "controller_suppressed_by_airborne_v1"
        ]
        .sum()
    )
)


# ============================================================
# EPISODES
# ============================================================

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

    display_columns = [
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
            display_columns
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


# ============================================================
# QA:
# USER-IDENTIFIED LONG AIRBORNE AREA
#
# This is NOT used by the algorithm.
# It is only an evaluation window.
# ============================================================

qa_start = 177
qa_end = 240


qa = frames_df[
    (
        frames_df[
            "frame"
        ]
        >=
        qa_start
    )

    &

    (
        frames_df[
            "frame"
        ]
        <=
        qa_end
    )
].copy()


qa_airborne = int(
    qa[
        "airborne_active_v1"
    ]
    .sum()
)


qa_total = len(
    qa
)


coverage = (

    100.0
    *
    qa_airborne
    /
    qa_total

    if qa_total > 0

    else 0.0
)


print("")

print(
    "========================================"
)

print(
    "QA | USER-IDENTIFIED ~177-240 AREA"
)

print(
    "========================================"
)

print("")


print(
    f"Airborne coverage: "
    f"{qa_airborne}/{qa_total} "
    f"({coverage:.1f}%)"
)


# ============================================================
# KEY FRAMES
# ============================================================

print("")

print(
    "KEY FRAME CHECK"
)


for frame in [
    172,
    175,
    177,
    178,
    203,
    213,
    225,
    226,
    232,
    235,
    240,
    253,
    281,
    650,
]:

    rows = frames_df[
        frames_df[
            "frame"
        ]
        ==
        frame
    ]


    if len(rows) == 0:
        continue


    row = rows.iloc[0]


    print(
        f"frame={frame:3d} | "
        f"time={row.get('time_sec', np.nan):5.2f}s | "
        f"airborne={bool(row['airborne_active_v1'])} | "
        f"state={row['airborne_state_v1']} | "
        f"possession={row['possession_airborne_v1']} | "
        f"controller="
        f"{row['controller_team_airborne_v1']} "
        f"{row['controller_id_airborne_v1']} | "
        f"suppressed="
        f"{bool(row['controller_suppressed_by_airborne_v1'])}"
    )


# ============================================================
# COMPRESSED SEGMENTS 160-260
# ============================================================

print("")

print(
    "========================================"
)

print(
    "STATE SEGMENTS | FRAMES 160-260"
)

print(
    "========================================"
)

print("")


segment_df = frames_df[
    (
        frames_df[
            "frame"
        ]
        >=
        160
    )

    &

    (
        frames_df[
            "frame"
        ]
        <=
        260
    )
][
    [
        "frame",
        "airborne_active_v1",
        "airborne_state_v1",
        "possession_airborne_v1",
    ]
].copy()


segments = []

start = None
end = None
last_key = None


for _, row in (
    segment_df.iterrows()
):

    frame = int(
        row[
            "frame"
        ]
    )


    key = (
        bool(
            row[
                "airborne_active_v1"
            ]
        ),

        str(
            row[
                "airborne_state_v1"
            ]
        ),

        str(
            row[
                "possession_airborne_v1"
            ]
        ),
    )


    if start is None:

        start = frame
        end = frame
        last_key = key
        continue


    if (
        frame
        ==
        end + 1

        and

        key
        ==
        last_key
    ):

        end = frame


    else:

        segments.append(
            (
                start,
                end,
                last_key,
            )
        )

        start = frame
        end = frame
        last_key = key


if start is not None:

    segments.append(
        (
            start,
            end,
            last_key,
        )
    )


for (
    start,
    end,
    key,
) in segments:

    active, state, team = key


    print(
        f"{start:3d}-{end:3d} | "
        f"{start/fps:5.2f}-{end/fps:5.2f}s | "
        f"airborne={active!s:5s} | "
        f"{state:24s} | "
        f"possession={team}"
    )


# ============================================================
# OUTPUT
# ============================================================

print("")

print(
    "Frames CSV:",
    OUTPUT_FRAMES_CSV
)

print(
    "Episodes CSV:",
    OUTPUT_EPISODES_CSV
)

print(
    "Receptions CSV:",
    OUTPUT_RECEPTIONS_CSV
)
