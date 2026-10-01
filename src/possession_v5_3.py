import os

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

POSSESSION_V5_2_CSV = "outputs/possession_v5_2_frames.csv"
AIR_TOUCH_V4_CSV = "outputs/air_touch_v4_events.csv"

OUTPUT_FRAMES_CSV = "outputs/possession_v5_3_frames.csv"
OUTPUT_SUMMARY_CSV = "outputs/possession_v5_3_summary.csv"
OUTPUT_AUDIT_CSV = "outputs/possession_v5_3_audit.csv"
OUTPUT_EPISODES_CSV = "outputs/possession_v5_3_contest_episodes.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# CONTEST EPISODE
#
# Same window already validated by Controller Gate V2:
#
# confirmed contest frame
# +
# short post-contest lock
#
# This is a MINIMUM persistence window.
#
# High-confidence AirTouch can resolve it early.
# ============================================================

CONTEST_POST_LOCK_FRAMES = 4


# ============================================================
# REGRESSION FRAMES
# ============================================================

FRAME_159 = 159

FRAME_172 = 172
FRAME_173 = 173
FRAME_174 = 174
FRAME_175 = 175
FRAME_176 = 176
FRAME_177 = 177

FRAME_203 = 203
FRAME_226 = 226
FRAME_253 = 253
FRAME_281 = 281
FRAME_287 = 287
FRAME_650 = 650


# ============================================================
# HELPERS
# ============================================================

def clean_team(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {
        "A",
        "B",
    }:
        return value

    return None


def clean_id(value):

    if pd.isna(value):
        return np.nan

    try:
        return float(value)

    except Exception:
        return np.nan


def clean_text(
    value,
    default=None,
):

    if pd.isna(value):
        return default

    text = str(value).strip()

    if (
        text == ""
        or
        text.lower()
        in {
            "none",
            "nan",
        }
    ):
        return default

    return text


def same_player_id(
    a,
    b,
):

    if (
        pd.isna(a)
        or
        pd.isna(b)
    ):
        return False

    return (
        abs(
            float(a)
            -
            float(b)
        )
        <
        0.1
    )


# ============================================================
# LOAD
# ============================================================

print("")
print("Loading Possession V5.2...")

possession = pd.read_csv(
    POSSESSION_V5_2_CSV
)


print("Loading AirTouch V4 events...")

events = pd.read_csv(
    AIR_TOUCH_V4_CSV
)


# ============================================================
# NORMALIZE POSSESSION
# ============================================================

numeric_possession_columns = [
    "frame",
    "time_sec",

    "controller_id_v5_2",
    "last_touch_id_v5_2",

    "ball_speed_mps",

    "controller_gate_run_length",
    "controller_gate_run_position",
]


for column in numeric_possession_columns:

    if column in possession.columns:

        possession[column] = pd.to_numeric(
            possession[column],
            errors="coerce",
        )


possession = possession.dropna(
    subset=[
        "frame"
    ]
).copy()


possession["frame"] = (
    possession["frame"]
    .astype(int)
)


possession = (
    possession
    .sort_values(
        "frame"
    )
    .reset_index(
        drop=True
    )
)


possession_by_frame = (
    possession
    .set_index(
        "frame"
    )
)


# ============================================================
# FPS
# ============================================================

fps = 25.0


if (
    "time_sec"
    in possession.columns
):

    timing = (
        possession[
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


    if len(timing) >= 2:

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
                frame_diff
                >
                0
            )

            &

            (
                time_diff
                >
                0
            )
        )


        if valid.any():

            fps_values = (

                frame_diff[
                    valid
                ]

                /

                time_diff[
                    valid
                ]
            )


            fps_values = fps_values[
                np.isfinite(
                    fps_values
                )
            ]


            if len(fps_values) > 0:

                fps = float(
                    fps_values.median()
                )


print(
    f"Estimated FPS: {fps:.2f}"
)


# ============================================================
# NORMALIZE PHYSICAL EVENTS
# ============================================================

numeric_event_columns = [
    "physical_event_id",
    "start_frame",
    "end_frame",
    "last_touch_track_id",
]


for column in numeric_event_columns:

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
    events[
        "start_frame"
    ]
    .astype(int)
)


events["end_frame"] = (
    events[
        "end_frame"
    ]
    .astype(int)
)


events = (
    events
    .sort_values(
        [
            "start_frame",
            "end_frame",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# HIGH-CONFIDENCE AIR TOUCH LOOKUP
#
# Confirmed AirTouch is allowed to resolve a contest episode
# early.
# ============================================================

air_touch_frames = set()


confirmed_air_touch_events = events[
    events[
        "final_class"
    ]
    ==
    "AirTouch"
].copy()


for _, event in (
    confirmed_air_touch_events.iterrows()
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


    for frame in range(
        start,
        end + 1,
    ):

        air_touch_frames.add(
            frame
        )


# ============================================================
# BUILD CONTEST EPISODES
# ============================================================

physical_contests = events[
    events[
        "final_class"
    ]
    ==
    "AerialContest"
].copy()


episode_rows = []


for _, event in physical_contests.iterrows():

    confirmed_start = int(
        event[
            "start_frame"
        ]
    )


    confirmed_end = int(
        event[
            "end_frame"
        ]
    )


    proposed_end = (
        confirmed_end
        +
        CONTEST_POST_LOCK_FRAMES
    )


    actual_end = (
        proposed_end
    )


    resolution_reason = (
        "minimum_contest_persistence_complete"
    )


    # ========================================================
    # EARLY RESOLUTION:
    # confirmed AirTouch
    #
    # If an AirTouch occurs AFTER the confirmed contest starts,
    # the contest ends immediately before that AirTouch.
    # ========================================================

    for frame in range(
        confirmed_start + 1,
        proposed_end + 1,
    ):

        if (
            frame
            in
            air_touch_frames
        ):

            actual_end = (
                frame - 1
            )

            resolution_reason = (
                "resolved_by_confirmed_air_touch"
            )

            break


    # Never create negative-duration episode.
    actual_end = max(
        confirmed_end,
        actual_end,
    )


    episode_rows.append(
        {
            "contest_event_id":
                int(
                    event.get(
                        "physical_event_id",
                        len(
                            episode_rows
                        )
                        +
                        1,
                    )
                ),

            "confirmed_start_frame":
                confirmed_start,

            "confirmed_end_frame":
                confirmed_end,

            "episode_start_frame":
                confirmed_start,

            "episode_end_frame":
                actual_end,

            "episode_start_time_sec":
                confirmed_start
                /
                fps,

            "episode_end_time_sec":
                actual_end
                /
                fps,

            "participants":
                event.get(
                    "participants",
                    None,
                ),

            "resolution_reason":
                resolution_reason,
        }
    )


contest_episodes = pd.DataFrame(
    episode_rows
)


contest_episodes.to_csv(
    OUTPUT_EPISODES_CSV,
    index=False,
)


# ============================================================
# FRAME -> CONTEST EPISODE LOOKUP
# ============================================================

contest_episode_by_frame = {}


for _, episode in (
    contest_episodes.iterrows()
):

    start = int(
        episode[
            "episode_start_frame"
        ]
    )

    end = int(
        episode[
            "episode_end_frame"
        ]
    )


    for frame in range(
        start,
        end + 1,
    ):

        contest_episode_by_frame[
            frame
        ] = episode


# ============================================================
# OUTPUT
# ============================================================

rows = []
audit_rows = []


print("")
print(
    "========================================"
)

print(
    "POSSESSION V5.3"
)

print(
    "Contest Episode / Resolution Gate"
)

print(
    "========================================"
)

print("")


# ============================================================
# MAIN LOOP
# ============================================================

for _, base in possession.iterrows():

    frame = int(
        base[
            "frame"
        ]
    )


    row = base.to_dict()


    # ========================================================
    # V5.2 BASELINE
    # ========================================================

    possession_v5_2 = clean_text(
        base.get(
            "possession_v5_2",
            None,
        ),
        "Unknown",
    )


    ball_state_v5_2 = clean_text(
        base.get(
            "ball_state_v5_2",
            None,
        ),
        "Unknown",
    )


    controller_team_v5_2 = clean_team(
        base.get(
            "controller_team_v5_2",
            None,
        )
    )


    controller_id_v5_2 = clean_id(
        base.get(
            "controller_id_v5_2",
            np.nan,
        )
    )


    last_touch_team_v5_2 = clean_team(
        base.get(
            "last_touch_team_v5_2",
            None,
        )
    )


    last_touch_id_v5_2 = clean_id(
        base.get(
            "last_touch_id_v5_2",
            np.nan,
        )
    )


    last_touch_type_v5_2 = clean_text(
        base.get(
            "last_touch_type_v5_2",
            None,
        ),
        None,
    )


    source_v5_2 = clean_text(
        base.get(
            "source_v5_2",
            None,
        ),
        "baseline_v4",
    )


    # ========================================================
    # DEFAULT V5.3 = V5.2
    # ========================================================

    possession_v5_3 = (
        possession_v5_2
    )


    ball_state_v5_3 = (
        ball_state_v5_2
    )


    controller_team_v5_3 = (
        controller_team_v5_2
    )


    controller_id_v5_3 = (
        controller_id_v5_2
    )


    last_touch_team_v5_3 = (
        last_touch_team_v5_2
    )


    last_touch_id_v5_3 = (
        last_touch_id_v5_2
    )


    last_touch_type_v5_3 = (
        last_touch_type_v5_2
    )


    source_v5_3 = (
        source_v5_2
    )


    contest_episode_active = False
    contest_episode_id = np.nan
    contest_episode_phase = None
    contest_episode_participants = None


    # ========================================================
    # CONTEST EPISODE
    # ========================================================

    episode = contest_episode_by_frame.get(
        frame,
        None
    )


    # ========================================================
    # IMPORTANT PRIORITY
    #
    # A confirmed AirTouch always beats a carried contest.
    #
    # Normally early-resolution building already prevents
    # overlap, but keep this guard explicit.
    # ========================================================

    confirmed_air_touch_here = (

        frame
        in
        air_touch_frames
    )


    if (
        episode is not None

        and

        not confirmed_air_touch_here
    ):

        contest_episode_active = True


        contest_episode_id = episode.get(
            "contest_event_id",
            np.nan,
        )


        contest_episode_participants = (
            episode.get(
                "participants",
                None,
            )
        )


        confirmed_start = int(
            episode[
                "confirmed_start_frame"
            ]
        )


        confirmed_end = int(
            episode[
                "confirmed_end_frame"
            ]
        )


        # ====================================================
        # CONTEST PHASE
        # ====================================================

        if (
            frame
            >=
            confirmed_start

            and

            frame
            <=
            confirmed_end
        ):

            contest_episode_phase = (
                "ContestConfirmed"
            )


            source_v5_3 = (
                "contest_episode_confirmed"
            )


        else:

            contest_episode_phase = (
                "ContestCarry"
            )


            source_v5_3 = (
                "contest_episode_carry"
            )


        # ====================================================
        # CONTEST OVERRIDE
        # ====================================================

        possession_v5_3 = (
            "Contested"
        )


        ball_state_v5_3 = (
            "AerialContest"
        )


        controller_team_v5_3 = None
        controller_id_v5_3 = np.nan


        last_touch_team_v5_3 = None
        last_touch_id_v5_3 = np.nan


        last_touch_type_v5_3 = (
            "AerialContestUnknown"
        )


    # ========================================================
    # SAVE
    # ========================================================

    row[
        "possession_v5_3"
    ] = possession_v5_3


    row[
        "ball_state_v5_3"
    ] = ball_state_v5_3


    row[
        "controller_team_v5_3"
    ] = controller_team_v5_3


    row[
        "controller_id_v5_3"
    ] = controller_id_v5_3


    row[
        "last_touch_team_v5_3"
    ] = last_touch_team_v5_3


    row[
        "last_touch_id_v5_3"
    ] = last_touch_id_v5_3


    row[
        "last_touch_type_v5_3"
    ] = last_touch_type_v5_3


    row[
        "source_v5_3"
    ] = source_v5_3


    row[
        "contest_episode_active"
    ] = contest_episode_active


    row[
        "contest_episode_id"
    ] = contest_episode_id


    row[
        "contest_episode_phase"
    ] = contest_episode_phase


    row[
        "contest_episode_participants"
    ] = contest_episode_participants


    # ========================================================
    # CHANGED FROM V5.2?
    # ========================================================

    changed_from_v5_2 = (

        str(
            possession_v5_3
        )
        !=
        str(
            possession_v5_2
        )

        or

        str(
            ball_state_v5_3
        )
        !=
        str(
            ball_state_v5_2
        )

        or

        clean_team(
            controller_team_v5_3
        )
        !=
        clean_team(
            controller_team_v5_2
        )

        or

        (
            pd.notna(
                controller_id_v5_3
            )
            !=
            pd.notna(
                controller_id_v5_2
            )
        )
    )


    row[
        "contest_changed_from_v5_2"
    ] = (
        changed_from_v5_2
    )


    rows.append(
        row
    )


    # ========================================================
    # AUDIT ALL CONTEST EPISODE FRAMES
    # ========================================================

    if contest_episode_active:

        audit_rows.append(
            {
                "frame":
                    frame,

                "time_sec":
                    base.get(
                        "time_sec",
                        frame / fps,
                    ),

                "possession_v5_2":
                    possession_v5_2,

                "possession_v5_3":
                    possession_v5_3,

                "ball_state_v5_2":
                    ball_state_v5_2,

                "ball_state_v5_3":
                    ball_state_v5_3,

                "controller_team_v5_2":
                    controller_team_v5_2,

                "controller_id_v5_2":
                    controller_id_v5_2,

                "controller_team_v5_3":
                    controller_team_v5_3,

                "controller_id_v5_3":
                    controller_id_v5_3,

                "contest_episode_id":
                    contest_episode_id,

                "contest_episode_phase":
                    contest_episode_phase,

                "participants":
                    contest_episode_participants,

                "changed_from_v5_2":
                    changed_from_v5_2,

                "source_v5_2":
                    source_v5_2,

                "source_v5_3":
                    source_v5_3,
            }
        )


# ============================================================
# DATAFRAMES
# ============================================================

v5_3 = pd.DataFrame(
    rows
)


audit = pd.DataFrame(
    audit_rows
)


v5_3.to_csv(
    OUTPUT_FRAMES_CSV,
    index=False,
)


audit.to_csv(
    OUTPUT_AUDIT_CSV,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for key, value in (
    v5_3[
        "possession_v5_3"
    ]
    .value_counts()
    .items()
):

    summary_rows.append(
        {
            "type":
                "possession",

            "name":
                key,

            "frames":
                int(
                    value
                ),

            "seconds":
                float(
                    value
                )
                /
                fps,
        }
    )


for key, value in (
    v5_3[
        "ball_state_v5_3"
    ]
    .value_counts()
    .items()
):

    summary_rows.append(
        {
            "type":
                "ball_state",

            "name":
                key,

            "frames":
                int(
                    value
                ),

            "seconds":
                float(
                    value
                )
                /
                fps,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUTPUT_SUMMARY_CSV,
    index=False,
)


# ============================================================
# FRAME LOOKUP
# ============================================================

v5_3_by_frame = (
    v5_3
    .set_index(
        "frame"
    )
)


def get_frame(
    frame
):

    if (
        frame
        not in
        v5_3_by_frame.index
    ):

        return None


    result = v5_3_by_frame.loc[
        frame
    ]


    if isinstance(
        result,
        pd.DataFrame
    ):

        result = result.iloc[0]


    return result


def print_frame(
    frame
):

    result = get_frame(
        frame
    )


    print("")


    if result is None:

        print(
            f"frame {frame}: NOT FOUND"
        )

        return


    print(
        f"FRAME {frame} | "
        f"{result.get('time_sec', np.nan):.2f}s"
    )


    print(
        "  V5.2 | "
        f"possession="
        f"{result.get('possession_v5_2')} | "
        f"state="
        f"{result.get('ball_state_v5_2')} | "
        f"controller="
        f"{result.get('controller_team_v5_2')} "
        f"{result.get('controller_id_v5_2')}"
    )


    print(
        "  V5.3 | "
        f"possession="
        f"{result.get('possession_v5_3')} | "
        f"state="
        f"{result.get('ball_state_v5_3')} | "
        f"controller="
        f"{result.get('controller_team_v5_3')} "
        f"{result.get('controller_id_v5_3')} | "
        f"episode="
        f"{result.get('contest_episode_phase')} | "
        f"source="
        f"{result.get('source_v5_3')}"
    )


# ============================================================
# REGRESSION HELPERS
# ============================================================

def is_contest_frame(
    frame,
):

    result = get_frame(
        frame
    )


    if result is None:
        return False


    return (

        str(
            result[
                "possession_v5_3"
            ]
        )
        ==
        "Contested"

        and

        str(
            result[
                "ball_state_v5_3"
            ]
        )
        ==
        "AerialContest"

        and

        clean_team(
            result[
                "controller_team_v5_3"
            ]
        )
        is None
    )


# ============================================================
# REGRESSIONS
# ============================================================

r159 = get_frame(
    FRAME_159
)


reg159 = (

    r159 is not None

    and

    str(
        r159[
            "possession_v5_3"
        ]
    )
    ==
    "B"

    and

    str(
        r159[
            "ball_state_v5_3"
        ]
    )
    ==
    "InTransit"
)


reg172 = is_contest_frame(
    FRAME_172
)

reg173 = is_contest_frame(
    FRAME_173
)

reg174 = is_contest_frame(
    FRAME_174
)

reg175 = is_contest_frame(
    FRAME_175
)

reg176 = is_contest_frame(
    FRAME_176
)


r177 = get_frame(
    FRAME_177
)


reg177 = (

    r177 is not None

    and

    str(
        r177[
            "possession_v5_3"
        ]
    )
    ==
    str(
        r177[
            "possession_v5_2"
        ]
    )

    and

    str(
        r177[
            "ball_state_v5_3"
        ]
    )
    ==
    str(
        r177[
            "ball_state_v5_2"
        ]
    )

    and

    bool(
        r177[
            "contest_episode_active"
        ]
    )
    is False
)


# ------------------------------------------------------------
# Existing regressions must remain unchanged.
# ------------------------------------------------------------

r203 = get_frame(
    FRAME_203
)


reg203 = (

    r203 is not None

    and

    str(
        r203[
            "possession_v5_3"
        ]
    )
    ==
    str(
        r203[
            "possession_v5_2"
        ]
    )

    and

    str(
        r203[
            "ball_state_v5_3"
        ]
    )
    ==
    str(
        r203[
            "ball_state_v5_2"
        ]
    )
)


r226 = get_frame(
    FRAME_226
)


reg226 = (

    r226 is not None

    and

    str(
        r226[
            "possession_v5_3"
        ]
    )
    ==
    "A"

    and

    str(
        r226[
            "ball_state_v5_3"
        ]
    )
    ==
    "Uncontrolled"

    and

    clean_team(
        r226[
            "controller_team_v5_3"
        ]
    )
    is None
)


r253 = get_frame(
    FRAME_253
)


reg253 = (

    r253 is not None

    and

    str(
        r253[
            "possession_v5_3"
        ]
    )
    ==
    "Contested"

    and

    str(
        r253[
            "ball_state_v5_3"
        ]
    )
    ==
    "AerialContest"

    and

    clean_team(
        r253[
            "controller_team_v5_3"
        ]
    )
    is None
)


r281 = get_frame(
    FRAME_281
)


reg281 = (

    r281 is not None

    and

    str(
        r281[
            "possession_v5_3"
        ]
    )
    ==
    "B"

    and

    str(
        r281[
            "ball_state_v5_3"
        ]
    )
    ==
    "InTransit"

    and

    clean_team(
        r281[
            "controller_team_v5_3"
        ]
    )
    is None

    and

    clean_team(
        r281[
            "last_touch_team_v5_3"
        ]
    )
    ==
    "B"

    and

    same_player_id(
        r281[
            "last_touch_id_v5_3"
        ],
        10,
    )
)


r287 = get_frame(
    FRAME_287
)


reg287 = (

    r287 is not None

    and

    str(
        r287[
            "possession_v5_3"
        ]
    )
    ==
    str(
        r287[
            "possession_v5_2"
        ]
    )

    and

    str(
        r287[
            "ball_state_v5_3"
        ]
    )
    ==
    str(
        r287[
            "ball_state_v5_2"
        ]
    )
)


r650 = get_frame(
    FRAME_650
)


reg650 = (

    r650 is not None

    and

    str(
        r650[
            "possession_v5_3"
        ]
    )
    ==
    "A"

    and

    str(
        r650[
            "ball_state_v5_3"
        ]
    )
    ==
    "Controlled"

    and

    clean_team(
        r650[
            "controller_team_v5_3"
        ]
    )
    ==
    "A"

    and

    same_player_id(
        r650[
            "controller_id_v5_3"
        ],
        11,
    )
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V5.3 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Frames:",
    len(
        v5_3
    )
)

print(
    "FPS:",
    f"{fps:.2f}"
)


print("")

print(
    "FINAL POSSESSION"
)

print(
    v5_3[
        "possession_v5_3"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "FINAL BALL STATE"
)

print(
    v5_3[
        "ball_state_v5_3"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "Contest episodes:",
    len(
        contest_episodes
    )
)


print(
    "Contest episode frames:",
    int(
        v5_3[
            "contest_episode_active"
        ]
        .sum()
    )
)


print(
    "Frames changed from V5.2:",
    int(
        v5_3[
            "contest_changed_from_v5_2"
        ]
        .sum()
    )
)


# ============================================================
# REGRESSION SUMMARY
# ============================================================

print("")

print(
    "REGRESSION CHECKS"
)


print(
    "frame 159 | "
    "B long ball preserved:",
    (
        "PASS"
        if reg159
        else
        "FAIL"
    )
)


print(
    "frame 172 | "
    "contest confirmed:",
    (
        "PASS"
        if reg172
        else
        "FAIL"
    )
)


print(
    "frame 173 | "
    "contest carry:",
    (
        "PASS"
        if reg173
        else
        "FAIL"
    )
)


print(
    "frame 174 | "
    "contest carry:",
    (
        "PASS"
        if reg174
        else
        "FAIL"
    )
)


print(
    "frame 175 | "
    "false controller remains blocked:",
    (
        "PASS"
        if reg175
        else
        "FAIL"
    )
)


print(
    "frame 176 | "
    "false controller remains blocked:",
    (
        "PASS"
        if reg176
        else
        "FAIL"
    )
)


print(
    "frame 177 | "
    "contest resolves -> V5.2:",
    (
        "PASS"
        if reg177
        else
        "FAIL"
    )
)


print(
    "frame 203 | "
    "reception/bounce preserved:",
    (
        "PASS"
        if reg203
        else
        "FAIL"
    )
)


print(
    "frame 226 | "
    "false airborne controller preserved as Uncontrolled:",
    (
        "PASS"
        if reg226
        else
        "FAIL"
    )
)


print(
    "frame 253 | "
    "existing aerial contest preserved:",
    (
        "PASS"
        if reg253
        else
        "FAIL"
    )
)


print(
    "frame 281 | "
    "B#10 AirTouch preserved:",
    (
        "PASS"
        if reg281
        else
        "FAIL"
    )
)


print(
    "frame 287 | "
    "post-contest fallback preserved:",
    (
        "PASS"
        if reg287
        else
        "FAIL"
    )
)


print(
    "frame 650 | "
    "valid A#11 dribble preserved:",
    (
        "PASS"
        if reg650
        else
        "FAIL"
    )
)


# ============================================================
# DETAILED FRAMES
# ============================================================

for frame in [
    FRAME_159,

    FRAME_172,
    FRAME_173,
    FRAME_174,
    FRAME_175,
    FRAME_176,
    FRAME_177,

    FRAME_203,
    FRAME_226,
    FRAME_253,
    FRAME_281,
    FRAME_287,
    FRAME_650,
]:

    print_frame(
        frame
    )


# ============================================================
# CONTEST EPISODES TABLE
# ============================================================

print("")
print(
    "========================================"
)

print(
    "CONTEST EPISODES"
)

print(
    "========================================"
)

print("")


if len(
    contest_episodes
) > 0:

    print(
        contest_episodes
        .to_string(
            index=False
        )
    )


# ============================================================
# WINDOW
# ============================================================

print("")
print(
    "========================================"
)

print(
    "WINDOW | 6.72s - 7.20s"
)

print(
    "========================================"
)

print("")


window = v5_3[
    (
        v5_3[
            "frame"
        ]
        >=
        168
    )

    &

    (
        v5_3[
            "frame"
        ]
        <=
        180
    )
][
    [
        "frame",
        "time_sec",

        "possession_v5_2",
        "possession_v5_3",

        "ball_state_v5_2",
        "ball_state_v5_3",

        "controller_team_v5_2",
        "controller_id_v5_2",

        "controller_team_v5_3",
        "controller_id_v5_3",

        "contest_episode_active",
        "contest_episode_phase",

        "source_v5_3",
    ]
]


print(
    window
    .to_string(
        index=False
    )
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
    "Summary CSV:",
    OUTPUT_SUMMARY_CSV
)

print(
    "Audit CSV:",
    OUTPUT_AUDIT_CSV
)

print(
    "Contest Episodes CSV:",
    OUTPUT_EPISODES_CSV
)
