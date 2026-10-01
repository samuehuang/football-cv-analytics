import os

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

POSSESSION_V5_3_CSV = "outputs/possession_v5_3_frames.csv"

OUTPUT_FRAMES_CSV = "outputs/possession_v5_4_frames.csv"
OUTPUT_SUMMARY_CSV = "outputs/possession_v5_4_summary.csv"
OUTPUT_AUDIT_CSV = "outputs/possession_v5_4_audit.csv"
OUTPUT_RESOLUTION_CSV = "outputs/possession_v5_4_post_contest_resolution.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# POST-CONTEST QUARANTINE
#
# Contest ends:
#
#   DO NOT immediately restore baseline ownership.
#
# Wait for fresh trustworthy evidence:
#
#   1. Confirmed AirTouch
#   2. Controlled + Controller Gate KEEP
#
# Bound the quarantine so it cannot create a catastrophic
# long Unknown sequence.
# ============================================================

MAX_POST_CONTEST_QUARANTINE_FRAMES = 20


# ============================================================
# REGRESSION FRAMES
# ============================================================

FRAME_159 = 159

FRAME_172 = 172
FRAME_175 = 175
FRAME_176 = 176

FRAME_177 = 177
FRAME_178 = 178

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


def as_bool(value):

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


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


def same_player(
    team_a,
    id_a,
    team_b,
    id_b,
):

    return (

        clean_team(team_a)
        ==
        clean_team(team_b)

        and

        clean_team(team_a)
        is not None

        and

        same_player_id(
            id_a,
            id_b,
        )
    )


# ============================================================
# LOAD
# ============================================================

print("")
print("Loading Possession V5.3...")

df = pd.read_csv(
    POSSESSION_V5_3_CSV
)


# ============================================================
# NORMALIZE
# ============================================================

numeric_columns = [
    "frame",
    "time_sec",

    "controller_id_v5",
    "controller_id_v5_2",
    "controller_id_v5_3",

    "last_touch_id_v5",
    "last_touch_id_v5_2",
    "last_touch_id_v5_3",

    "ball_speed_mps",
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


df = df.dropna(
    subset=[
        "frame"
    ]
).copy()


df["frame"] = (
    df["frame"]
    .astype(int)
)


df = (
    df
    .sort_values("frame")
    .reset_index(drop=True)
)


# ============================================================
# FPS
# ============================================================

fps = 25.0


if (
    "time_sec"
    in df.columns
):

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


    if len(timing) >= 2:

        fd = (
            timing["frame"]
            .diff()
        )

        td = (
            timing["time_sec"]
            .diff()
        )


        valid = (

            fd.notna()

            &

            td.notna()

            &

            (fd > 0)

            &

            (td > 0)
        )


        values = (

            fd[valid]
            /
            td[valid]
        )


        values = values[
            np.isfinite(values)
        ]


        if len(values) > 0:

            fps = float(
                values.median()
            )


print(
    f"Estimated FPS: {fps:.2f}"
)


# ============================================================
# TRUSTED RESOLUTION
# ============================================================

def get_trusted_resolution(
    row,
):

    # --------------------------------------------------------
    # 1. Confirmed AirTouch
    # --------------------------------------------------------

    source = clean_text(
        row.get(
            "source_v5_3",
            None,
        ),
        "",
    )


    last_touch_type = clean_text(
        row.get(
            "last_touch_type_v5_3",
            None,
        ),
        "",
    )


    last_touch_team = clean_team(
        row.get(
            "last_touch_team_v5_3",
            None,
        )
    )


    last_touch_id = clean_id(
        row.get(
            "last_touch_id_v5_3",
            np.nan,
        )
    )


    if (

        (
            source
            ==
            "confirmed_air_touch"
        )

        or

        (
            last_touch_type
            ==
            "AirTouch"
        )

    ):

        if (
            last_touch_team
            is not None

            and

            pd.notna(
                last_touch_id
            )
        ):

            return {
                "resolved":
                    True,

                "resolution_type":
                    "confirmed_air_touch",

                "team":
                    last_touch_team,

                "track_id":
                    last_touch_id,
            }


    # --------------------------------------------------------
    # 2. Trusted ground control
    #
    # Controlled alone is NOT enough.
    # It must have survived Controller Gate V2.
    # --------------------------------------------------------

    state = clean_text(
        row.get(
            "ball_state_v5_3",
            None,
        ),
        "",
    )


    controller_team = clean_team(
        row.get(
            "controller_team_v5_3",
            None,
        )
    )


    controller_id = clean_id(
        row.get(
            "controller_id_v5_3",
            np.nan,
        )
    )


    gate_status = clean_text(
        row.get(
            "controller_gate_status",
            None,
        ),
        "",
    )


    if (

        state
        ==
        "Controlled"

        and

        controller_team
        is not None

        and

        pd.notna(
            controller_id
        )

        and

        gate_status
        ==
        "KEEP"

    ):

        return {
            "resolved":
                True,

            "resolution_type":
                "trusted_ground_control",

            "team":
                controller_team,

            "track_id":
                controller_id,
        }


    return {
        "resolved":
            False,

        "resolution_type":
            None,

        "team":
            None,

        "track_id":
            np.nan,
    }


# ============================================================
# STATE
# ============================================================

rows = []
audit_rows = []
resolution_rows = []


previous_contest_active = False

quarantine_active = False
quarantine_age = 0

quarantine_start_frame = None
quarantine_origin_end_frame = None

stale_team = None
stale_id = np.nan

stale_last_touch_block_active = False

contest_rejected_candidates = []


# ============================================================
# MAIN LOOP
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V5.4"
)

print(
    "Post-Contest Resolution Gate"
)

print(
    "========================================"
)

print("")


for _, base in df.iterrows():

    frame = int(
        base[
            "frame"
        ]
    )


    row = base.to_dict()


    # ========================================================
    # V5.3 BASE
    # ========================================================

    possession_v5_3 = clean_text(
        base.get(
            "possession_v5_3",
            None,
        ),
        "Unknown",
    )


    ball_state_v5_3 = clean_text(
        base.get(
            "ball_state_v5_3",
            None,
        ),
        "Unknown",
    )


    controller_team_v5_3 = clean_team(
        base.get(
            "controller_team_v5_3",
            None,
        )
    )


    controller_id_v5_3 = clean_id(
        base.get(
            "controller_id_v5_3",
            np.nan,
        )
    )


    last_touch_team_v5_3 = clean_team(
        base.get(
            "last_touch_team_v5_3",
            None,
        )
    )


    last_touch_id_v5_3 = clean_id(
        base.get(
            "last_touch_id_v5_3",
            np.nan,
        )
    )


    last_touch_type_v5_3 = clean_text(
        base.get(
            "last_touch_type_v5_3",
            None,
        ),
        None,
    )


    source_v5_3 = clean_text(
        base.get(
            "source_v5_3",
            None,
        ),
        "baseline_v4",
    )


    contest_active = as_bool(
        base.get(
            "contest_episode_active",
            False,
        )
    )


    # ========================================================
    # DEFAULT V5.4 = V5.3
    # ========================================================

    possession_v5_4 = (
        possession_v5_3
    )


    ball_state_v5_4 = (
        ball_state_v5_3
    )


    controller_team_v5_4 = (
        controller_team_v5_3
    )


    controller_id_v5_4 = (
        controller_id_v5_3
    )


    last_touch_team_v5_4 = (
        last_touch_team_v5_3
    )


    last_touch_id_v5_4 = (
        last_touch_id_v5_3
    )


    last_touch_type_v5_4 = (
        last_touch_type_v5_3
    )


    source_v5_4 = (
        source_v5_3
    )


    post_contest_active = False
    post_contest_age = 0
    post_contest_resolution = None
    stale_last_touch_blocked = False


    # ========================================================
    # WHILE INSIDE CONTEST:
    #
    # Capture rejected controllers that must never simply
    # resurrect afterward as a fake LastTouch.
    # ========================================================

    if contest_active:

        gate_status = clean_text(
            base.get(
                "controller_gate_status",
                None,
            ),
            None,
        )


        gate_reason = clean_text(
            base.get(
                "controller_gate_reason",
                None,
            ),
            None,
        )


        original_controller_team = clean_team(
            base.get(
                "controller_team_v5",
                None,
            )
        )


        original_controller_id = clean_id(
            base.get(
                "controller_id_v5",
                np.nan,
            )
        )


        if (

            gate_status
            ==
            "REJECT"

            and

            gate_reason
            ==
            "contest_lock"

            and

            original_controller_team
            is not None

            and

            pd.notna(
                original_controller_id
            )

        ):

            candidate = (
                original_controller_team,
                float(
                    original_controller_id
                ),
            )


            if (
                candidate
                not in
                contest_rejected_candidates
            ):

                contest_rejected_candidates.append(
                    candidate
                )


    # ========================================================
    # CONTEST JUST ENDED
    # ========================================================

    contest_just_ended = (

        previous_contest_active

        and

        not contest_active
    )


    if contest_just_ended:

        quarantine_active = True

        quarantine_age = 0

        quarantine_start_frame = frame

        quarantine_origin_end_frame = (
            frame - 1
        )


        # Pick most recent rejected controller.
        if (
            len(
                contest_rejected_candidates
            )
            >
            0
        ):

            stale_team = (
                contest_rejected_candidates[
                    -1
                ][0]
            )


            stale_id = float(
                contest_rejected_candidates[
                    -1
                ][1]
            )


            stale_last_touch_block_active = True


        else:

            stale_team = None
            stale_id = np.nan


        contest_rejected_candidates = []


    # ========================================================
    # IF ANOTHER CONTEST STARTS:
    #
    # Pause post-contest quarantine.
    # A new contest takes semantic priority.
    # ========================================================

    if contest_active:

        quarantine_active = False
        quarantine_age = 0


    # ========================================================
    # POST-CONTEST RESOLUTION
    # ========================================================

    if (
        quarantine_active

        and

        not contest_active
    ):

        resolution = (
            get_trusted_resolution(
                base
            )
        )


        # ====================================================
        # RESOLVED BY NEW TRUSTED EVIDENCE
        # ====================================================

        if (
            resolution[
                "resolved"
            ]
        ):

            post_contest_resolution = (
                resolution[
                    "resolution_type"
                ]
            )


            resolution_rows.append(
                {
                    "contest_end_frame":
                        quarantine_origin_end_frame,

                    "quarantine_start_frame":
                        quarantine_start_frame,

                    "resolution_frame":
                        frame,

                    "resolution_time_sec":
                        base.get(
                            "time_sec",
                            frame / fps,
                        ),

                    "wait_frames":
                        quarantine_age,

                    "resolution_type":
                        resolution[
                            "resolution_type"
                        ],

                    "resolution_team":
                        resolution[
                            "team"
                        ],

                    "resolution_track_id":
                        resolution[
                            "track_id"
                        ],

                    "timed_out":
                        False,
                }
            )


            quarantine_active = False
            quarantine_age = 0

            stale_last_touch_block_active = False

            stale_team = None
            stale_id = np.nan


            # Preserve the trustworthy V5.3 frame.
            source_v5_4 = (

                "post_contest_resolved_"
                +
                str(
                    resolution[
                        "resolution_type"
                    ]
                )
            )


        # ====================================================
        # STILL UNRESOLVED
        # ====================================================

        else:

            quarantine_age += 1

            post_contest_active = True
            post_contest_age = quarantine_age


            # ------------------------------------------------
            # Keep physical ball state where sensible.
            #
            # What we are invalidating here is OWNERSHIP,
            # not the fact that the ball can be InTransit,
            # Loose, Missing, etc.
            # ------------------------------------------------

            if (
                ball_state_v5_3
                ==
                "AerialContest"
            ):

                possession_v5_4 = (
                    "Contested"
                )

                ball_state_v5_4 = (
                    "AerialContest"
                )


            elif (
                ball_state_v5_3
                ==
                "Controlled"
            ):

                # A control not accepted by Gate KEEP cannot
                # resolve the contest.
                possession_v5_4 = (
                    "Unknown"
                )

                ball_state_v5_4 = (
                    "Uncontrolled"
                )


            else:

                possession_v5_4 = (
                    "Unknown"
                )

                ball_state_v5_4 = (
                    ball_state_v5_3
                )


            controller_team_v5_4 = None
            controller_id_v5_4 = np.nan


            last_touch_team_v5_4 = None
            last_touch_id_v5_4 = np.nan
            last_touch_type_v5_4 = (
                "PostContestUnknown"
            )


            source_v5_4 = (
                "post_contest_quarantine"
            )


            # =================================================
            # TIMEOUT
            #
            # Stop changing possession after the bounded
            # quarantine window.
            #
            # BUT stale LastTouch remains blocked separately.
            # =================================================

            if (
                quarantine_age
                >=
                MAX_POST_CONTEST_QUARANTINE_FRAMES
            ):

                resolution_rows.append(
                    {
                        "contest_end_frame":
                            quarantine_origin_end_frame,

                        "quarantine_start_frame":
                            quarantine_start_frame,

                        "resolution_frame":
                            frame,

                        "resolution_time_sec":
                            base.get(
                                "time_sec",
                                frame / fps,
                            ),

                        "wait_frames":
                            quarantine_age,

                        "resolution_type":
                            "timeout",

                        "resolution_team":
                            None,

                        "resolution_track_id":
                            np.nan,

                        "timed_out":
                            True,
                    }
                )


                quarantine_active = False


    # ========================================================
    # STALE LAST TOUCH BLOCK
    #
    # This survives the short ownership quarantine timeout.
    #
    # It prevents:
    #
    #   rejected A#3 controller
    #       ->
    #   LastTouch A#3 (Controlled)
    #
    # from magically returning later.
    #
    # Fresh trusted AirTouch / KEEP control clears the block.
    # ========================================================

    if (
        stale_last_touch_block_active

        and

        not contest_active
    ):

        fresh_resolution = (
            get_trusted_resolution(
                base
            )
        )


        if (
            fresh_resolution[
                "resolved"
            ]
        ):

            stale_last_touch_block_active = False

            stale_team = None
            stale_id = np.nan


        else:

            stale_is_current_last_touch = (

                same_player(
                    last_touch_team_v5_4,
                    last_touch_id_v5_4,
                    stale_team,
                    stale_id,
                )

                and

                clean_text(
                    last_touch_type_v5_4,
                    None,
                )
                ==
                "Controlled"
            )


            if stale_is_current_last_touch:

                last_touch_team_v5_4 = None
                last_touch_id_v5_4 = np.nan

                last_touch_type_v5_4 = (
                    "StaleControlledTouchBlocked"
                )


                stale_last_touch_blocked = True


                if (
                    source_v5_4
                    !=
                    "post_contest_quarantine"
                ):

                    source_v5_4 = (
                        "post_contest_stale_last_touch_block"
                    )


    # ========================================================
    # WRITE V5.4
    # ========================================================

    row[
        "possession_v5_4"
    ] = possession_v5_4


    row[
        "ball_state_v5_4"
    ] = ball_state_v5_4


    row[
        "controller_team_v5_4"
    ] = controller_team_v5_4


    row[
        "controller_id_v5_4"
    ] = controller_id_v5_4


    row[
        "last_touch_team_v5_4"
    ] = last_touch_team_v5_4


    row[
        "last_touch_id_v5_4"
    ] = last_touch_id_v5_4


    row[
        "last_touch_type_v5_4"
    ] = last_touch_type_v5_4


    row[
        "source_v5_4"
    ] = source_v5_4


    row[
        "post_contest_quarantine_active"
    ] = post_contest_active


    row[
        "post_contest_quarantine_age"
    ] = post_contest_age


    row[
        "post_contest_resolution"
    ] = post_contest_resolution


    row[
        "stale_last_touch_block_active"
    ] = stale_last_touch_block_active


    row[
        "stale_last_touch_blocked"
    ] = stale_last_touch_blocked


    row[
        "stale_rejected_team"
    ] = stale_team


    row[
        "stale_rejected_id"
    ] = stale_id


    # ========================================================
    # CHANGED?
    # ========================================================

    changed = (

        str(
            possession_v5_4
        )
        !=
        str(
            possession_v5_3
        )

        or

        str(
            ball_state_v5_4
        )
        !=
        str(
            ball_state_v5_3
        )

        or

        clean_team(
            controller_team_v5_4
        )
        !=
        clean_team(
            controller_team_v5_3
        )

        or

        (
            pd.notna(
                controller_id_v5_4
            )
            !=
            pd.notna(
                controller_id_v5_3
            )
        )

        or

        clean_team(
            last_touch_team_v5_4
        )
        !=
        clean_team(
            last_touch_team_v5_3
        )

        or

        (
            pd.notna(
                last_touch_id_v5_4
            )
            !=
            pd.notna(
                last_touch_id_v5_3
            )
        )
    )


    row[
        "changed_from_v5_3"
    ] = changed


    rows.append(
        row
    )


    if changed:

        audit_rows.append(
            {
                "frame":
                    frame,

                "time_sec":
                    base.get(
                        "time_sec",
                        frame / fps,
                    ),

                "possession_v5_3":
                    possession_v5_3,

                "possession_v5_4":
                    possession_v5_4,

                "ball_state_v5_3":
                    ball_state_v5_3,

                "ball_state_v5_4":
                    ball_state_v5_4,

                "controller_team_v5_3":
                    controller_team_v5_3,

                "controller_id_v5_3":
                    controller_id_v5_3,

                "controller_team_v5_4":
                    controller_team_v5_4,

                "controller_id_v5_4":
                    controller_id_v5_4,

                "last_touch_team_v5_3":
                    last_touch_team_v5_3,

                "last_touch_id_v5_3":
                    last_touch_id_v5_3,

                "last_touch_type_v5_3":
                    last_touch_type_v5_3,

                "last_touch_team_v5_4":
                    last_touch_team_v5_4,

                "last_touch_id_v5_4":
                    last_touch_id_v5_4,

                "last_touch_type_v5_4":
                    last_touch_type_v5_4,

                "quarantine_active":
                    post_contest_active,

                "quarantine_age":
                    post_contest_age,

                "stale_last_touch_blocked":
                    stale_last_touch_blocked,

                "source_v5_4":
                    source_v5_4,
            }
        )


    previous_contest_active = (
        contest_active
    )


# ============================================================
# OUTPUT DATAFRAMES
# ============================================================

v5_4 = pd.DataFrame(
    rows
)


audit = pd.DataFrame(
    audit_rows
)


resolution_df = pd.DataFrame(
    resolution_rows
)


v5_4.to_csv(
    OUTPUT_FRAMES_CSV,
    index=False,
)


audit.to_csv(
    OUTPUT_AUDIT_CSV,
    index=False,
)


resolution_df.to_csv(
    OUTPUT_RESOLUTION_CSV,
    index=False,
)


# ============================================================
# SUMMARY CSV
# ============================================================

summary_rows = []


for key, value in (
    v5_4[
        "possession_v5_4"
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
                int(value),

            "seconds":
                float(value)
                /
                fps,
        }
    )


for key, value in (
    v5_4[
        "ball_state_v5_4"
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
                int(value),

            "seconds":
                float(value)
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
# LOOKUP
# ============================================================

by_frame = (
    v5_4
    .set_index(
        "frame"
    )
)


def get_frame(frame):

    if (
        frame
        not in
        by_frame.index
    ):

        return None


    result = by_frame.loc[
        frame
    ]


    if isinstance(
        result,
        pd.DataFrame
    ):

        result = result.iloc[0]


    return result


def no_controller(row):

    return (

        row is not None

        and

        clean_team(
            row.get(
                "controller_team_v5_4",
                None,
            )
        )
        is None
    )


def no_last_touch(row):

    return (

        row is not None

        and

        clean_team(
            row.get(
                "last_touch_team_v5_4",
                None,
            )
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
            "possession_v5_4"
        ]
    )
    ==
    "B"

    and

    str(
        r159[
            "ball_state_v5_4"
        ]
    )
    ==
    "InTransit"
)


r172 = get_frame(
    FRAME_172
)


reg172 = (

    r172 is not None

    and

    str(
        r172[
            "possession_v5_4"
        ]
    )
    ==
    "Contested"

    and

    str(
        r172[
            "ball_state_v5_4"
        ]
    )
    ==
    "AerialContest"

    and

    no_controller(
        r172
    )
)


r175 = get_frame(
    FRAME_175
)


reg175 = (

    r175 is not None

    and

    str(
        r175[
            "possession_v5_4"
        ]
    )
    ==
    "Contested"

    and

    str(
        r175[
            "ball_state_v5_4"
        ]
    )
    ==
    "AerialContest"

    and

    no_controller(
        r175
    )
)


r176 = get_frame(
    FRAME_176
)


reg176 = (

    r176 is not None

    and

    str(
        r176[
            "possession_v5_4"
        ]
    )
    ==
    "Contested"

    and

    no_controller(
        r176
    )
)


# ------------------------------------------------------------
# NEW REGRESSION:
# 177 / 178 must no longer inherit false A#3 ownership.
# ------------------------------------------------------------

r177 = get_frame(
    FRAME_177
)


reg177 = (

    r177 is not None

    and

    str(
        r177[
            "possession_v5_4"
        ]
    )
    ==
    "Unknown"

    and

    str(
        r177[
            "ball_state_v5_4"
        ]
    )
    ==
    "InTransit"

    and

    no_controller(
        r177
    )

    and

    no_last_touch(
        r177
    )
)


r178 = get_frame(
    FRAME_178
)


reg178 = (

    r178 is not None

    and

    str(
        r178[
            "possession_v5_4"
        ]
    )
    ==
    "Unknown"

    and

    str(
        r178[
            "ball_state_v5_4"
        ]
    )
    ==
    "InTransit"

    and

    no_controller(
        r178
    )

    and

    no_last_touch(
        r178
    )
)


# ------------------------------------------------------------
# Existing regression suite
# ------------------------------------------------------------

r203 = get_frame(
    FRAME_203
)


reg203 = (

    r203 is not None

    and

    str(
        r203[
            "ball_state_v5_4"
        ]
    )
    ==
    str(
        r203[
            "ball_state_v5_3"
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
            "ball_state_v5_4"
        ]
    )
    ==
    "Uncontrolled"

    and

    no_controller(
        r226
    )
)


r253 = get_frame(
    FRAME_253
)


reg253 = (

    r253 is not None

    and

    str(
        r253[
            "possession_v5_4"
        ]
    )
    ==
    "Contested"

    and

    str(
        r253[
            "ball_state_v5_4"
        ]
    )
    ==
    "AerialContest"
)


r281 = get_frame(
    FRAME_281
)


reg281 = (

    r281 is not None

    and

    str(
        r281[
            "possession_v5_4"
        ]
    )
    ==
    "B"

    and

    str(
        r281[
            "ball_state_v5_4"
        ]
    )
    ==
    "InTransit"

    and

    clean_team(
        r281[
            "last_touch_team_v5_4"
        ]
    )
    ==
    "B"

    and

    same_player_id(
        r281[
            "last_touch_id_v5_4"
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
            "possession_v5_4"
        ]
    )
    ==
    str(
        r287[
            "possession_v5_3"
        ]
    )

    and

    str(
        r287[
            "ball_state_v5_4"
        ]
    )
    ==
    str(
        r287[
            "ball_state_v5_3"
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
            "possession_v5_4"
        ]
    )
    ==
    "A"

    and

    str(
        r650[
            "ball_state_v5_4"
        ]
    )
    ==
    "Controlled"

    and

    clean_team(
        r650[
            "controller_team_v5_4"
        ]
    )
    ==
    "A"

    and

    same_player_id(
        r650[
            "controller_id_v5_4"
        ],
        11,
    )
)


# ============================================================
# PRINT
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V5.4 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Frames:",
    len(
        v5_4
    )
)


print("")

print(
    "FINAL POSSESSION"
)

print(
    v5_4[
        "possession_v5_4"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "FINAL BALL STATE"
)

print(
    v5_4[
        "ball_state_v5_4"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "Frames changed from V5.3:",
    int(
        v5_4[
            "changed_from_v5_3"
        ]
        .sum()
    )
)


print(
    "Post-contest quarantine frames:",
    int(
        v5_4[
            "post_contest_quarantine_active"
        ]
        .sum()
    )
)


print(
    "Stale LastTouch blocks:",
    int(
        v5_4[
            "stale_last_touch_blocked"
        ]
        .sum()
    )
)


# ============================================================
# RESOLUTION
# ============================================================

print("")

print(
    "========================================"
)

print(
    "POST-CONTEST RESOLUTION"
)

print(
    "========================================"
)

print("")


if len(
    resolution_df
) > 0:

    print(
        resolution_df
        .to_string(
            index=False
        )
    )


else:

    print(
        "No post-contest resolution records."
    )


# ============================================================
# REGRESSION
# ============================================================

print("")

print(
    "REGRESSION CHECKS"
)


checks = [
    (
        "159 | B long ball preserved",
        reg159,
    ),

    (
        "172 | physical contest preserved",
        reg172,
    ),

    (
        "175 | false controller blocked",
        reg175,
    ),

    (
        "176 | contest still blocked",
        reg176,
    ),

    (
        "177 | post-contest ownership Unknown",
        reg177,
    ),

    (
        "178 | airborne InTransit / no stale A#3",
        reg178,
    ),

    (
        "203 | reception/bounce state preserved",
        reg203,
    ),

    (
        "226 | false airborne controller preserved",
        reg226,
    ),

    (
        "253 | aerial contest preserved",
        reg253,
    ),

    (
        "281 | B#10 AirTouch preserved",
        reg281,
    ),

    (
        "287 | later fallback preserved",
        reg287,
    ),

    (
        "650 | valid A#11 dribble preserved",
        reg650,
    ),
]


for label, passed in checks:

    print(
        f"frame {label}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


# ============================================================
# IMPORTANT WINDOW
# ============================================================

print("")

print(
    "========================================"
)

print(
    "WINDOW | 6.80s - 7.60s"
)

print(
    "========================================"
)

print("")


window = v5_4[
    (
        v5_4[
            "frame"
        ]
        >=
        170
    )

    &

    (
        v5_4[
            "frame"
        ]
        <=
        190
    )
][
    [
        "frame",
        "time_sec",

        "possession_v5_3",
        "possession_v5_4",

        "ball_state_v5_3",
        "ball_state_v5_4",

        "controller_team_v5_4",
        "controller_id_v5_4",

        "last_touch_team_v5_4",
        "last_touch_id_v5_4",
        "last_touch_type_v5_4",

        "post_contest_quarantine_active",
        "post_contest_quarantine_age",

        "controller_gate_status",
        "controller_gate_reason",

        "source_v5_4",
    ]
]


print(
    window
    .to_string(
        index=False
    )
)


# ============================================================
# CHANGED FRAMES
# ============================================================

print("")

print(
    "========================================"
)

print(
    "CHANGED FRAMES"
)

print(
    "========================================"
)

print("")


if len(audit) > 0:

    print(
        audit[
            [
                "frame",
                "time_sec",

                "possession_v5_3",
                "possession_v5_4",

                "ball_state_v5_3",
                "ball_state_v5_4",

                "last_touch_team_v5_3",
                "last_touch_id_v5_3",

                "last_touch_team_v5_4",
                "last_touch_id_v5_4",

                "quarantine_active",
                "quarantine_age",

                "source_v5_4",
            ]
        ]
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
    "Resolution CSV:",
    OUTPUT_RESOLUTION_CSV
)
