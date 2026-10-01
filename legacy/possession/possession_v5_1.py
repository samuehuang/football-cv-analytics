import os

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

POSSESSION_V4_CSV = "outputs/possession_v4_frames.csv"
AIR_TOUCH_V4_CSV = "outputs/air_touch_v4_events.csv"

OUTPUT_FRAMES_CSV = "outputs/possession_v5_1_frames.csv"
OUTPUT_SUMMARY_CSV = "outputs/possession_v5_1_summary.csv"
OUTPUT_AUDIT_CSV = "outputs/possession_v5_1_audit.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# AIR TOUCH CARRY
# ============================================================

# 50 frames @ 25fps ≈ 2 seconds
MAX_AIR_TOUCH_CARRY_FRAMES = 50

# Do not carry LastTouch ownership through long missing-ball
# periods.
MAX_AIR_TOUCH_MISSING_CARRY_FRAMES = 8


# ============================================================
# REGRESSION FRAMES
# ============================================================

FRAME_159 = 159
FRAME_172 = 172
FRAME_203 = 203
FRAME_253 = 253
FRAME_281 = 281
FRAME_287 = 287


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


def clean_source(value):

    if pd.isna(value):
        return "baseline_v4"

    text = str(value).strip()

    if (
        not text
        or
        text.lower()
        in {
            "none",
            "nan",
        }
    ):
        return "baseline_v4"

    return text


def same_player_id(a, b):

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
# LOAD DATA
# ============================================================

print("")
print("Loading Possession V4...")

v4 = pd.read_csv(
    POSSESSION_V4_CSV
)


print("Loading AirTouch V4...")

events = pd.read_csv(
    AIR_TOUCH_V4_CSV
)


# ============================================================
# NORMALIZE V4
# ============================================================

v4["frame"] = pd.to_numeric(
    v4["frame"],
    errors="coerce"
)


v4 = v4.dropna(
    subset=[
        "frame"
    ]
).copy()


v4["frame"] = (
    v4["frame"]
    .astype(int)
)


for column in [
    "time_sec",
    "controller_id_v4",
]:

    if column in v4.columns:

        v4[column] = pd.to_numeric(
            v4[column],
            errors="coerce"
        )


v4 = (
    v4
    .sort_values(
        "frame"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# ESTIMATE FPS
# ============================================================

fps = 25.0


if (
    "time_sec"
    in
    v4.columns
):

    timing = (
        v4[
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

            values = (

                frame_diff[
                    valid
                ]

                /

                time_diff[
                    valid
                ]
            )


            values = values[
                np.isfinite(
                    values
                )
            ]


            if len(values) > 0:

                fps = float(
                    values.median()
                )


print(
    f"Estimated FPS: {fps:.2f}"
)


# ============================================================
# NORMALIZE AIR TOUCH V4 EVENTS
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
            errors="coerce"
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


# ============================================================
# ONLY HIGH-CONFIDENCE PHYSICAL EVENTS OVERRIDE V4
#
# AirTouch
#   -> LastTouch known
#   -> can change travelling-phase ownership
#
# AerialContest
#   -> possession contested
#   -> LastTouch unknown
#
# ReceptionOrBounce / WeakInteraction
#   -> do NOT override V4
# ============================================================

strong_events = events[
    events[
        "final_class"
    ].isin(
        [
            "AirTouch",
            "AerialContest",
        ]
    )
].copy()


# ============================================================
# EVENT LOOKUP BY FRAME
# ============================================================

event_by_frame = {}


for _, event in strong_events.iterrows():

    start_frame = int(
        event[
            "start_frame"
        ]
    )

    end_frame = int(
        event[
            "end_frame"
        ]
    )


    for frame in range(
        start_frame,
        end_frame + 1
    ):

        if (
            frame
            not in
            event_by_frame
        ):

            event_by_frame[
                frame
            ] = event


        else:

            existing = (
                event_by_frame[
                    frame
                ]
            )


            # AerialContest has priority over AirTouch
            # when physical events overlap.
            if (
                str(
                    event[
                        "final_class"
                    ]
                )
                ==
                "AerialContest"
            ):

                event_by_frame[
                    frame
                ] = event


# ============================================================
# STATE MEMORY
# ============================================================

# Team that produced the latest confirmed AirTouch.
#
# This is NOT controller memory.
# It only represents phase ownership while the ball travels.
air_owner_team = None
air_owner_id = np.nan
air_owner_start_frame = None


# LastTouch memory
last_touch_team = None
last_touch_id = np.nan
last_touch_type = None


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
    "POSSESSION V5.1"
)

print(
    "Conservative LastTouch integration"
)

print(
    "========================================"
)

print("")


# ============================================================
# MAIN LOOP
# ============================================================

for _, base in v4.iterrows():

    frame = int(
        base[
            "frame"
        ]
    )


    row = base.to_dict()


    # ========================================================
    # V4 BASELINE
    # ========================================================

    base_possession = str(
        base.get(
            "possession_v4",
            "Unknown"
        )
    )


    base_ball_state = str(
        base.get(
            "ball_state_v4",
            "Unknown"
        )
    )


    base_controller_team = clean_team(
        base.get(
            "controller_team_v4",
            None
        )
    )


    base_controller_id = clean_id(
        base.get(
            "controller_id_v4",
            np.nan
        )
    )


    base_source = clean_source(
        base.get(
            "source_v4",
            None
        )
    )


    has_controller = (

        base_controller_team
        in
        {
            "A",
            "B",
        }

        and

        pd.notna(
            base_controller_id
        )
    )


    # ========================================================
    # DEFAULT V5 = V4
    # ========================================================

    possession_v5 = (
        base_possession
    )


    ball_state_v5 = (
        base_ball_state
    )


    controller_team_v5 = (
        base_controller_team
    )


    controller_id_v5 = (
        base_controller_id
    )


    source_v5 = (
        base_source
    )


    physical_event_class = None
    physical_event_id = np.nan
    physical_participants = None


    # ========================================================
    # PHYSICAL EVENT AT CURRENT FRAME
    # ========================================================

    event = event_by_frame.get(
        frame,
        None
    )


    # ========================================================
    # PRIORITY 1
    #
    # CONFIRMED PHYSICAL AERIAL CONTEST
    #
    # AerialContest only overrides the actual event.
    #
    # It does NOT create permanent Unknown possession.
    # ========================================================

    if (

        event is not None

        and

        str(
            event[
                "final_class"
            ]
        )
        ==
        "AerialContest"

    ):

        physical_event_class = (
            "AerialContest"
        )


        physical_event_id = event.get(
            "physical_event_id",
            np.nan
        )


        physical_participants = event.get(
            "participants",
            None
        )


        possession_v5 = (
            "Contested"
        )


        ball_state_v5 = (
            "AerialContest"
        )


        controller_team_v5 = None
        controller_id_v5 = np.nan


        source_v5 = (
            "physical_aerial_contest"
        )


        # ----------------------------------------------------
        # A new duel invalidates previous AirTouch ownership.
        # ----------------------------------------------------

        air_owner_team = None
        air_owner_id = np.nan
        air_owner_start_frame = None


        # ----------------------------------------------------
        # During the actual duel:
        # LastTouch is unknown.
        # ----------------------------------------------------

        last_touch_team = None
        last_touch_id = np.nan
        last_touch_type = (
            "AerialContestUnknown"
        )


    # ========================================================
    # PRIORITY 2
    #
    # CONFIRMED PHYSICAL AIR TOUCH
    # ========================================================

    elif (

        event is not None

        and

        str(
            event[
                "final_class"
            ]
        )
        ==
        "AirTouch"

    ):

        touch_team = clean_team(
            event.get(
                "last_touch_team",
                None
            )
        )


        touch_id = clean_id(
            event.get(
                "last_touch_track_id",
                np.nan
            )
        )


        physical_event_class = (
            "AirTouch"
        )


        physical_event_id = event.get(
            "physical_event_id",
            np.nan
        )


        physical_participants = event.get(
            "participants",
            None
        )


        if (
            touch_team
            in
            {
                "A",
                "B",
            }
        ):

            possession_v5 = (
                touch_team
            )


            # A header / aerial touch is not automatically
            # controlled possession.
            ball_state_v5 = (
                "InTransit"
            )


            controller_team_v5 = None
            controller_id_v5 = np.nan


            source_v5 = (
                "confirmed_air_touch"
            )


            # ------------------------------------------------
            # Start travelling-phase ownership.
            # ------------------------------------------------

            air_owner_team = (
                touch_team
            )


            air_owner_id = (
                touch_id
            )


            air_owner_start_frame = (
                frame
            )


            # ------------------------------------------------
            # LastTouch is known.
            # ------------------------------------------------

            last_touch_team = (
                touch_team
            )


            last_touch_id = (
                touch_id
            )


            last_touch_type = (
                "AirTouch"
            )


    # ========================================================
    # PRIORITY 3
    #
    # CONFIRMED V4 CONTROLLER
    #
    # Controlled possession is stronger than an older
    # travelling AirTouch.
    # ========================================================

    elif has_controller:

        possession_v5 = (
            base_controller_team
        )


        ball_state_v5 = (
            base_ball_state
        )


        controller_team_v5 = (
            base_controller_team
        )


        controller_id_v5 = (
            base_controller_id
        )


        source_v5 = (
            base_source
        )


        # ----------------------------------------------------
        # Controlled reception ends AirTouch carry.
        # ----------------------------------------------------

        air_owner_team = None
        air_owner_id = np.nan
        air_owner_start_frame = None


        # ----------------------------------------------------
        # Controller is also the latest known controlled touch.
        # ----------------------------------------------------

        last_touch_team = (
            base_controller_team
        )


        last_touch_id = (
            base_controller_id
        )


        last_touch_type = (
            "Controlled"
        )


    # ========================================================
    # PRIORITY 4
    #
    # V4 ITSELF SAYS CONTEST
    #
    # A contest terminates old AirTouch ownership.
    #
    # It only marks LastTouch unknown WHILE contest is active.
    # ========================================================

    elif (

        base_ball_state
        in
        {
            "AerialContest",
            "Contested",
        }

        or

        base_possession
        ==
        "Contested"

    ):

        possession_v5 = (
            base_possession
        )


        ball_state_v5 = (
            base_ball_state
        )


        controller_team_v5 = None
        controller_id_v5 = np.nan


        source_v5 = (
            base_source
        )


        # ----------------------------------------------------
        # Contest terminates previous AirTouch ownership.
        # ----------------------------------------------------

        air_owner_team = None
        air_owner_id = np.nan
        air_owner_start_frame = None


        # ----------------------------------------------------
        # Current contest:
        # final toucher unknown.
        # ----------------------------------------------------

        last_touch_team = None
        last_touch_id = np.nan
        last_touch_type = (
            "ContestUnknown"
        )


    # ========================================================
    # PRIORITY 5
    #
    # CARRY OWNERSHIP AFTER CONFIRMED AIR TOUCH
    # ========================================================

    elif (

        air_owner_team
        in
        {
            "A",
            "B",
        }

        and

        air_owner_start_frame
        is not None

    ):

        carry_age = (
            frame
            -
            air_owner_start_frame
        )


        carry_allowed = False


        # ----------------------------------------------------
        # Travelling / loose ball
        # ----------------------------------------------------

        if (
            base_ball_state
            in
            {
                "InTransit",
                "Loose",
            }
        ):

            carry_allowed = (

                carry_age
                <=
                MAX_AIR_TOUCH_CARRY_FRAMES
            )


        # ----------------------------------------------------
        # Missing ball:
        # only tolerate a short gap.
        # ----------------------------------------------------

        elif (
            base_ball_state
            ==
            "Missing"
        ):

            carry_allowed = (

                carry_age
                <=
                MAX_AIR_TOUCH_MISSING_CARRY_FRAMES
            )


        # ----------------------------------------------------
        # Other physical states:
        # stop AirTouch ownership.
        # ----------------------------------------------------

        else:

            carry_allowed = False


        if carry_allowed:

            possession_v5 = (
                air_owner_team
            )


            ball_state_v5 = (
                base_ball_state
            )


            controller_team_v5 = None
            controller_id_v5 = np.nan


            source_v5 = (
                "carried_after_air_touch"
            )


            # LastTouch stays the confirmed AirTouch player.


        else:

            # ------------------------------------------------
            # AirTouch carry has expired.
            # Return to V4.
            # ------------------------------------------------

            air_owner_team = None
            air_owner_id = np.nan
            air_owner_start_frame = None


            possession_v5 = (
                base_possession
            )


            ball_state_v5 = (
                base_ball_state
            )


            controller_team_v5 = (
                base_controller_team
            )


            controller_id_v5 = (
                base_controller_id
            )


            source_v5 = (
                base_source
            )


            # ------------------------------------------------
            # IMPORTANT:
            #
            # An expired AirTouch must NOT stay forever in
            # LastTouch display.
            # ------------------------------------------------

            if (
                last_touch_type
                ==
                "AirTouch"
            ):

                last_touch_team = None
                last_touch_id = np.nan
                last_touch_type = None


            # ------------------------------------------------
            # Likewise contest-unknown information is only
            # meaningful during the contest itself.
            # ------------------------------------------------

            if (
                last_touch_type
                in
                {
                    "ContestUnknown",
                    "AerialContestUnknown",
                }
            ):

                last_touch_team = None
                last_touch_id = np.nan
                last_touch_type = None


    # ========================================================
    # PRIORITY 6
    #
    # NORMAL V4 BASELINE
    #
    # IMPORTANT V5.1 FIX:
    #
    # ContestUnknown / AerialContestUnknown only describe
    # the contest frame itself.
    #
    # Once the contest has ended, clear that stale memory.
    # ========================================================

    else:

        possession_v5 = (
            base_possession
        )


        ball_state_v5 = (
            base_ball_state
        )


        controller_team_v5 = (
            base_controller_team
        )


        controller_id_v5 = (
            base_controller_id
        )


        source_v5 = (
            base_source
        )


        # ----------------------------------------------------
        # FIX:
        # Do not keep stale contest LastTouch information.
        # ----------------------------------------------------

        if (
            last_touch_type
            in
            {
                "ContestUnknown",
                "AerialContestUnknown",
            }
        ):

            last_touch_team = None
            last_touch_id = np.nan
            last_touch_type = None


    # ========================================================
    # SAVE V5 FIELDS
    # ========================================================

    row[
        "possession_v5"
    ] = possession_v5


    row[
        "ball_state_v5"
    ] = ball_state_v5


    row[
        "controller_team_v5"
    ] = controller_team_v5


    row[
        "controller_id_v5"
    ] = controller_id_v5


    row[
        "last_touch_team_v5"
    ] = last_touch_team


    row[
        "last_touch_id_v5"
    ] = last_touch_id


    row[
        "last_touch_type_v5"
    ] = last_touch_type


    row[
        "source_v5"
    ] = source_v5


    row[
        "physical_event_class_v5"
    ] = physical_event_class


    row[
        "physical_event_id_v5"
    ] = physical_event_id


    row[
        "physical_event_participants_v5"
    ] = physical_participants


    rows.append(
        row
    )


    # ========================================================
    # AUDIT
    # ========================================================

    changed = (

        str(
            possession_v5
        )
        !=
        str(
            base_possession
        )

        or

        str(
            ball_state_v5
        )
        !=
        str(
            base_ball_state
        )

        or

        clean_team(
            controller_team_v5
        )
        !=
        clean_team(
            base_controller_team
        )

        or

        physical_event_class
        is not None

        or

        source_v5
        ==
        "carried_after_air_touch"
    )


    if changed:

        audit_rows.append(
            {
                "frame":
                    frame,

                "time_sec":
                    base.get(
                        "time_sec",
                        frame / fps
                    ),

                "possession_v4":
                    base_possession,

                "possession_v5":
                    possession_v5,

                "ball_state_v4":
                    base_ball_state,

                "ball_state_v5":
                    ball_state_v5,

                "controller_team_v4":
                    base_controller_team,

                "controller_id_v4":
                    base_controller_id,

                "controller_team_v5":
                    controller_team_v5,

                "controller_id_v5":
                    controller_id_v5,

                "last_touch_team_v5":
                    last_touch_team,

                "last_touch_id_v5":
                    last_touch_id,

                "last_touch_type_v5":
                    last_touch_type,

                "source_v4":
                    base_source,

                "source_v5":
                    source_v5,

                "physical_event_class":
                    physical_event_class,

                "physical_event_id":
                    physical_event_id,

                "participants":
                    physical_participants,
            }
        )


# ============================================================
# DATAFRAMES
# ============================================================

v5 = pd.DataFrame(
    rows
)


audit = pd.DataFrame(
    audit_rows
)


v5.to_csv(
    OUTPUT_FRAMES_CSV,
    index=False
)


audit.to_csv(
    OUTPUT_AUDIT_CSV,
    index=False
)


# ============================================================
# SUMMARY CSV
# ============================================================

summary_rows = []


for key, value in (
    v5[
        "possession_v5"
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
    v5[
        "ball_state_v5"
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
    index=False
)


# ============================================================
# FRAME LOOKUP
# ============================================================

v5_by_frame = (
    v5
    .set_index(
        "frame"
    )
)


def get_frame(frame):

    if (
        frame
        not in
        v5_by_frame.index
    ):

        return None


    result = v5_by_frame.loc[
        frame
    ]


    if isinstance(
        result,
        pd.DataFrame
    ):

        result = result.iloc[0]


    return result


def print_frame(frame):

    r = get_frame(
        frame
    )


    print("")


    if r is None:

        print(
            f"frame {frame}: NOT FOUND"
        )

        return


    print(
        f"FRAME {frame} | "
        f"{r.get('time_sec', np.nan):.2f}s"
    )


    print(
        "  V4 | "
        f"possession={r.get('possession_v4')} | "
        f"state={r.get('ball_state_v4')} | "
        f"controller="
        f"{r.get('controller_team_v4')} "
        f"{r.get('controller_id_v4')}"
    )


    print(
        "  V5 | "
        f"possession={r.get('possession_v5')} | "
        f"state={r.get('ball_state_v5')} | "
        f"controller="
        f"{r.get('controller_team_v5')} "
        f"{r.get('controller_id_v5')} | "
        f"last_touch="
        f"{r.get('last_touch_team_v5')} "
        f"{r.get('last_touch_id_v5')} "
        f"({r.get('last_touch_type_v5')}) | "
        f"source={r.get('source_v5')}"
    )


# ============================================================
# REGRESSION TESTS
# ============================================================

# ------------------------------------------------------------
# FRAME 159
# ------------------------------------------------------------

r159 = get_frame(
    FRAME_159
)


reg159 = (

    r159 is not None

    and

    str(
        r159[
            "possession_v5"
        ]
    )
    ==
    "B"

    and

    str(
        r159[
            "ball_state_v5"
        ]
    )
    ==
    "InTransit"

    and

    clean_team(
        r159[
            "controller_team_v5"
        ]
    )
    is None

    and

    pd.isna(
        r159[
            "last_touch_id_v5"
        ]
    )

    and

    pd.isna(
        r159[
            "last_touch_team_v5"
        ]
    )
)


# ------------------------------------------------------------
# FRAME 172
# ------------------------------------------------------------

r172 = get_frame(
    FRAME_172
)


reg172 = (

    r172 is not None

    and

    str(
        r172[
            "possession_v5"
        ]
    )
    ==
    "Contested"

    and

    str(
        r172[
            "ball_state_v5"
        ]
    )
    ==
    "AerialContest"

    and

    clean_team(
        r172[
            "controller_team_v5"
        ]
    )
    is None

    and

    str(
        r172[
            "last_touch_type_v5"
        ]
    )
    ==
    "AerialContestUnknown"
)


# ------------------------------------------------------------
# FRAME 203
# ------------------------------------------------------------

r203 = get_frame(
    FRAME_203
)


reg203 = (

    r203 is not None

    and

    str(
        r203[
            "source_v5"
        ]
    )
    not in
    {
        "confirmed_air_touch",
        "carried_after_air_touch",
    }

    and

    str(
        r203[
            "possession_v5"
        ]
    )
    ==
    str(
        r203[
            "possession_v4"
        ]
    )

    and

    pd.isna(
        r203[
            "last_touch_team_v5"
        ]
    )

    and

    pd.isna(
        r203[
            "last_touch_id_v5"
        ]
    )
)


# ------------------------------------------------------------
# FRAME 253
# ------------------------------------------------------------

r253 = get_frame(
    FRAME_253
)


reg253 = (

    r253 is not None

    and

    str(
        r253[
            "possession_v5"
        ]
    )
    ==
    "Contested"

    and

    str(
        r253[
            "ball_state_v5"
        ]
    )
    ==
    "AerialContest"

    and

    clean_team(
        r253[
            "controller_team_v5"
        ]
    )
    is None

    and

    str(
        r253[
            "last_touch_type_v5"
        ]
    )
    ==
    "ContestUnknown"
)


# ------------------------------------------------------------
# FRAME 281
# ------------------------------------------------------------

r281 = get_frame(
    FRAME_281
)


reg281 = (

    r281 is not None

    and

    str(
        r281[
            "possession_v5"
        ]
    )
    ==
    "B"

    and

    str(
        r281[
            "ball_state_v5"
        ]
    )
    ==
    "InTransit"

    and

    clean_team(
        r281[
            "controller_team_v5"
        ]
    )
    is None

    and

    clean_team(
        r281[
            "last_touch_team_v5"
        ]
    )
    ==
    "B"

    and

    same_player_id(
        r281[
            "last_touch_id_v5"
        ],
        10
    )

    and

    str(
        r281[
            "last_touch_type_v5"
        ]
    )
    ==
    "AirTouch"
)


# ------------------------------------------------------------
# FRAME 287
# ------------------------------------------------------------

r287 = get_frame(
    FRAME_287
)


reg287 = (

    r287 is not None

    and

    str(
        r287[
            "possession_v5"
        ]
    )
    ==
    str(
        r287[
            "possession_v4"
        ]
    )

    and

    str(
        r287[
            "source_v5"
        ]
    )
    !=
    "unresolved_after_aerial_contest"

    and

    pd.isna(
        r287[
            "last_touch_team_v5"
        ]
    )

    and

    pd.isna(
        r287[
            "last_touch_id_v5"
        ]
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
    "POSSESSION V5.1 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Frames:",
    len(v5)
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
    v5[
        "possession_v5"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "FINAL BALL STATE"
)

print(
    v5[
        "ball_state_v5"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "V5 SOURCES"
)

print(
    v5[
        "source_v5"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "Changed / audited frames:",
    len(audit)
)


print("")

print(
    "REGRESSION CHECKS"
)


print(
    "frame 159 | "
    "B InTransit + clean LastTouch:",
    (
        "PASS"
        if reg159
        else
        "FAIL"
    )
)


print(
    "frame 172 | "
    "A#1 + B#14 -> AerialContest:",
    (
        "PASS"
        if reg172
        else
        "FAIL"
    )
)


print(
    "frame 203 | "
    "A#6 reception/bounce + no stale LastTouch:",
    (
        "PASS"
        if reg203
        else
        "FAIL"
    )
)


print(
    "frame 253 | "
    "aerial contest remains contested:",
    (
        "PASS"
        if reg253
        else
        "FAIL"
    )
)


print(
    "frame 281 | "
    "B#10 AirTouch -> B ownership:",
    (
        "PASS"
        if reg281
        else
        "FAIL"
    )
)


print(
    "frame 287 | "
    "contest ended -> V4 + clear LastTouch:",
    (
        "PASS"
        if reg287
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
    FRAME_203,
    FRAME_253,
    FRAME_281,
    FRAME_287,
]:

    print_frame(
        frame
    )


# ============================================================
# POST HEADER WINDOW
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POST-HEADER WINDOW"
)

print(
    "========================================"
)

print("")


window = v5[
    (
        v5[
            "frame"
        ]
        >=
        281
    )

    &

    (
        v5[
            "frame"
        ]
        <=
        301
    )
][
    [
        "frame",
        "time_sec",

        "possession_v4",
        "possession_v5",

        "ball_state_v5",

        "controller_team_v5",
        "controller_id_v5",

        "last_touch_team_v5",
        "last_touch_id_v5",
        "last_touch_type_v5",

        "source_v5",
    ]
]


print(
    window
    .to_string(
        index=False
    )
)


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