import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

POSSESSION_V4_CSV = "outputs/possession_v4_frames.csv"

AIR_TOUCH_V4_CSV = "outputs/air_touch_v4_events.csv"

OUTPUT_FRAMES_CSV = "outputs/possession_v5_frames.csv"
OUTPUT_SUMMARY_CSV = "outputs/possession_v5_summary.csv"
OUTPUT_AUDIT_CSV = "outputs/possession_v5_event_audit.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# V5 STATE SETTINGS
# ============================================================

# After a confirmed AirTouch, keep that team's phase ownership
# while the ball is still travelling.
#
# 50 frames @ 25 fps ~= 2.0 sec
MAX_AIR_TOUCH_CARRY_FRAMES = 50


# ============================================================
# REGRESSION FRAMES
# ============================================================

# V3 false turnover:
# Must remain B / InTransit / controller NONE
REGRESSION_IN_TRANSIT_FRAME = 159


# Known aerial duel:
# A #1 + B #14
# Must become Contested / AerialContest
REGRESSION_CONTEST_FRAME = 172


# A #6 bounce / reception:
# Must NOT be turned into AirTouch
REGRESSION_RECEPTION_FRAME = 203


# V3 false control:
# Must remain AerialContest / Contested
REGRESSION_OLD_AERIAL_FRAME = 253


# Confirmed B #10 header:
# Must switch phase ownership from A -> B,
# while controller remains NONE
REGRESSION_HEADER_FRAME = 281


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


def same_player_id(a, b):

    if (
        pd.isna(a)
        or
        pd.isna(b)
    ):
        return False

    return abs(
        float(a)
        -
        float(b)
    ) < 0.1


# ============================================================
# LOAD
# ============================================================

print("")
print("Loading Possession V4...")

v4 = pd.read_csv(
    POSSESSION_V4_CSV
)


print("Loading AirTouch V4...")

physical_events = pd.read_csv(
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


if (
    "time_sec"
    in
    v4.columns
):

    v4["time_sec"] = pd.to_numeric(
        v4["time_sec"],
        errors="coerce"
    )


if (
    "controller_id_v4"
    in
    v4.columns
):

    v4["controller_id_v4"] = pd.to_numeric(
        v4["controller_id_v4"],
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
# INFER FPS
# ============================================================

fps = 25.0


if (
    "time_sec"
    in
    v4.columns
):

    valid_times = (
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


    time_diffs = (
        valid_times[
            "time_sec"
        ]
        .diff()
    )


    frame_diffs = (
        valid_times[
            "frame"
        ]
        .diff()
    )


    valid_dt = (

        time_diffs.notna()

        &

        frame_diffs.notna()

        &

        (
            time_diffs
            >
            0
        )

        &

        (
            frame_diffs
            >
            0
        )
    )


    if valid_dt.any():

        estimated_fps = (

            frame_diffs[
                valid_dt
            ]

            /

            time_diffs[
                valid_dt
            ]
        )


        estimated_fps = (
            estimated_fps[
                np.isfinite(
                    estimated_fps
                )
            ]
        )


        if (
            len(
                estimated_fps
            )
            >
            0
        ):

            fps = float(
                estimated_fps.median()
            )


print(
    f"Estimated FPS: {fps:.2f}"
)


# ============================================================
# NORMALIZE PHYSICAL EVENTS
# ============================================================

for column in [
    "physical_event_id",
    "start_frame",
    "end_frame",
    "last_touch_track_id",
]:

    if (
        column
        in
        physical_events.columns
    ):

        physical_events[column] = pd.to_numeric(
            physical_events[column],
            errors="coerce"
        )


physical_events = physical_events.dropna(
    subset=[
        "start_frame",
        "end_frame",
        "final_class",
    ]
).copy()


physical_events[
    "start_frame"
] = (
    physical_events[
        "start_frame"
    ]
    .astype(int)
)


physical_events[
    "end_frame"
] = (
    physical_events[
        "end_frame"
    ]
    .astype(int)
)


# ============================================================
# ONLY HIGH-CONFIDENCE PHYSICAL OVERRIDES
#
# ReceptionOrBounce:
#     does NOT override V4
#
# WeakInteraction:
#     does NOT override V4
#
# AirTouch:
#     updates Last Touch and team phase ownership
#
# AerialContest:
#     possession -> Contested
#     controller -> NONE
#     Last Touch -> Unknown
# ============================================================

strong_events = physical_events[
    physical_events[
        "final_class"
    ].isin(
        [
            "AirTouch",
            "AerialContest",
        ]
    )
].copy()


# ============================================================
# FRAME -> PHYSICAL EVENT LOOKUP
# ============================================================

physical_event_by_frame = {}


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

        # ----------------------------------------------------
        # If somehow multiple strong physical events overlap,
        # AerialContest has priority because ownership is
        # ambiguous.
        # ----------------------------------------------------

        if (
            frame
            not in
            physical_event_by_frame
        ):

            physical_event_by_frame[
                frame
            ] = event


        else:

            existing = (
                physical_event_by_frame[
                    frame
                ]
            )


            if (
                str(
                    event[
                        "final_class"
                    ]
                )
                ==
                "AerialContest"
            ):

                physical_event_by_frame[
                    frame
                ] = event


# ============================================================
# STATE MEMORY
# ============================================================

# A confirmed AirTouch can change who owns the next travelling
# phase without creating a controller.
air_touch_owner_team = None
air_touch_owner_id = np.nan
air_touch_start_frame = None


# After an unresolved aerial contest, do NOT immediately return
# to stale pre-contest ownership.
#
# Wait until a new confirmed controller / AirTouch resolves it.
post_contest_uncertain = False


# Last confirmed touch memory.
last_touch_team = None
last_touch_id = np.nan
last_touch_type = None


# ============================================================
# OUTPUT
# ============================================================

output_rows = []
audit_rows = []


print("")
print(
    "========================================"
)

print(
    "POSSESSION V5"
)

print(
    "AirTouch / LastTouch integration"
)

print(
    "========================================"
)

print("")


# ============================================================
# MAIN FRAME LOOP
# ============================================================

for _, base_row in v4.iterrows():

    row = base_row.to_dict()


    frame = int(
        base_row[
            "frame"
        ]
    )


    # ========================================================
    # BASELINE V4 VALUES
    # ========================================================

    base_possession = str(
        base_row.get(
            "possession_v4",
            "Unknown"
        )
    )


    base_ball_state = str(
        base_row.get(
            "ball_state_v4",
            "Unknown"
        )
    )


    base_source = str(
        base_row.get(
            "source_v4",
            ""
        )
    )


    base_controller_team = clean_team(
        base_row.get(
            "controller_team_v4",
            None
        )
    )


    base_controller_id = clean_id(
        base_row.get(
            "controller_id_v4",
            np.nan
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
    # IS V4 SHOWING A REAL CONFIRMED CONTROLLER?
    #
    # A controller is stronger evidence than carry-memory.
    # ========================================================

    has_confirmed_controller = (

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
    # PHYSICAL EVENT AT THIS FRAME
    # ========================================================

    event = physical_event_by_frame.get(
        frame,
        None
    )


    # ========================================================
    # PRIORITY 1:
    # PHYSICAL AERIAL CONTEST
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
        # Ownership is unresolved now.
        # ----------------------------------------------------

        air_touch_owner_team = None
        air_touch_owner_id = np.nan
        air_touch_start_frame = None


        post_contest_uncertain = True


        last_touch_team = None
        last_touch_id = np.nan
        last_touch_type = (
            "AerialContestUnknown"
        )


    # ========================================================
    # PRIORITY 2:
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


        # ----------------------------------------------------
        # Only apply if team identity is valid.
        # ----------------------------------------------------

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


            # A header/touch sending the ball away is not
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
            # Start phase-owner override.
            # ------------------------------------------------

            air_touch_owner_team = (
                touch_team
            )


            air_touch_owner_id = (
                touch_id
            )


            air_touch_start_frame = (
                frame
            )


            post_contest_uncertain = (
                False
            )


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
    # PRIORITY 3:
    # CONFIRMED CONTROLLER
    #
    # New controlled possession resolves all previous
    # in-transit / contest ambiguity.
    # ========================================================

    elif has_confirmed_controller:

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

        air_touch_owner_team = None
        air_touch_owner_id = np.nan
        air_touch_start_frame = None


        post_contest_uncertain = False


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
    # PRIORITY 4:
    # V4 ITSELF SAYS AERIAL CONTEST
    #
    # A new duel makes stale AirTouch ownership unsafe.
    # ========================================================

    elif (
        base_ball_state
        ==
        "AerialContest"
    ):

        possession_v5 = (
            "Contested"
        )


        ball_state_v5 = (
            "AerialContest"
        )


        controller_team_v5 = None
        controller_id_v5 = np.nan


        source_v5 = (
            base_source
        )


        air_touch_owner_team = None
        air_touch_owner_id = np.nan
        air_touch_start_frame = None


        post_contest_uncertain = True


        last_touch_team = None
        last_touch_id = np.nan
        last_touch_type = (
            "AerialContestUnknown"
        )


    # ========================================================
    # PRIORITY 5:
    # CARRY OWNERSHIP AFTER CONFIRMED AIR TOUCH
    #
    # Example:
    #
    # A possession
    # -> B #10 header
    # -> ball keeps travelling
    #
    # V4 would retain A.
    # V5 must retain B.
    # ========================================================

    elif (
        air_touch_owner_team
        in
        {
            "A",
            "B",
        }

        and

        air_touch_start_frame
        is not None
    ):

        carry_age = (
            frame
            -
            air_touch_start_frame
        )


        if (
            carry_age
            <=
            MAX_AIR_TOUCH_CARRY_FRAMES
        ):

            possession_v5 = (
                air_touch_owner_team
            )


            controller_team_v5 = None
            controller_id_v5 = np.nan


            # Keep V4's physical ball state.
            ball_state_v5 = (
                base_ball_state
            )


            source_v5 = (
                "carried_after_air_touch"
            )


        else:

            # ------------------------------------------------
            # Do not keep an AirTouch forever.
            # ------------------------------------------------

            air_touch_owner_team = None
            air_touch_owner_id = np.nan
            air_touch_start_frame = None


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
                "air_touch_carry_timeout_fallback"
            )


    # ========================================================
    # PRIORITY 6:
    # AFTER UNRESOLVED AERIAL CONTEST
    #
    # Do not return immediately to stale V4 carried ownership.
    #
    # Wait for:
    # - confirmed controller
    # - confirmed AirTouch
    # ========================================================

    elif post_contest_uncertain:

        possession_v5 = (
            "Unknown"
        )


        ball_state_v5 = (
            base_ball_state
        )


        controller_team_v5 = None
        controller_id_v5 = np.nan


        source_v5 = (
            "unresolved_after_aerial_contest"
        )


    # ========================================================
    # OTHERWISE:
    # KEEP V4
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


    output_rows.append(
        row
    )


    # ========================================================
    # AUDIT ONLY CHANGED / IMPORTANT FRAMES
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
        in
        {
            "carried_after_air_touch",
            "unresolved_after_aerial_contest",
        }
    )


    if changed:

        audit_rows.append(
            {
                "frame":
                    frame,

                "time_sec":
                    base_row.get(
                        "time_sec",
                        frame
                        /
                        fps
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

                "physical_participants":
                    physical_participants,
            }
        )


# ============================================================
# DATAFRAME
# ============================================================

v5 = pd.DataFrame(
    output_rows
)


audit_df = pd.DataFrame(
    audit_rows
)


v5.to_csv(
    OUTPUT_FRAMES_CSV,
    index=False
)


audit_df.to_csv(
    OUTPUT_AUDIT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


# ------------------------------------------------------------
# Frame count
# ------------------------------------------------------------

summary_rows.append(
    {
        "metric":
            "frames",

        "value":
            len(
                v5
            ),
    }
)


# ------------------------------------------------------------
# Possession counts
# ------------------------------------------------------------

possession_counts = (
    v5[
        "possession_v5"
    ]
    .value_counts(
        dropna=False
    )
)


for key, value in possession_counts.items():

    summary_rows.append(
        {
            "metric":
                f"possession_{key}_frames",

            "value":
                int(
                    value
                ),
        }
    )


# ------------------------------------------------------------
# Ball states
# ------------------------------------------------------------

ball_state_counts = (
    v5[
        "ball_state_v5"
    ]
    .value_counts(
        dropna=False
    )
)


for key, value in ball_state_counts.items():

    summary_rows.append(
        {
            "metric":
                f"ball_state_{key}_frames",

            "value":
                int(
                    value
                ),
        }
    )


# ------------------------------------------------------------
# Sources
# ------------------------------------------------------------

source_counts = (
    v5[
        "source_v5"
    ]
    .value_counts(
        dropna=False
    )
)


for key, value in source_counts.items():

    summary_rows.append(
        {
            "metric":
                f"source_{key}_frames",

            "value":
                int(
                    value
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    OUTPUT_SUMMARY_CSV,
    index=False
)


# ============================================================
# QA HELPERS
# ============================================================

v5_by_frame = (
    v5
    .set_index(
        "frame"
    )
)


def print_frame_check(
    frame,
    title
):

    print("")
    print(
        title
    )


    if (
        frame
        not in
        v5_by_frame.index
    ):

        print(
            f"frame {frame}: NOT FOUND"
        )

        return


    row = v5_by_frame.loc[
        frame
    ]


    if isinstance(
        row,
        pd.DataFrame
    ):

        row = row.iloc[0]


    print(
        f"frame={frame} | "
        f"time={row.get('time_sec', np.nan):.2f}s"
    )


    print(
        f"  V4: "
        f"possession={row.get('possession_v4')} | "
        f"state={row.get('ball_state_v4')} | "
        f"controller="
        f"{row.get('controller_team_v4')} "
        f"{row.get('controller_id_v4')} | "
        f"source={row.get('source_v4')}"
    )


    print(
        f"  V5: "
        f"possession={row.get('possession_v5')} | "
        f"state={row.get('ball_state_v5')} | "
        f"controller="
        f"{row.get('controller_team_v5')} "
        f"{row.get('controller_id_v5')} | "
        f"last_touch="
        f"{row.get('last_touch_team_v5')} "
        f"{row.get('last_touch_id_v5')} | "
        f"source={row.get('source_v5')}"
    )


# ============================================================
# REGRESSION TESTS
# ============================================================

def get_frame_row(frame):

    if (
        frame
        not in
        v5_by_frame.index
    ):

        return None


    row = v5_by_frame.loc[
        frame
    ]


    if isinstance(
        row,
        pd.DataFrame
    ):

        row = row.iloc[0]


    return row


# ------------------------------------------------------------
# 159:
# B / InTransit / no controller
# ------------------------------------------------------------

r159 = get_frame_row(
    REGRESSION_IN_TRANSIT_FRAME
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
)


# ------------------------------------------------------------
# 172:
# AerialContest / Contested
# ------------------------------------------------------------

r172 = get_frame_row(
    REGRESSION_CONTEST_FRAME
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
)


# ------------------------------------------------------------
# 203:
# Must NOT be AirTouch override
# ------------------------------------------------------------

r203 = get_frame_row(
    REGRESSION_RECEPTION_FRAME
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
        r203.get(
            "physical_event_class_v5",
            ""
        )
    )
    !=
    "AirTouch"
)


# ------------------------------------------------------------
# 253:
# Known aerial contest remains contested
# ------------------------------------------------------------

r253 = get_frame_row(
    REGRESSION_OLD_AERIAL_FRAME
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
)


# ------------------------------------------------------------
# 281:
# B header -> B / InTransit / no controller / last touch B#10
# ------------------------------------------------------------

r281 = get_frame_row(
    REGRESSION_HEADER_FRAME
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
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V5 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    f"Frames: {len(v5)}"
)

print(
    f"FPS: {fps:.2f}"
)


if (
    "time_sec"
    in
    v5.columns

    and

    v5[
        "time_sec"
    ].notna().any()
):

    print(
        "Window: "
        f"{v5['time_sec'].min():.2f}s "
        f"-> "
        f"{v5['time_sec'].max():.2f}s"
    )


print("")

print(
    "FINAL POSSESSION V5"
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
    "FINAL BALL STATE V5"
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
    len(
        audit_df
    )
)


print("")

print(
    "REGRESSION CHECKS"
)

print(
    "frame 159 | "
    "B InTransit, no controller:",
    (
        "PASS"
        if reg159
        else
        "FAIL"
    )
)


print(
    "frame 172 | "
    "A#1 + B#14 aerial contest:",
    (
        "PASS"
        if reg172
        else
        "FAIL"
    )
)


print(
    "frame 203 | "
    "A#6 reception/bounce not AirTouch:",
    (
        "PASS"
        if reg203
        else
        "FAIL"
    )
)


print(
    "frame 253 | "
    "old false control remains aerial contest:",
    (
        "PASS"
        if reg253
        else
        "FAIL"
    )
)


print(
    "frame 281 | "
    "B#10 header -> B InTransit, no controller:",
    (
        "PASS"
        if reg281
        else
        "FAIL"
    )
)


# ============================================================
# DETAILED CHECKS
# ============================================================

print_frame_check(
    REGRESSION_IN_TRANSIT_FRAME,
    "FRAME 159"
)


print_frame_check(
    REGRESSION_CONTEST_FRAME,
    "FRAME 172"
)


print_frame_check(
    REGRESSION_RECEPTION_FRAME,
    "FRAME 203"
)


print_frame_check(
    REGRESSION_OLD_AERIAL_FRAME,
    "FRAME 253"
)


print_frame_check(
    REGRESSION_HEADER_FRAME,
    "FRAME 281"
)


# ============================================================
# SHOW AIR TOUCH CARRY AFTER FRAME 281
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POST-HEADER OWNERSHIP WINDOW"
)

print(
    "========================================"
)

print("")


post_header = v5[
    (
        v5[
            "frame"
        ]
        >=
        REGRESSION_HEADER_FRAME
    )

    &

    (
        v5[
            "frame"
        ]
        <=
        REGRESSION_HEADER_FRAME
        +
        20
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
        "source_v5",
    ]
]


print(
    post_header
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
