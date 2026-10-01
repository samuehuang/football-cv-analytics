import os
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "outputs/possession_v3_frames.csv"

OUTPUT_FRAME_CSV = "outputs/possession_v4_frames.csv"
OUTPUT_SUMMARY_CSV = "outputs/possession_v4_summary.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Basic
# ============================================================

FPS = 25.0


# ============================================================
# Reception / control
#
# IMPORTANT:
# A player is NOT controller merely because the projected
# ball position is close.
#
# Reception requires:
#   1. pitch proximity
#   2. image foot proximity
#   3. sufficiently slow ball
#   4. consecutive-frame confirmation
# ============================================================

CONTROL_RADIUS_M = 4.0

FOOT_CONTROL_MAX_PX = 95.0

# Conservative:
# 15.4 m/s example must NOT become controller.
RECEPTION_MAX_SPEED_MPS = 10.0

# Same player must satisfy reception evidence
# for 3 consecutive frames.
RECEPTION_CONFIRM_FRAMES = 3


# ============================================================
# In-transit
# ============================================================

# If ball is moving faster than this, proximity is NOT enough
# to generate a new controller.
IN_TRANSIT_SPEED_MPS = 10.0

# Retain previous team's possession while pass / aerial ball
# is travelling.
MAX_IN_TRANSIT_HOLD_FRAMES = 50   # 2.0 sec


# ============================================================
# Aerial contest
#
# Ball is travelling / projection mismatch and players from
# BOTH teams are sufficiently close to the projected area.
# ============================================================

AERIAL_CONTEST_MIN_SPEED_MPS = 8.0

AERIAL_CONTEST_PITCH_RADIUS_M = 6.0

AERIAL_CONTEST_IMAGE_RADIUS_PX = 180.0

AERIAL_CONTEST_PITCH_MARGIN_M = 1.5

AERIAL_CONTEST_IMAGE_MARGIN_PX = 70.0


# ============================================================
# Loose / missing
# ============================================================

LOOSE_HOLD_FRAMES = 12            # 0.48 sec

BALL_MISSING_HOLD_FRAMES = 8      # 0.32 sec

PLAYER_QUALITY_HOLD_FRAMES = 5    # 0.20 sec


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)

df = (
    df
    .sort_values("frame")
    .reset_index(drop=True)
)


# ============================================================
# Normalize bool
# ============================================================

def normalize_bool(series):

    return (
        series
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes"
            ]
        )
    )


df["ball_usable_bool"] = normalize_bool(
    df["ball_usable"]
)

df["player_frame_good_bool"] = normalize_bool(
    df["player_frame_good"]
)

df["projection_mismatch_bool"] = normalize_bool(
    df["projection_mismatch"]
)


# ============================================================
# Numeric columns
# ============================================================

numeric_columns = [

    "frame",
    "time_sec",

    "ball_speed_mps",

    "nearest_A_id",
    "nearest_B_id",

    "nearest_A_pitch_distance_m",
    "nearest_B_pitch_distance_m",

    "nearest_A_image_foot_distance_px",
    "nearest_B_image_foot_distance_px",
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# Helpers
# ============================================================

def player_control_candidate(
    row,
    team
):

    pitch_distance = row[
        f"nearest_{team}_pitch_distance_m"
    ]

    foot_distance = row[
        f"nearest_{team}_image_foot_distance_px"
    ]

    speed = row[
        "ball_speed_mps"
    ]


    if (
        pd.isna(pitch_distance)
        or
        pd.isna(foot_distance)
    ):

        return False


    spatial_ok = (

        pitch_distance
        <=
        CONTROL_RADIUS_M

        and

        foot_distance
        <=
        FOOT_CONTROL_MAX_PX
    )


    # First frame of a trajectory may have no speed.
    # In that case allow spatial evidence, but it still needs
    # multi-frame reception confirmation.
    speed_ok = (

        pd.isna(speed)

        or

        speed
        <=
        RECEPTION_MAX_SPEED_MPS
    )


    return (
        spatial_ok
        and
        speed_ok
    )


def is_aerial_contest(
    row
):

    A_pitch = row[
        "nearest_A_pitch_distance_m"
    ]

    B_pitch = row[
        "nearest_B_pitch_distance_m"
    ]

    A_foot = row[
        "nearest_A_image_foot_distance_px"
    ]

    B_foot = row[
        "nearest_B_image_foot_distance_px"
    ]

    speed = row[
        "ball_speed_mps"
    ]


    if any(
        pd.isna(value)
        for value in [
            A_pitch,
            B_pitch,
            A_foot,
            B_foot
        ]
    ):

        return False


    both_near = (

        A_pitch
        <=
        AERIAL_CONTEST_PITCH_RADIUS_M

        and

        B_pitch
        <=
        AERIAL_CONTEST_PITCH_RADIUS_M

        and

        A_foot
        <=
        AERIAL_CONTEST_IMAGE_RADIUS_PX

        and

        B_foot
        <=
        AERIAL_CONTEST_IMAGE_RADIUS_PX
    )


    if not both_near:

        return False


    similar_distance = (

        abs(
            A_pitch
            -
            B_pitch
        )
        <=
        AERIAL_CONTEST_PITCH_MARGIN_M

        or

        abs(
            A_foot
            -
            B_foot
        )
        <=
        AERIAL_CONTEST_IMAGE_MARGIN_PX
    )


    ball_is_aerial_or_fast = (

        row[
            "projection_mismatch_bool"
        ]

        or

        (
            pd.notna(speed)
            and
            speed
            >=
            AERIAL_CONTEST_MIN_SPEED_MPS
        )
    )


    return (

        similar_distance

        and

        ball_is_aerial_or_fast
    )


# ============================================================
# PASS 1
#
# Physical evidence per frame.
#
# This does NOT yet assign possession.
# ============================================================

raw_states = []

candidate_teams = []
candidate_ids = []


for _, row in df.iterrows():

    # --------------------------------------------------------
    # Missing ball
    # --------------------------------------------------------

    if not row[
        "ball_usable_bool"
    ]:

        raw_states.append(
            "Missing"
        )

        candidate_teams.append(
            None
        )

        candidate_ids.append(
            np.nan
        )

        continue


    # --------------------------------------------------------
    # Player geometry insufficient
    # --------------------------------------------------------

    if not row[
        "player_frame_good_bool"
    ]:

        raw_states.append(
            "PlayerQualityLow"
        )

        candidate_teams.append(
            None
        )

        candidate_ids.append(
            np.nan
        )

        continue


    speed = row[
        "ball_speed_mps"
    ]


    # --------------------------------------------------------
    # Aerial duel takes priority.
    #
    # Nobody becomes controller.
    # --------------------------------------------------------

    if is_aerial_contest(
        row
    ):

        raw_states.append(
            "AerialContest"
        )

        candidate_teams.append(
            None
        )

        candidate_ids.append(
            np.nan
        )

        continue


    # --------------------------------------------------------
    # High-speed ball = in transit
    #
    # IMPORTANT:
    # Cannot generate a new controller.
    # --------------------------------------------------------

    if (
        pd.notna(speed)
        and
        speed
        >
        IN_TRANSIT_SPEED_MPS
    ):

        raw_states.append(
            "InTransit"
        )

        candidate_teams.append(
            None
        )

        candidate_ids.append(
            np.nan
        )

        continue


    # --------------------------------------------------------
    # Possible reception / control
    # --------------------------------------------------------

    A_candidate = player_control_candidate(
        row,
        "A"
    )

    B_candidate = player_control_candidate(
        row,
        "B"
    )


    # --------------------------------------------------------
    # Both teams satisfy control region
    # --------------------------------------------------------

    if (
        A_candidate
        and
        B_candidate
    ):

        raw_states.append(
            "Contested"
        )

        candidate_teams.append(
            None
        )

        candidate_ids.append(
            np.nan
        )

        continue


    # --------------------------------------------------------
    # A possible reception
    # --------------------------------------------------------

    if A_candidate:

        raw_states.append(
            "ReceptionCandidate"
        )

        candidate_teams.append(
            "A"
        )

        candidate_ids.append(
            row[
                "nearest_A_id"
            ]
        )

        continue


    # --------------------------------------------------------
    # B possible reception
    # --------------------------------------------------------

    if B_candidate:

        raw_states.append(
            "ReceptionCandidate"
        )

        candidate_teams.append(
            "B"
        )

        candidate_ids.append(
            row[
                "nearest_B_id"
            ]
        )

        continue


    # --------------------------------------------------------
    # Visible ball but nobody controls it
    # --------------------------------------------------------

    raw_states.append(
        "Loose"
    )

    candidate_teams.append(
        None
    )

    candidate_ids.append(
        np.nan
    )


df["raw_ball_state"] = raw_states

df["reception_candidate_team"] = (
    candidate_teams
)

df["reception_candidate_id"] = (
    candidate_ids
)


# ============================================================
# PASS 2
#
# Reception confirmation + team possession.
# ============================================================

df["ball_state_v4"] = "Unknown"

df["controller_team_v4"] = None

df["controller_id_v4"] = np.nan

df["possession_v4"] = "Unknown"

df["possession_source_v4"] = "unknown"

df["reception_candidate_count"] = 0


current_team = None

current_controller_team = None
current_controller_id = None


reception_team = None
reception_id = None
reception_count = 0


transit_streak = 0
loose_streak = 0
missing_streak = 0
quality_streak = 0


confirmed_receptions = 0
confirmed_turnovers = 0
aerial_contest_frames = 0


for i in range(len(df)):

    raw_state = df.loc[
        i,
        "raw_ball_state"
    ]


    # ========================================================
    # Reception candidate
    # ========================================================

    if raw_state == "ReceptionCandidate":

        transit_streak = 0
        loose_streak = 0
        missing_streak = 0
        quality_streak = 0


        team = df.loc[
            i,
            "reception_candidate_team"
        ]

        player_id = df.loc[
            i,
            "reception_candidate_id"
        ]


        same_candidate = (

            reception_team
            ==
            team

            and

            reception_id
            is not None

            and

            pd.notna(player_id)

            and

            abs(
                float(reception_id)
                -
                float(player_id)
            )
            <
            0.1
        )


        if same_candidate:

            reception_count += 1

        else:

            reception_team = team

            reception_id = (
                float(player_id)
                if pd.notna(player_id)
                else None
            )

            reception_count = 1


        df.loc[
            i,
            "reception_candidate_count"
        ] = reception_count


        # ----------------------------------------------------
        # Reception confirmed
        # ----------------------------------------------------

        if (
            reception_count
            >=
            RECEPTION_CONFIRM_FRAMES
        ):

            new_team = team

            new_player = player_id


            # Turnover only occurs AFTER controlled reception.
            turnover = (

                current_team is not None

                and

                current_team
                !=
                new_team
            )


            if turnover:

                confirmed_turnovers += 1


            # Count only first confirmed frame of reception
            if (
                reception_count
                ==
                RECEPTION_CONFIRM_FRAMES
            ):

                confirmed_receptions += 1


            current_team = (
                new_team
            )

            current_controller_team = (
                new_team
            )

            current_controller_id = (
                new_player
            )


            df.loc[
                i,
                "ball_state_v4"
            ] = "Controlled"


            df.loc[
                i,
                "controller_team_v4"
            ] = new_team


            df.loc[
                i,
                "controller_id_v4"
            ] = new_player


            df.loc[
                i,
                "possession_v4"
            ] = new_team


            if turnover:

                df.loc[
                    i,
                    "possession_source_v4"
                ] = "confirmed_controlled_turnover"

            elif (
                reception_count
                ==
                RECEPTION_CONFIRM_FRAMES
            ):

                df.loc[
                    i,
                    "possession_source_v4"
                ] = "confirmed_reception"

            else:

                df.loc[
                    i,
                    "possession_source_v4"
                ] = "controlled"


        # ----------------------------------------------------
        # Reception not yet confirmed
        # ----------------------------------------------------

        else:

            df.loc[
                i,
                "ball_state_v4"
            ] = "Loose"


            df.loc[
                i,
                "controller_team_v4"
            ] = None


            df.loc[
                i,
                "controller_id_v4"
            ] = np.nan


            if current_team is not None:

                df.loc[
                    i,
                    "possession_v4"
                ] = current_team


                df.loc[
                    i,
                    "possession_source_v4"
                ] = "pending_reception"

            else:

                df.loc[
                    i,
                    "possession_v4"
                ] = "Unknown"


                df.loc[
                    i,
                    "possession_source_v4"
                ] = "pending_initial_reception"


    # ========================================================
    # In transit
    # ========================================================

    elif raw_state == "InTransit":

        reception_team = None
        reception_id = None
        reception_count = 0

        loose_streak = 0
        missing_streak = 0
        quality_streak = 0

        transit_streak += 1


        current_controller_team = None
        current_controller_id = None


        df.loc[
            i,
            "ball_state_v4"
        ] = "InTransit"


        if (
            current_team is not None

            and

            transit_streak
            <=
            MAX_IN_TRANSIT_HOLD_FRAMES
        ):

            df.loc[
                i,
                "possession_v4"
            ] = current_team


            df.loc[
                i,
                "possession_source_v4"
            ] = "carried_in_transit"

        else:

            df.loc[
                i,
                "possession_v4"
            ] = "Unknown"


            df.loc[
                i,
                "possession_source_v4"
            ] = "transit_too_long"


    # ========================================================
    # Aerial contest
    # ========================================================

    elif raw_state == "AerialContest":

        reception_team = None
        reception_id = None
        reception_count = 0

        transit_streak = 0
        loose_streak = 0
        missing_streak = 0
        quality_streak = 0

        current_controller_team = None
        current_controller_id = None

        aerial_contest_frames += 1


        df.loc[
            i,
            "ball_state_v4"
        ] = "AerialContest"


        df.loc[
            i,
            "possession_v4"
        ] = "Contested"


        df.loc[
            i,
            "possession_source_v4"
        ] = "aerial_contest"


        # IMPORTANT:
        # current_team is NOT erased.
        #
        # After the aerial duel, a confirmed reception decides
        # whether the old team retains it or possession turns over.


    # ========================================================
    # Ground / low-ball contest
    # ========================================================

    elif raw_state == "Contested":

        reception_team = None
        reception_id = None
        reception_count = 0

        transit_streak = 0
        loose_streak = 0
        missing_streak = 0
        quality_streak = 0

        current_controller_team = None
        current_controller_id = None


        df.loc[
            i,
            "ball_state_v4"
        ] = "Contested"


        df.loc[
            i,
            "possession_v4"
        ] = "Contested"


        df.loc[
            i,
            "possession_source_v4"
        ] = "spatial_contest"


    # ========================================================
    # Loose ball
    # ========================================================

    elif raw_state == "Loose":

        reception_team = None
        reception_id = None
        reception_count = 0

        transit_streak = 0
        missing_streak = 0
        quality_streak = 0

        loose_streak += 1


        current_controller_team = None
        current_controller_id = None


        df.loc[
            i,
            "ball_state_v4"
        ] = "Loose"


        if (
            current_team is not None

            and

            loose_streak
            <=
            LOOSE_HOLD_FRAMES
        ):

            df.loc[
                i,
                "possession_v4"
            ] = current_team


            df.loc[
                i,
                "possession_source_v4"
            ] = "carried_loose"

        else:

            df.loc[
                i,
                "possession_v4"
            ] = "Unknown"


            df.loc[
                i,
                "possession_source_v4"
            ] = "loose_too_long"


    # ========================================================
    # Missing
    # ========================================================

    elif raw_state == "Missing":

        reception_team = None
        reception_id = None
        reception_count = 0

        transit_streak = 0
        loose_streak = 0
        quality_streak = 0

        missing_streak += 1


        current_controller_team = None
        current_controller_id = None


        df.loc[
            i,
            "ball_state_v4"
        ] = "Missing"


        if (
            current_team is not None

            and

            missing_streak
            <=
            BALL_MISSING_HOLD_FRAMES
        ):

            df.loc[
                i,
                "possession_v4"
            ] = current_team


            df.loc[
                i,
                "possession_source_v4"
            ] = "carried_missing"

        else:

            df.loc[
                i,
                "possession_v4"
            ] = "Unknown"


            df.loc[
                i,
                "possession_source_v4"
            ] = "missing_too_long"


    # ========================================================
    # Player quality low
    # ========================================================

    elif raw_state == "PlayerQualityLow":

        reception_team = None
        reception_id = None
        reception_count = 0

        transit_streak = 0
        loose_streak = 0
        missing_streak = 0

        quality_streak += 1


        current_controller_team = None
        current_controller_id = None


        df.loc[
            i,
            "ball_state_v4"
        ] = "Unknown"


        if (
            current_team is not None

            and

            quality_streak
            <=
            PLAYER_QUALITY_HOLD_FRAMES
        ):

            df.loc[
                i,
                "possession_v4"
            ] = current_team


            df.loc[
                i,
                "possession_source_v4"
            ] = "carried_low_player_quality"

        else:

            df.loc[
                i,
                "possession_v4"
            ] = "Unknown"


            df.loc[
                i,
                "possession_source_v4"
            ] = "player_quality_too_low"


    # ========================================================
    # Fallback
    # ========================================================

    else:

        reception_team = None
        reception_id = None
        reception_count = 0

        transit_streak = 0
        loose_streak = 0
        missing_streak = 0
        quality_streak = 0


        current_controller_team = None
        current_controller_id = None


        df.loc[
            i,
            "ball_state_v4"
        ] = "Unknown"


        df.loc[
            i,
            "possession_v4"
        ] = "Unknown"


        df.loc[
            i,
            "possession_source_v4"
        ] = "unknown"


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT_FRAME_CSV,
    index=False
)


# ============================================================
# Summary
# ============================================================

total = len(df)


A_frames = int(
    (
        df["possession_v4"]
        ==
        "A"
    ).sum()
)

B_frames = int(
    (
        df["possession_v4"]
        ==
        "B"
    ).sum()
)

contested_frames = int(
    (
        df["possession_v4"]
        ==
        "Contested"
    ).sum()
)

unknown_frames = int(
    (
        df["possession_v4"]
        ==
        "Unknown"
    ).sum()
)


known_team = (
    A_frames
    +
    B_frames
)


if known_team > 0:

    A_share = (
        A_frames
        /
        known_team
        *
        100.0
    )

    B_share = (
        B_frames
        /
        known_team
        *
        100.0
    )

else:

    A_share = 0.0
    B_share = 0.0


summary = pd.DataFrame(
    [
        {
            "state": "A",
            "frames": A_frames,
            "seconds": A_frames / FPS,
            "known_share_pct": A_share,
        },

        {
            "state": "B",
            "frames": B_frames,
            "seconds": B_frames / FPS,
            "known_share_pct": B_share,
        },

        {
            "state": "Contested",
            "frames": contested_frames,
            "seconds": contested_frames / FPS,
            "known_share_pct": np.nan,
        },

        {
            "state": "Unknown",
            "frames": unknown_frames,
            "seconds": unknown_frames / FPS,
            "known_share_pct": np.nan,
        },
    ]
)


summary.to_csv(
    OUTPUT_SUMMARY_CSV,
    index=False
)


# ============================================================
# Regression cases
#
# These were V3 failure examples.
# ============================================================

REGRESSION_FRAMES = [
    159,   # 6.36s, ~39.72 m/s
    253,   # 10.12s, ~15.4 m/s
]


regression_columns = [

    "frame",
    "time_sec",

    "ball_speed_mps",

    "raw_ball_state",
    "ball_state_v4",

    "possession_v4",

    "controller_team_v4",
    "controller_id_v4",

    "reception_candidate_team",
    "reception_candidate_id",
    "reception_candidate_count",

    "possession_source_v4",
]


regression = df[
    df["frame"]
    .isin(
        REGRESSION_FRAMES
    )
][
    regression_columns
]


# ============================================================
# Print
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V4 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Frames:",
    total
)

print(
    "Window:",
    f"{df['time_sec'].min():.2f}s",
    "->",
    f"{df['time_sec'].max():.2f}s"
)


print("")
print(
    "RAW BALL STATE"
)

print(
    df[
        "raw_ball_state"
    ]
    .value_counts()
    .to_string()
)


print("")
print(
    "FINAL BALL STATE"
)

print(
    df[
        "ball_state_v4"
    ]
    .value_counts()
    .to_string()
)


print("")
print(
    "FINAL POSSESSION"
)

print(
    df[
        "possession_v4"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "Team A:",
    A_frames,
    "frames",
    f"({A_frames / FPS:.2f}s)"
)

print(
    "Team B:",
    B_frames,
    "frames",
    f"({B_frames / FPS:.2f}s)"
)

print(
    "Contested:",
    contested_frames
)

print(
    "Unknown:",
    unknown_frames
)


print("")
print(
    "Known team possession:"
)

print(
    "A:",
    f"{A_share:.1f}%"
)

print(
    "B:",
    f"{B_share:.1f}%"
)


print("")
print(
    "Confirmed receptions:",
    confirmed_receptions
)

print(
    "Confirmed controlled turnovers:",
    confirmed_turnovers
)

print(
    "Aerial contest frames:",
    aerial_contest_frames
)


print("")
print(
    "POSSESSION SOURCE"
)

print(
    df[
        "possession_source_v4"
    ]
    .value_counts()
    .to_string()
)


print("")
print(
    "========================================"
)

print(
    "REGRESSION CHECK"
)

print(
    "========================================"
)

print("")

print(
    regression
    .round(2)
    .to_string(
        index=False
    )
)


print("")
print(
    "Expected:"
)

print(
    "frame 159 must NOT be Controlled / turnover to A"
)

print(
    "frame 253 must NOT be Controlled / turnover to B"
)


print("")
print(
    "Frame CSV:",
    OUTPUT_FRAME_CSV
)

print(
    "Summary CSV:",
    OUTPUT_SUMMARY_CSV
)
