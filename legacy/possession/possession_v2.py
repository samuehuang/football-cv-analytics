import os
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "outputs/possession_v1_frames.csv"

OUTPUT_FRAME_CSV = "outputs/possession_v2_frames.csv"
OUTPUT_SUMMARY_CSV = "outputs/possession_v2_summary.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

FPS = 25.0


# ------------------------------------------------------------
# Spatial evidence
#
# NOTE:
# This is an operational CV threshold, not a literal statement
# that a footballer physically controls a ball from 4 meters.
#
# Homography, ball height and localization error all contribute
# to pitch-coordinate error.
# ------------------------------------------------------------

CONTROL_RADIUS_M = 4.0

CONTESTED_RADIUS_M = 4.5

CONTESTED_DISTANCE_MARGIN_M = 0.75


# ------------------------------------------------------------
# Temporal possession model
# ------------------------------------------------------------

# Opponent must show control for 4 consecutive frames
# before possession changes.
SWITCH_CONFIRM_FRAMES = 4


# Ball is visible but nobody is within control radius.
#
# Usually pass / loose-ball transit.
#
# 12 frames @ 25fps = 0.48 sec
FREE_BALL_HOLD_FRAMES = 12


# Ball detector temporarily missing.
#
# Be more conservative.
#
# 5 frames @ 25fps = 0.20 sec
BALL_MISSING_HOLD_FRAMES = 5


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
# Normalize
# ============================================================

df["ball_usable_bool"] = (
    df["ball_usable"]
    .astype(str)
    .str.lower()
    .isin(
        [
            "true",
            "1"
        ]
    )
)


for column in [
    "nearest_A_distance_m",
    "nearest_B_distance_m",
    "nearest_distance_m",
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# PASS 1
# Direct frame-level evidence
# ============================================================

direct_states = []


for _, row in df.iterrows():

    ball_usable = row[
        "ball_usable_bool"
    ]


    # --------------------------------------------------------
    # Ball tracker unavailable
    # --------------------------------------------------------

    if not ball_usable:

        direct_states.append(
            "BallMissing"
        )

        continue


    dA = row[
        "nearest_A_distance_m"
    ]

    dB = row[
        "nearest_B_distance_m"
    ]


    # --------------------------------------------------------
    # No valid player geometry
    # --------------------------------------------------------

    if (
        pd.isna(dA)
        and
        pd.isna(dB)
    ):

        direct_states.append(
            "Unknown"
        )

        continue


    # --------------------------------------------------------
    # Contested
    # --------------------------------------------------------

    if (
        pd.notna(dA)
        and
        pd.notna(dB)

        and

        dA
        <=
        CONTESTED_RADIUS_M

        and

        dB
        <=
        CONTESTED_RADIUS_M

        and

        abs(
            dA - dB
        )
        <=
        CONTESTED_DISTANCE_MARGIN_M
    ):

        direct_states.append(
            "Contested"
        )

        continue


    # --------------------------------------------------------
    # Team A direct control evidence
    # --------------------------------------------------------

    if (
        pd.notna(dA)

        and

        dA
        <=
        CONTROL_RADIUS_M

        and

        (
            pd.isna(dB)
            or
            dA < dB
        )
    ):

        direct_states.append(
            "A"
        )

        continue


    # --------------------------------------------------------
    # Team B direct control evidence
    # --------------------------------------------------------

    if (
        pd.notna(dB)

        and

        dB
        <=
        CONTROL_RADIUS_M

        and

        (
            pd.isna(dA)
            or
            dB < dA
        )
    ):

        direct_states.append(
            "B"
        )

        continue


    # --------------------------------------------------------
    # Ball is visible but not controlled by anyone
    # --------------------------------------------------------

    direct_states.append(
        "FreeBall"
    )


df["direct_state"] = (
    direct_states
)


# ============================================================
# PASS 2
# Temporal state machine
# ============================================================

df["possession_v2"] = "Unknown"

df["possession_source"] = "unknown"

df["switch_candidate"] = None

df["switch_candidate_count"] = 0

df["hold_streak"] = 0


current_team = None

switch_candidate = None
switch_count = 0

hold_streak = 0

switches = 0


for i in range(len(df)):

    direct = df.loc[
        i,
        "direct_state"
    ]


    # ========================================================
    # DIRECT TEAM EVIDENCE
    # ========================================================

    if direct in [
        "A",
        "B"
    ]:

        hold_streak = 0


        # ----------------------------------------------------
        # First known possession
        # ----------------------------------------------------

        if current_team is None:

            current_team = direct

            switch_candidate = None
            switch_count = 0


            df.loc[
                i,
                "possession_v2"
            ] = current_team


            df.loc[
                i,
                "possession_source"
            ] = "direct_control"


        # ----------------------------------------------------
        # Same team as current possession
        # ----------------------------------------------------

        elif direct == current_team:

            switch_candidate = None
            switch_count = 0


            df.loc[
                i,
                "possession_v2"
            ] = current_team


            df.loc[
                i,
                "possession_source"
            ] = "direct_control"


        # ----------------------------------------------------
        # Opponent evidence
        # ----------------------------------------------------

        else:

            if (
                switch_candidate
                ==
                direct
            ):

                switch_count += 1

            else:

                switch_candidate = direct

                switch_count = 1


            # -----------------------------------------------
            # Turnover confirmed
            # -----------------------------------------------

            if (
                switch_count
                >=
                SWITCH_CONFIRM_FRAMES
            ):

                current_team = direct

                switches += 1

                switch_candidate = None
                switch_count = 0


                df.loc[
                    i,
                    "possession_v2"
                ] = current_team


                df.loc[
                    i,
                    "possession_source"
                ] = "confirmed_switch"


            # -----------------------------------------------
            # Not enough evidence yet
            # -----------------------------------------------

            else:

                df.loc[
                    i,
                    "possession_v2"
                ] = current_team


                df.loc[
                    i,
                    "possession_source"
                ] = "pending_switch"


    # ========================================================
    # CONTESTED
    # ========================================================

    elif direct == "Contested":

        hold_streak = 0

        switch_candidate = None
        switch_count = 0


        df.loc[
            i,
            "possession_v2"
        ] = "Contested"


        df.loc[
            i,
            "possession_source"
        ] = "direct_contested"


    # ========================================================
    # FREE BALL
    #
    # Ball visible but nobody has direct control.
    #
    # Keep previous team for a short period.
    # ========================================================

    elif direct == "FreeBall":

        switch_candidate = None
        switch_count = 0

        hold_streak += 1


        if (
            current_team is not None

            and

            hold_streak
            <=
            FREE_BALL_HOLD_FRAMES
        ):

            df.loc[
                i,
                "possession_v2"
            ] = current_team


            df.loc[
                i,
                "possession_source"
            ] = "carried_free_ball"


        else:

            df.loc[
                i,
                "possession_v2"
            ] = "Unknown"


            df.loc[
                i,
                "possession_source"
            ] = "free_ball_too_long"


    # ========================================================
    # BALL TRACKER MISSING
    # ========================================================

    elif direct == "BallMissing":

        switch_candidate = None
        switch_count = 0

        hold_streak += 1


        if (
            current_team is not None

            and

            hold_streak
            <=
            BALL_MISSING_HOLD_FRAMES
        ):

            df.loc[
                i,
                "possession_v2"
            ] = current_team


            df.loc[
                i,
                "possession_source"
            ] = "carried_ball_missing"


        else:

            df.loc[
                i,
                "possession_v2"
            ] = "Unknown"


            df.loc[
                i,
                "possession_source"
            ] = "ball_missing_too_long"


    # ========================================================
    # OTHER UNKNOWN
    # ========================================================

    else:

        switch_candidate = None
        switch_count = 0

        hold_streak += 1


        df.loc[
            i,
            "possession_v2"
        ] = "Unknown"


        df.loc[
            i,
            "possession_source"
        ] = "unknown_geometry"


    # ========================================================
    # Debug columns
    # ========================================================

    df.loc[
        i,
        "switch_candidate"
    ] = switch_candidate


    df.loc[
        i,
        "switch_candidate_count"
    ] = switch_count


    df.loc[
        i,
        "hold_streak"
    ] = hold_streak


# ============================================================
# Save frame-level result
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
        df[
            "possession_v2"
        ]
        ==
        "A"
    ).sum()
)


B_frames = int(
    (
        df[
            "possession_v2"
        ]
        ==
        "B"
    ).sum()
)


contested_frames = int(
    (
        df[
            "possession_v2"
        ]
        ==
        "Contested"
    ).sum()
)


unknown_frames = int(
    (
        df[
            "possession_v2"
        ]
        ==
        "Unknown"
    ).sum()
)


known_team_frames = (
    A_frames
    +
    B_frames
)


resolved_frames = (
    A_frames
    +
    B_frames
    +
    contested_frames
)


if known_team_frames > 0:

    A_share = (
        A_frames
        /
        known_team_frames
        *
        100.0
    )

    B_share = (
        B_frames
        /
        known_team_frames
        *
        100.0
    )

else:

    A_share = 0.0
    B_share = 0.0


known_coverage = (
    known_team_frames
    /
    total
    *
    100.0
)


resolved_coverage = (
    resolved_frames
    /
    total
    *
    100.0
)


summary = pd.DataFrame(
    [
        {
            "state": "A",
            "frames": A_frames,
            "seconds": A_frames / FPS,
            "share_known_team_pct":
                A_share,
        },

        {
            "state": "B",
            "frames": B_frames,
            "seconds": B_frames / FPS,
            "share_known_team_pct":
                B_share,
        },

        {
            "state": "Contested",
            "frames": contested_frames,
            "seconds":
                contested_frames
                /
                FPS,
            "share_known_team_pct":
                np.nan,
        },

        {
            "state": "Unknown",
            "frames": unknown_frames,
            "seconds":
                unknown_frames
                /
                FPS,
            "share_known_team_pct":
                np.nan,
        },
    ]
)


summary.to_csv(
    OUTPUT_SUMMARY_CSV,
    index=False
)


# ============================================================
# Print
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V2 SUMMARY"
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
    "PARAMETERS"
)

print(
    "Control radius:",
    CONTROL_RADIUS_M,
    "m"
)

print(
    "Contested radius:",
    CONTESTED_RADIUS_M,
    "m"
)

print(
    "Switch confirmation:",
    SWITCH_CONFIRM_FRAMES,
    "frames"
)

print(
    "Free-ball hold:",
    FREE_BALL_HOLD_FRAMES,
    "frames"
)

print(
    "Ball-missing hold:",
    BALL_MISSING_HOLD_FRAMES,
    "frames"
)


print("")
print(
    "DIRECT EVIDENCE"
)

print(
    df[
        "direct_state"
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
        "possession_v2"
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
    contested_frames,
    "frames"
)

print(
    "Unknown:",
    unknown_frames,
    "frames"
)


print("")
print(
    "Known team possession:"
)

print(
    "Team A:",
    f"{A_share:.1f}%"
)

print(
    "Team B:",
    f"{B_share:.1f}%"
)


print("")

print(
    "Known-team coverage:",
    f"{known_coverage:.1f}%"
)

print(
    "Resolved coverage:",
    f"{resolved_coverage:.1f}%"
)

print(
    "Confirmed possession switches:",
    switches
)


print("")
print(
    "POSSESSION SOURCE"
)

print(
    df[
        "possession_source"
    ]
    .value_counts()
    .to_string()
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
